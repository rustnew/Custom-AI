use std::{
    collections::VecDeque,
    convert::Infallible,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use futures_util::stream;
use serde_json::{Value, json};
use tokio::sync::mpsc;

pub mod prefix_cache;
pub mod worker;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

mod protocol;
mod tool_output;
pub use protocol::{ChatRequest, Message};
use tool_output::ToolOutput;

pub struct Job {
    pub request: ChatRequest,
    pub prompt: String,
    pub events: mpsc::Sender<GenerationEvent>,
}

impl Job {
    /// Appelé depuis le thread GPU : un client lent ne doit pas le bloquer indéfiniment.
    pub fn emit(&self, event: GenerationEvent) -> bool {
        self.emit_with_timeout(event, Duration::from_secs(5))
    }

    pub fn emit_with_timeout(&self, event: GenerationEvent, timeout: Duration) -> bool {
        match self.events.try_send(event) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Closed(_)) => false,
            Err(mpsc::error::TrySendError::Full(event)) => {
                let Ok(runtime) = tokio::runtime::Handle::try_current() else {
                    return false;
                };
                runtime
                    .block_on(self.events.send_timeout(event, timeout))
                    .is_ok()
            }
        }
    }
}

pub enum GenerationEvent {
    Started { tokens: usize, cached_tokens: usize },
    Delta(String),
    Finished { tokens: usize, reason: &'static str },
    Failed(GenerationFailure),
}

#[derive(Debug, Clone)]
pub struct GenerationFailure {
    pub status: StatusCode,
    pub message: String,
}

impl GenerationFailure {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
    pub fn engine(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

#[derive(Clone)]
struct ApiState {
    jobs: mpsc::Sender<Job>,
    model: String,
    max_context: usize,
    status: worker::WorkerStatus,
}

pub fn router(jobs: mpsc::Sender<Job>, model: String, max_context: usize) -> Router {
    router_with_status(jobs, model, max_context, worker::WorkerStatus::default())
}

pub fn router_with_status(
    jobs: mpsc::Sender<Job>,
    model: String,
    max_context: usize,
    status: worker::WorkerStatus,
) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(ApiState {
            jobs,
            model,
            max_context,
            status,
        })
}

async fn health(State(state): State<ApiState>) -> StatusCode {
    if state.jobs.is_closed() || state.status.is_recovering() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}

async fn models(State(state): State<ApiState>) -> Json<Value> {
    Json(json!({"object":"list", "data":[{
        "id":state.model, "object":"model", "created":0, "owned_by":"custom-ai-local",
        "max_model_len":state.max_context
    }]}))
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.into(), "type":"local_inference_error"}})),
    )
        .into_response()
}

async fn chat(
    State(state): State<ApiState>,
    request: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if state.status.is_recovering() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "moteur en cours de récupération",
        );
    }
    let Json(request) = match request {
        Ok(request) => request,
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };
    let prompt = match request.prompt(&state.model, state.max_context) {
        Ok(prompt) => prompt,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let streaming = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let mut output = ToolOutput::new(&request, "");
    let (sender, mut events) = mpsc::channel(16);
    if let Err(failure) = state.jobs.try_send(Job {
        request,
        prompt,
        events: sender,
    }) {
        return match failure {
            mpsc::error::TrySendError::Full(_) => {
                error(StatusCode::TOO_MANY_REQUESTS, "file d'attente GPU pleine")
            }
            mpsc::error::TrySendError::Closed(_) => {
                error(StatusCode::SERVICE_UNAVAILABLE, "moteur arrêté")
            }
        };
    }
    let (prompt_tokens, cached_tokens) = match events.recv().await {
        Some(GenerationEvent::Started {
            tokens,
            cached_tokens,
        }) => (tokens, cached_tokens),
        Some(GenerationEvent::Failed(failure)) => return error(failure.status, failure.message),
        _ => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "moteur interrompu avant la génération",
            );
        }
    };
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let id = format!(
        "chatcmpl-local-{created}-{}",
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    output.set_id(&id);
    if streaming {
        let prefix = json!({"id":id, "object":"chat.completion.chunk", "created":created, "model":state.model});
        return streaming_response(
            events,
            prefix,
            output,
            prompt_tokens,
            cached_tokens,
            include_usage,
        );
    }
    while let Some(event) = events.recv().await {
        match event {
            GenerationEvent::Delta(delta) => {
                if let Err(message) = output.push(&delta) {
                    return error(StatusCode::BAD_GATEWAY, message);
                }
            }
            GenerationEvent::Finished { tokens, reason } => {
                if let Err(message) = output.finish() {
                    return error(StatusCode::BAD_GATEWAY, message);
                }
                return Json(json!({
                    "id":id,"object":"chat.completion","created":created,"model":state.model,
                    "choices":[{"index":0,"message":output.message(),"finish_reason":output.finish_reason(reason)}],
                    "usage": usage(prompt_tokens, cached_tokens, tokens)
                })).into_response();
            }
            GenerationEvent::Failed(failure) => return error(failure.status, failure.message),
            GenerationEvent::Started { .. } => {}
        }
    }
    error(StatusCode::SERVICE_UNAVAILABLE, "génération interrompue")
}

fn usage(prompt_tokens: usize, cached_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens}
    })
}

struct StreamState {
    events: mpsc::Receiver<GenerationEvent>,
    prefix: Value,
    output: ToolOutput,
    prompt_tokens: usize,
    cached_tokens: usize,
    include_usage: bool,
    queued: VecDeque<Value>,
    sent_role: bool,
    terminal: bool,
    done: bool,
}

impl StreamState {
    fn delta(&self, delta: Value, reason: Option<&str>) -> Value {
        let mut chunk = self.prefix.clone();
        chunk["choices"] = json!([{"index":0,"delta":delta,"finish_reason":reason}]);
        if self.include_usage {
            chunk["usage"] = Value::Null;
        }
        chunk
    }

    fn fail(&mut self, message: &str) {
        self.terminal = true;
        self.queued
            .push_back(json!({"error":{"message":message,"type":"local_inference_error"}}));
    }
}

fn streaming_response(
    events: mpsc::Receiver<GenerationEvent>,
    prefix: Value,
    output: ToolOutput,
    prompt_tokens: usize,
    cached_tokens: usize,
    include_usage: bool,
) -> Response {
    let state = StreamState {
        events,
        prefix,
        output,
        prompt_tokens,
        cached_tokens,
        include_usage,
        queued: VecDeque::new(),
        sent_role: false,
        terminal: false,
        done: false,
    };
    let stream = stream::unfold(state, |mut state| async move {
        if state.done {
            return None;
        }
        let data = loop {
            if !state.sent_role {
                state.sent_role = true;
                break state
                    .delta(json!({"role":"assistant","content":""}), None)
                    .to_string();
            }
            if let Some(chunk) = state.queued.pop_front() {
                break chunk.to_string();
            }
            if state.terminal {
                state.done = true;
                break "[DONE]".to_owned();
            }
            match state.events.recv().await {
                Some(GenerationEvent::Delta(text)) => match state.output.push(&text) {
                    Ok(deltas) => {
                        for delta in deltas {
                            let chunk = state.delta(delta, None);
                            state.queued.push_back(chunk);
                        }
                    }
                    Err(message) => state.fail(message),
                },
                Some(GenerationEvent::Finished { tokens, reason }) => match state.output.finish() {
                    Ok(deltas) => {
                        for delta in deltas {
                            let chunk = state.delta(delta, None);
                            state.queued.push_back(chunk);
                        }
                        let reason = state.output.finish_reason(reason);
                        let chunk = state.delta(json!({}), Some(reason));
                        state.queued.push_back(chunk);
                        if state.include_usage {
                            let mut chunk = state.prefix.clone();
                            chunk["choices"] = json!([]);
                            chunk["usage"] =
                                usage(state.prompt_tokens, state.cached_tokens, tokens);
                            state.queued.push_back(chunk);
                        }
                        state.terminal = true;
                    }
                    Err(message) => state.fail(message),
                },
                Some(GenerationEvent::Failed(failure)) => state.fail(&failure.message),
                _ => state.fail("génération interrompue"),
            }
        };
        Some((Ok::<_, Infallible>(Event::default().data(data)), state))
    });
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
