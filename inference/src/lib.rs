use std::{
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
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

pub mod prefix_cache;
pub mod worker;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub stream: bool,
}

fn default_temperature() -> f64 {
    0.7
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl ChatRequest {
    pub fn prompt(&self, model: &str, max_context: usize) -> Result<String, &'static str> {
        if self.model != model {
            return Err("modèle inconnu");
        }
        if self.messages.is_empty() {
            return Err("messages ne doit pas être vide");
        }
        if self
            .max_tokens
            .is_some_and(|tokens| tokens == 0 || tokens >= max_context)
        {
            return Err("max_tokens doit être positif et inférieur au contexte");
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err("temperature doit être comprise entre 0 et 2");
        }
        let mut prompt = String::new();
        for message in &self.messages {
            if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                return Err("les rôles acceptés sont system, user et assistant");
            }
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                message.role, message.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        Ok(prompt)
    }

    /// Un budget explicite reste strict ; sans budget, utilise tout le contexte restant.
    pub fn completion_budget(
        &self,
        prompt_tokens: usize,
        max_context: usize,
    ) -> Result<usize, &'static str> {
        let remaining = max_context
            .checked_sub(prompt_tokens)
            .filter(|remaining| *remaining > 0)
            .ok_or("le prompt ne laisse aucune place pour une réponse")?;
        match self.max_tokens {
            Some(tokens) if tokens == 0 || tokens > remaining => {
                Err("prompt et réponse demandée dépassent le contexte disponible")
            }
            Some(tokens) => Ok(tokens),
            None => Ok(remaining),
        }
    }
}

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
    if streaming {
        let prefix = json!({"id":id, "object":"chat.completion.chunk", "created":created, "model":state.model});
        return streaming_response(events, prefix);
    }
    let mut text = String::new();
    while let Some(event) = events.recv().await {
        match event {
            GenerationEvent::Delta(delta) => text.push_str(&delta),
            GenerationEvent::Finished { tokens, reason } => return Json(json!({
                "id":id,"object":"chat.completion","created":created,"model":state.model,
                "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":reason}],
                "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":tokens,"total_tokens":prompt_tokens+tokens,
                    "prompt_tokens_details":{"cached_tokens":cached_tokens}}
            })).into_response(),
            GenerationEvent::Failed(failure) => return error(failure.status, failure.message),
            GenerationEvent::Started { .. } => {},
        }
    }
    error(StatusCode::SERVICE_UNAVAILABLE, "génération interrompue")
}

struct StreamState {
    events: mpsc::Receiver<GenerationEvent>,
    prefix: Value,
    sent_role: bool,
    terminal: bool,
    done: bool,
}

fn streaming_response(events: mpsc::Receiver<GenerationEvent>, prefix: Value) -> Response {
    let state = StreamState {
        events,
        prefix,
        sent_role: false,
        terminal: false,
        done: false,
    };
    let stream = stream::unfold(state, |mut state| async move {
        if state.done {
            return None;
        }
        let mut chunk = state.prefix.clone();
        let data = if !state.sent_role {
            state.sent_role = true;
            chunk["choices"] =
                json!([{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]);
            chunk.to_string()
        } else if state.terminal {
            state.done = true;
            "[DONE]".to_owned()
        } else {
            match state.events.recv().await {
                Some(GenerationEvent::Delta(text)) => {
                    chunk["choices"] =
                        json!([{"index":0,"delta":{"content":text},"finish_reason":null}]);
                    chunk.to_string()
                }
                Some(GenerationEvent::Finished { reason, .. }) => {
                    state.terminal = true;
                    chunk["choices"] = json!([{"index":0,"delta":{},"finish_reason":reason}]);
                    chunk.to_string()
                }
                event => {
                    state.terminal = true;
                    let message = match event {
                        Some(GenerationEvent::Failed(failure)) => failure.message,
                        _ => "génération interrompue".to_owned(),
                    };
                    json!({"error":{"message":message,"type":"local_inference_error"}}).to_string()
                }
            }
        };
        Some((Ok::<_, Infallible>(Event::default().data(data)), state))
    });
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
