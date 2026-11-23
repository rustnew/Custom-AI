use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use inference::{GenerationEvent, Job};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

fn request(stream: bool) -> Request<Body> {
    Request::post("/v1/chat/completions").header("content-type", "application/json")
        .body(Body::from(json!({"model":"qwen3-4b-instruct","messages":[{"role":"user","content":"Bonjour"}],"temperature":0,"stream":stream}).to_string())).unwrap()
}

fn app() -> (axum::Router, mpsc::Receiver<Job>) {
    let (sender, receiver) = mpsc::channel(1);
    (
        inference::router(sender, "qwen3-4b-instruct".into(), 8192),
        receiver,
    )
}

#[tokio::test]
async fn json_response_preserves_french_text_finish_reason_and_usage() {
    let (app, mut jobs) = app();
    let worker = tokio::spawn(async move {
        let job = jobs.recv().await.unwrap();
        job.events
            .send(GenerationEvent::Started {
                tokens: 9,
                cached_tokens: 0,
            })
            .await
            .unwrap();
        job.events
            .send(GenerationEvent::Delta("Bonjour, ça va !".into()))
            .await
            .unwrap();
        job.events
            .send(GenerationEvent::Finished {
                tokens: 6,
                reason: "stop",
            })
            .await
            .unwrap();
    });
    let response = app.oneshot(request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 10000).await.unwrap()).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Bonjour, ça va !");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 15);
    worker.await.unwrap();
}

#[tokio::test]
async fn streaming_sends_role_content_finish_and_done() {
    let (app, mut jobs) = app();
    let worker = tokio::spawn(async move {
        let job = jobs.recv().await.unwrap();
        for event in [
            GenerationEvent::Started {
                tokens: 9,
                cached_tokens: 0,
            },
            GenerationEvent::Delta("français".into()),
            GenerationEvent::Finished {
                tokens: 3,
                reason: "length",
            },
        ] {
            job.events.send(event).await.unwrap();
        }
    });
    let response = app.oneshot(request(true)).await.unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let bytes = to_bytes(response.into_body(), 10000).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(body.contains("\"role\":\"assistant\""));
    assert!(body.contains("français"));
    assert!(body.contains("\"finish_reason\":\"length\""));
    assert!(body.ends_with("data: [DONE]\n\n"));
    worker.await.unwrap();
}

#[tokio::test]
async fn interrupted_worker_emits_stream_error_instead_of_a_successful_finish() {
    let (app, mut jobs) = app();
    let worker = tokio::spawn(async move {
        let job = jobs.recv().await.unwrap();
        job.events
            .send(GenerationEvent::Started {
                tokens: 9,
                cached_tokens: 0,
            })
            .await
            .unwrap();
    });
    let response = app.oneshot(request(true)).await.unwrap();
    let body = String::from_utf8(
        to_bytes(response.into_body(), 10000)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(body.contains("génération interrompue"));
    assert!(!body.contains("\"finish_reason\":\"stop\""));
    worker.await.unwrap();
}

#[tokio::test]
async fn dead_worker_is_unhealthy_and_requests_return_service_unavailable() {
    let (app, jobs) = app();
    drop(jobs);
    let health = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        app.oneshot(request(false)).await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn invalid_request_is_rejected_before_it_reaches_the_gpu() {
    let (app, mut jobs) = app();
    let invalid = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model":"unknown","messages":[]}).to_string(),
        ))
        .unwrap();
    assert_eq!(
        app.oneshot(invalid).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    assert!(jobs.try_recv().is_err());
}

#[test]
fn chat_template_preserves_message_roles_and_validates_limits() {
    let mut request: inference::ChatRequest = serde_json::from_value(json!({
        "model":"qwen3-4b-instruct", "messages":[{"role":"system","content":"Sois bref."},{"role":"user","content":"Bonjour !"}]
    })).unwrap();
    assert_eq!(
        request.prompt("qwen3-4b-instruct", 8192).unwrap(),
        "<|im_start|>system\nSois bref.<|im_end|>\n<|im_start|>user\nBonjour !<|im_end|>\n<|im_start|>assistant\n"
    );
    for temperature in [-1.0, 3.0, f64::NAN] {
        request.temperature = temperature;
        assert!(request.prompt("qwen3-4b-instruct", 8192).is_err());
    }
    request.temperature = 0.0;
    request.max_tokens = Some(8192);
    assert!(request.prompt("qwen3-4b-instruct", 8192).is_err());
}

#[test]
fn default_output_uses_remaining_context_and_explicit_limits_remain_strict() {
    let mut request: inference::ChatRequest = serde_json::from_value(json!({
        "model":"qwen3-4b-instruct", "messages":[{"role":"user","content":"Bonjour"}]
    }))
    .unwrap();
    assert_eq!(request.completion_budget(10, 32768).unwrap(), 32758);
    assert_eq!(request.completion_budget(32767, 32768).unwrap(), 1);
    assert!(request.completion_budget(32768, 32768).is_err());
    assert!(request.completion_budget(usize::MAX, 32768).is_err());
    request.max_tokens = Some(128);
    assert_eq!(request.completion_budget(10, 32768).unwrap(), 128);
    assert!(request.completion_budget(32700, 32768).is_err());
    request.max_tokens = Some(0);
    assert!(request.completion_budget(10, 32768).is_err());
}

#[tokio::test]
async fn pre_generation_engine_failure_is_reported_as_500_and_valid_json() {
    let (app, mut jobs) = app();
    let worker = tokio::spawn(async move {
        let job = jobs.recv().await.unwrap();
        job.events
            .send(GenerationEvent::Failed(
                inference::GenerationFailure::engine("simulated failure"),
            ))
            .await
            .unwrap();
    });
    let response = app.oneshot(request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 10000).await.unwrap()).unwrap();
    assert_eq!(body["error"]["message"], "simulated failure");
    worker.await.unwrap();
}

#[tokio::test]
async fn full_queue_rejects_new_requests_instead_of_allocating_more_jobs() {
    let (sender, _queue) = mpsc::channel(1);
    let (events, _receiver) = mpsc::channel(4);
    sender
        .send(Job {
            request: serde_json::from_value(json!({"model":"qwen3-4b-instruct","messages":[]}))
                .unwrap(),
            prompt: String::new(),
            events,
        })
        .await
        .unwrap();
    let app = inference::router(sender, "qwen3-4b-instruct".into(), 32768);
    assert_eq!(
        app.oneshot(request(false)).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );
}

#[tokio::test]
async fn dropping_a_request_closes_the_generation_receiver() {
    let (app, mut queue) = app();
    let request = tokio::spawn(app.oneshot(request(false)));
    let job = queue.recv().await.unwrap();
    request.abort();
    let _ = request.await;
    tokio::time::timeout(std::time::Duration::from_secs(1), job.events.closed())
        .await
        .unwrap();
}
