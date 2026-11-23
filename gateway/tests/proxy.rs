use std::{convert::Infallible, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    http::{HeaderMap, Method, Request, StatusCode, Uri, header},
    routing::get,
};
use futures_util::{StreamExt, stream};
use http_body_util::BodyExt;
use tokio::{net::TcpListener, sync::Notify, task::JoinHandle, time::timeout};
use tower::ServiceExt;

struct Upstream {
    url: String,
    task: JoinHandle<()>,
}

impl Upstream {
    async fn start(router: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self { url, task }
    }
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn json(body: Body) -> serde_json::Value {
    serde_json::from_slice(&to_bytes(body, 1024 * 1024).await.unwrap()).unwrap()
}

#[tokio::test]
async fn preserves_method_encoded_path_query_body_and_accept_without_leaking_key() {
    let upstream = Upstream::start(Router::new().fallback(
        |method: Method, uri: Uri, headers: HeaderMap, body: Bytes| async move {
            Json(serde_json::json!({
                "method": method.as_str(),
                "uri": uri.to_string(),
                "body": String::from_utf8(body.to_vec()).unwrap(),
                "accept": headers.get(header::ACCEPT).unwrap().to_str().unwrap(),
                "content_type": headers.get(header::CONTENT_TYPE).unwrap().to_str().unwrap(),
                "authorization": headers.get(header::AUTHORIZATION).map(|v| v.to_str().unwrap()),
            }))
        },
    ))
    .await;
    let response = gateway::app(&format!("{}/", upstream.url), Some("test-key"))
        .unwrap()
        .oneshot(
            Request::post("/v1/files/a%2Fb?after=x%2By&limit=2")
                .header(header::AUTHORIZATION, "Bearer test-key")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::ACCEPT, "text/event-stream")
                .body(Body::from(r#"{"stream":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json(response.into_body()).await,
        serde_json::json!({
            "method": "POST",
            "uri": "/v1/files/a%2Fb?after=x%2By&limit=2",
            "body": r#"{"stream":true}"#,
            "accept": "text/event-stream",
            "content_type": "application/json",
            "authorization": null,
        })
    );
}

#[tokio::test]
async fn health_is_public_but_api_requires_the_configured_key() {
    let app = gateway::app("http://127.0.0.1:1", Some("secret")).unwrap();
    for token in [None, Some("Bearer wrong"), Some("Basic secret")] {
        let mut request = Request::get("/v1/models");
        if let Some(token) = token {
            request = request.header(header::AUTHORIZATION, token);
        }
        let response = app
            .clone()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
    }
    let response = app
        .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn sends_first_sse_event_before_upstream_finishes() {
    let release = Arc::new(Notify::new());
    let upstream_release = release.clone();
    let upstream = Upstream::start(Router::new().route(
        "/v1/chat/completions",
        get(move || {
            let release = upstream_release.clone();
            async move {
                let events = stream::once(async { Ok::<_, Infallible>("data: hello\n\n") }).chain(
                    stream::once(async move {
                        release.notified().await;
                        Ok::<_, Infallible>("data: [DONE]\n\n")
                    }),
                );
                (
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    Body::from_stream(events),
                )
            }
        }),
    ))
    .await;
    let response = timeout(
        Duration::from_secs(3),
        gateway::app(&upstream.url, None).unwrap().oneshot(
            Request::get("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        ),
    )
    .await
    .expect("la gateway doit transmettre les en-têtes avant la fin du flux")
    .unwrap();
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let mut body = response.into_body();
    let frame = timeout(Duration::from_secs(3), body.frame())
        .await
        .expect("le premier événement doit arriver sans attendre [DONE]")
        .unwrap()
        .unwrap();
    assert_eq!(frame.into_data().unwrap(), "data: hello\n\n");
    release.notify_one();
    let rest = timeout(Duration::from_secs(3), to_bytes(body, 1024))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(rest, "data: [DONE]\n\n");
}

#[tokio::test]
async fn preserves_upstream_errors_and_headers_but_removes_connection_headers() {
    let upstream = Upstream::start(Router::new().fallback(|| async {
        (
            StatusCode::TOO_MANY_REQUESTS,
            [
                ("retry-after", "12"),
                ("x-request-id", "request-42"),
                ("connection", "x-internal"),
                ("x-internal", "private"),
            ],
            Json(serde_json::json!({"error": "busy"})),
        )
    }))
    .await;
    let response = gateway::app(&upstream.url, None)
        .unwrap()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "12");
    assert_eq!(response.headers()["x-request-id"], "request-42");
    assert!(!response.headers().contains_key("connection"));
    assert!(!response.headers().contains_key("x-internal"));
    assert_eq!(json(response.into_body()).await["error"], "busy");
}

#[tokio::test]
async fn readiness_tracks_upstream_health() {
    for (health, expected) in [
        (StatusCode::OK, StatusCode::OK),
        (
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (StatusCode::FOUND, StatusCode::SERVICE_UNAVAILABLE),
    ] {
        let upstream =
            Upstream::start(Router::new().route("/health", get(move || async move { health })))
                .await;
        let response = gateway::app(&upstream.url, Some("secret"))
            .unwrap()
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
    }
}

#[tokio::test]
async fn unreachable_upstream_reports_502_for_api_and_503_for_readiness() {
    let app = gateway::app("http://127.0.0.1:1", None).unwrap();
    for (path, expected) in [
        ("/v1/models", StatusCode::BAD_GATEWAY),
        ("/readyz", StatusCode::SERVICE_UNAVAILABLE),
    ] {
        let response = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
    }
}

#[test]
fn rejects_invalid_upstream_and_empty_or_malformed_api_keys() {
    for upstream in [
        "not-a-url",
        "ftp://localhost/model",
        "http://localhost?key=value",
        "http://localhost/#fragment",
        "http://user:password@localhost",
    ] {
        assert!(gateway::app(upstream, None).is_err());
    }
    for key in ["", "   ", "key\ninvalid"] {
        assert!(gateway::app("http://localhost:8000", Some(key)).is_err());
    }
}
