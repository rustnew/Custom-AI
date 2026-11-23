use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result, ensure};
use axum::{
    Router,
    body::{Body, Bytes},
    extract::{OriginalUri, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{any, get},
};

struct AppState {
    client: reqwest::Client,
    upstream: String,
    authorization: Option<HeaderValue>,
}

/// Construit la gateway sans ouvrir de socket, afin de pouvoir la tester.
pub fn app(upstream: &str, api_key: Option<&str>) -> Result<Router> {
    let url = reqwest::Url::parse(upstream).context("VLLM_UPSTREAM_URL invalide")?;
    ensure!(
        matches!(url.scheme(), "http" | "https")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "VLLM_UPSTREAM_URL doit être une URL HTTP(S) sans identifiants, paramètres ni fragment"
    );
    let authorization = api_key
        .map(|key| {
            ensure!(
                !key.trim().is_empty(),
                "GATEWAY_API_KEY ne doit pas être vide"
            );
            HeaderValue::from_str(&format!("Bearer {key}"))
                .context("GATEWAY_API_KEY contient des caractères interdits")
        })
        .transpose()?;
    let state = Arc::new(AppState {
        client: reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()?,
        upstream: url.as_str().trim_end_matches('/').to_owned(),
        authorization,
    });

    Ok(Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/readyz", get(ready))
        .route("/v1/{*path}", any(proxy))
        .with_state(state))
}

async fn ready(State(state): State<Arc<AppState>>) -> Response {
    match state
        .client
        .get(format!("{}/health", state.upstream))
        .timeout(Duration::from_secs(3))
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => "ok".into_response(),
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            "inference upstream not ready",
        )
            .into_response(),
    }
}

async fn proxy(
    State(state): State<Arc<AppState>>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(expected) = &state.authorization
        && headers.get(header::AUTHORIZATION) != Some(expected)
    {
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "missing or invalid bearer token",
        )
            .into_response();
    }

    // OriginalUri préserve les échappements du chemin et les paramètres d'URL.
    let path = uri.path_and_query().map_or("/", |value| value.as_str());
    let url = format!("{}{path}", state.upstream);
    let mut request = state.client.request(method, url).body(body);
    // La clé publique de la gateway reste locale ; elle n'est pas envoyée à vLLM.
    for name in [header::CONTENT_TYPE, header::ACCEPT] {
        if let Some(value) = headers.get(&name) {
            request = request.header(name, value);
        }
    }

    match request.send().await {
        Ok(upstream) => {
            let status = upstream.status();
            let mut headers = upstream.headers().clone();
            remove_connection_headers(&mut headers);
            let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
            *response.status_mut() = status;
            *response.headers_mut() = headers;
            response
        }
        Err(error) => {
            tracing::error!(%error, "upstream request failed");
            (StatusCode::BAD_GATEWAY, "inference upstream unreachable").into_response()
        }
    }
}

fn remove_connection_headers(headers: &mut HeaderMap) {
    // Connection peut nommer d'autres en-têtes valables pour une seule connexion.
    let connection_headers: Vec<_> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| header::HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }
}

/// Signaux d'arrêt partagés par le serveur autonome et la CLI locale.
pub async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.expect("signal Ctrl+C") };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("signal SIGTERM")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        _ = terminate => {},
    }
}
