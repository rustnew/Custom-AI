use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let upstream =
        std::env::var("VLLM_UPSTREAM_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".into());
    let bind_addr = std::env::var("GATEWAY_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let api_key = std::env::var("GATEWAY_API_KEY").ok();
    let app = gateway::app(&upstream, api_key.as_deref())?;
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("impossible d'écouter sur {bind_addr}"))?;

    tracing::info!("gateway listening on {}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(gateway::shutdown_signal())
        .await?;
    Ok(())
}
