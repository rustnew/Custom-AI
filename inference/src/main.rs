mod model;

use std::{future::IntoFuture, net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
};

#[derive(Parser)]
#[command(about = "Qwen3 local sur GPU CUDA avec Candle et la gateway Rust")]
struct Args {
    #[arg(
        long,
        default_value = "/home/rustspeak/neurax-ai-lab/models/Qwen3-4B-Instruct-2507"
    )]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    #[arg(long, default_value_t = 8000, value_parser = clap::value_parser!(u16).range(1..))]
    upstream_port: u16,
    #[arg(long, default_value = "qwen3-4b-instruct")]
    served_model_name: String,
    #[arg(long, default_value_t = 28672, value_parser = clap::value_parser!(u32).range(2..=262144))]
    max_model_len: u32,
    /// Tokens de préfixe réutilisables ; 0 désactive le cache pour comparaison
    #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(0..=8192))]
    prefix_cache_tokens: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.bind.port() != args.upstream_port,
        "ports gateway et moteur identiques"
    );
    let upstream = format!("127.0.0.1:{}", args.upstream_port);
    let key = std::env::var("GATEWAY_API_KEY").ok();
    let gateway = gateway::app(&format!("http://{upstream}"), key.as_deref())?;
    let public_listener = TcpListener::bind(args.bind)
        .await
        .context("port gateway indisponible")?;
    let engine_listener = TcpListener::bind(&upstream)
        .await
        .context("port moteur indisponible")?;
    let (jobs, mut queue) = mpsc::channel::<inference::Job>(8);
    let status = inference::worker::WorkerStatus::default();
    let api = inference::router_with_status(
        jobs,
        args.served_model_name,
        args.max_model_len as usize,
        status.clone(),
    );
    let (started, ready) = oneshot::channel();
    let mut worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut engine = model::Engine::load(
            &args.model,
            args.max_model_len as usize,
            args.prefix_cache_tokens as usize,
        )?;
        let _ = started.send(());
        inference::worker::run_worker(&mut engine, &mut queue, &status)
            .map_err(anyhow::Error::msg)?;
        Ok(())
    });
    if ready.await.is_err() {
        worker.await.context("arrêt du thread GPU")??;
        bail!("le moteur n'a pas terminé son initialisation");
    }
    println!(
        "Modèle prêt. Gateway : http://{}",
        public_listener.local_addr()?
    );
    println!("API : /v1/models, /v1/chat/completions ; Ctrl+C pour arrêter.");
    let public_server = axum::serve(public_listener, gateway).into_future();
    let engine_server = axum::serve(engine_listener, api).into_future();
    tokio::select! {
        result = public_server => result.context("gateway interrompue")?,
        result = engine_server => result.context("API moteur interrompue")?,
        result = &mut worker => { result.context("thread GPU interrompu")??; bail!("moteur arrêté"); },
        () = gateway::shutdown_signal() => {},
    }
    Ok(())
}
