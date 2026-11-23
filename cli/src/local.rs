use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    path::PathBuf,
    process::Stdio,
};

use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use tokio::{net::TcpListener, process::Command};

#[derive(serde::Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Args)]
pub struct Serve {
    /// Dossier local Hugging Face contenant config.json et les poids safetensors
    #[arg(
        long,
        default_value = "/home/rustspeak/neurax-ai-lab/models/Qwen3-4B-Instruct-2507"
    )]
    model: PathBuf,

    /// Exécutable vLLM (par exemple /tmp/custom-ai-runtime/bin/vllm)
    #[arg(long, default_value = "vllm")]
    vllm_bin: PathBuf,

    /// Adresse de la gateway publique
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Port local du moteur vLLM
    #[arg(long, default_value_t = 8000, value_parser = clap::value_parser!(u16).range(1..))]
    upstream_port: u16,

    /// Nom du modèle utilisé dans les requêtes OpenAI
    #[arg(long, default_value = "qwen3-4b-instruct")]
    served_model_name: String,

    /// Contexte maximal ; réduit pour démarrer sur la Quadro RTX 5000 de 16 Go
    #[arg(long, default_value_t = 8192, value_parser = clap::value_parser!(u32).range(1..))]
    max_model_len: u32,
}

pub async fn serve(options: &Serve) -> Result<()> {
    let model = options
        .model
        .canonicalize()
        .context("dossier du modèle introuvable")?;
    for filename in ["config.json", "tokenizer_config.json", "tokenizer.json"] {
        ensure!(
            model.join(filename).is_file(),
            "fichier du modèle absent : {filename}"
        );
    }
    let index: WeightIndex = serde_json::from_slice(
        &std::fs::read(model.join("model.safetensors.index.json"))
            .context("index des poids safetensors introuvable")?,
    )
    .context("index des poids invalide")?;
    let weight_map = index.weight_map;
    ensure!(!weight_map.is_empty(), "l'index des poids est vide");
    let shards: HashSet<_> = weight_map.values().collect();
    ensure!(!shards.is_empty(), "aucun fichier de poids dans l'index");
    for shard in shards {
        ensure!(
            model
                .join(shard)
                .metadata()
                .is_ok_and(|meta| meta.is_file() && meta.len() > 0),
            "fichier de poids absent ou vide : {shard}"
        );
    }
    ensure!(
        options.bind.port() != options.upstream_port,
        "la gateway et vLLM doivent utiliser des ports différents"
    );
    let upstream = format!("http://127.0.0.1:{}", options.upstream_port);
    let api_key = std::env::var("GATEWAY_API_KEY").ok();
    let app = gateway::app(&upstream, api_key.as_deref())?;
    let listener = TcpListener::bind(options.bind)
        .await
        .with_context(|| format!("adresse de la gateway indisponible : {}", options.bind))?;
    // Détecte un moteur existant avant de réserver le GPU pour un second processus.
    let engine_listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, options.upstream_port))
        .await
        .context("port vLLM déjà occupé ; choisir --upstream-port")?;
    drop(engine_listener);

    let mut engine = Command::new(&options.vllm_bin)
        .args(["serve"])
        .arg(&model)
        .args(["--host", "127.0.0.1", "--port"])
        .arg(options.upstream_port.to_string())
        .args(["--served-model-name", &options.served_model_name])
        .args(["--dtype", "float16", "--gpu-memory-utilization", "0.90"])
        .arg("--max-model-len")
        .arg(options.max_model_len.to_string())
        .args([
            "--max-num-seqs",
            "4",
            "--enforce-eager",
            "--disable-frontend-multiprocessing",
        ])
        // Profil Turing sm_75 : V0 + XFormers, sans FlashAttention ni bfloat16.
        .env("VLLM_USE_V1", "0")
        .env("VLLM_ATTENTION_BACKEND", "XFORMERS")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("VLLM_NO_USAGE_STATS", "1")
        .env("DO_NOT_TRACK", "1")
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| {
            format!(
                "impossible de lancer {} ; installer vLLM puis préciser --vllm-bin",
                options.vllm_bin.display()
            )
        })?;

    println!("Modèle : {}", model.display());
    println!("Gateway : http://{}", listener.local_addr()?);
    println!(
        "Disponibilité : http://{}/readyz (503 pendant le chargement)",
        listener.local_addr()?
    );
    println!("Ctrl+C pour arrêter le moteur et la gateway.");
    let server = axum_server(listener, app);
    let result = tokio::select! {
        result = server => result,
        status = engine.wait() => {
            bail!("vLLM s'est arrêté : {} ; consulter ses journaux ci-dessus", status?);
        }
    };
    if engine.try_wait()?.is_none() {
        engine.kill().await.context("échec de l'arrêt de vLLM")?;
    }
    result
}

async fn axum_server(listener: TcpListener, app: axum::Router) -> Result<()> {
    axum::serve(listener, app)
        .with_graceful_shutdown(gateway::shutdown_signal())
        .await
        .context("arrêt de la gateway")
}
