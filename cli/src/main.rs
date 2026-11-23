use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

mod agent;
mod chat;
mod local;
mod web;

#[derive(Parser)]
#[command(
    name = "cli",
    about = "Modèle local, agent de recherche web et déploiement custom-ai"
)]
struct Cli {
    /// Répertoire contenant les manifests Kubernetes
    #[arg(long, global = true, default_value = "deploy/k8s")]
    manifests_dir: PathBuf,

    /// Namespace Kubernetes cible
    #[arg(long, global = true, default_value = "custom-ai")]
    namespace: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Discute avec le modèle local ; sans prompt, ouvre une session interactive
    Chat(chat::Chat),
    /// Agent local avec recherche Google/Bing et lecture des sources
    Agent(agent::Agent),
    /// Recherche Google/Bing sans appeler le modèle
    Search(web::Search),
    /// Lit une page publique dans Chrome ou Firefox, sans appeler le modèle
    Browse(web::Read),
    /// Lance vLLM et la gateway en local, sans Kubernetes
    Serve(local::Serve),
    /// Applique tous les manifests (namespace, vllm, gateway)
    Deploy,
    /// Affiche l'état des pods/services du namespace
    Status,
    /// Affiche les logs d'un composant (vllm | gateway)
    Logs {
        component: String,
        #[arg(short, long)]
        follow: bool,
    },
    /// Change le nombre de replicas d'un composant
    Scale { component: String, replicas: u32 },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Chat(options) => chat::run(options).await,
        Commands::Agent(options) => agent::run(options).await,
        Commands::Search(options) => web::run(options).await,
        Commands::Browse(options) => web::run_read(options).await,
        Commands::Serve(options) => local::serve(options).await,
        Commands::Deploy => deploy(&cli.manifests_dir),
        Commands::Status => status(&cli.namespace),
        Commands::Logs { component, follow } => logs(&cli.namespace, component, *follow),
        Commands::Scale {
            component,
            replicas,
        } => scale(&cli.namespace, component, *replicas),
    }
}

fn kubectl(args: &[&str]) -> Result<()> {
    let status = Command::new("kubectl")
        .args(args)
        .status()
        .context("échec de l'exécution de kubectl (binaire installé ? cluster accessible ?)")?;
    if !status.success() {
        bail!("kubectl {:?} a échoué", args);
    }
    Ok(())
}

fn deploy(manifests_dir: &Path) -> Result<()> {
    if !manifests_dir.exists() {
        bail!(
            "répertoire de manifests introuvable: {}",
            manifests_dir.display()
        );
    }
    println!(
        "Déploiement des manifests depuis {}",
        manifests_dir.display()
    );
    kubectl(&[
        "apply",
        "-f",
        manifests_dir.to_str().unwrap(),
        "--recursive",
    ])
}

fn status(namespace: &str) -> Result<()> {
    kubectl(&["get", "pods,svc,deploy", "-n", namespace, "-o", "wide"])
}

fn logs(namespace: &str, component: &str, follow: bool) -> Result<()> {
    let selector = format!("app={component}");
    let mut args = vec!["logs", "-n", namespace, "-l", selector.as_str()];
    if follow {
        args.push("-f");
    }
    kubectl(&args)
}

fn scale(namespace: &str, component: &str, replicas: u32) -> Result<()> {
    let replicas_arg = format!("--replicas={replicas}");
    kubectl(&[
        "scale",
        "deployment",
        component,
        "-n",
        namespace,
        replicas_arg.as_str(),
    ])
}
