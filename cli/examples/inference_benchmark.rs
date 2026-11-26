//! Comparaison reproductible des versions du serveur, sans modifier les poids.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::Stdio,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{Child, Command},
    task::JoinHandle,
};

#[derive(Parser)]
struct Options {
    #[arg(long, default_value = "http://127.0.0.1:9090")]
    url: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    label: String,
    #[arg(
        long,
        default_value = "release; lto=false; codegen-units=16; CUDA_COMPUTE_CAP=75"
    )]
    build_description: String,
    /// Trois répétitions des deux workloads de décodage, sans les prompts longs
    #[arg(long)]
    quick: bool,
}

struct Monitor {
    child: Child,
    rows: JoinHandle<Vec<Value>>,
}

impl Monitor {
    fn start() -> Result<Self> {
        let mut child=Command::new("nvidia-smi").args([
            "--query-gpu=power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits","--loop-ms=250"
        ]).stdout(Stdio::piped()).stderr(Stdio::null()).kill_on_drop(true).spawn()?;
        let stdout = child.stdout.take().context("télémétrie GPU absente")?;
        let rows = tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            let mut rows = Vec::new();
            while let Ok(Some(line)) = lines.next_line().await {
                let columns: Vec<_> = line
                    .split(',')
                    .filter_map(|value| value.trim().parse::<f64>().ok())
                    .collect();
                if columns.len() == 7 {
                    rows.push(json!({"power_watts":columns[0],"sm_clock_mhz":columns[1],"memory_clock_mhz":columns[2],"gpu_busy_percent":columns[3],"memory_used_mib":columns[4],"memory_total_mib":columns[5],"temperature_celsius":columns[6]}));
                }
            }
            rows
        });
        Ok(Self { child, rows })
    }
    async fn finish(mut self) -> Vec<Value> {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
        self.rows.await.unwrap_or_default()
    }
}

fn payload(prompt: &str, temperature: f64, tokens: u32) -> Value {
    json!({"model":"qwen3-4b-instruct","messages":[{"role":"user","content":prompt}],"temperature":temperature,"max_tokens":tokens})
}

async fn request(client: &reqwest::Client, base: &str, body: &Value) -> Result<Value> {
    let started = Instant::now();
    let mut request = client
        .post(format!("{base}/v1/chat/completions"))
        .json(body);
    if let Ok(key) = std::env::var("GATEWAY_API_KEY") {
        request = request.bearer_auth(key);
    }
    let response = request.send().await?;
    let status = response.status();
    let result: Value = response.json().await?;
    let seconds = started.elapsed().as_secs_f64();
    let tokens = result["usage"]["completion_tokens"].as_u64();
    Ok(
        json!({"status":status.as_u16(),"seconds":seconds,"usage":result["usage"],"finish_reason":result["choices"][0]["finish_reason"],"text":result["choices"][0]["message"]["content"],"tokens_per_second":tokens.map(|tokens|tokens as f64/seconds),"error":result.get("error")}),
    )
}

async fn repeated(
    client: &reqwest::Client,
    base: &str,
    body: &Value,
    runs: usize,
) -> Result<Value> {
    let mut results = Vec::new();
    for run in 0..runs {
        let result = request(client, base, body).await?;
        ensure!(result["status"] == 200, "échec du benchmark : {result}");
        eprintln!(
            "essai {} : {:.3} s, {:.2} tokens/s",
            run + 1,
            result["seconds"].as_f64().unwrap_or_default(),
            result["tokens_per_second"].as_f64().unwrap_or_default()
        );
        results.push(result);
    }
    let seconds: f64 = results
        .iter()
        .filter_map(|result| result["seconds"].as_f64())
        .sum();
    let tokens: u64 = results
        .iter()
        .filter_map(|result| result["usage"]["completion_tokens"].as_u64())
        .sum();
    let identical = results
        .iter()
        .all(|result| result["text"] == results[0]["text"]);
    Ok(
        json!({"request":body,"runs":results,"aggregate_tokens_per_second":tokens as f64/seconds,"mean_seconds":seconds/runs as f64,"outputs_identical":identical}),
    )
}

async fn stream(
    client: &reqwest::Client,
    base: &str,
    mut body: Value,
    cancel: bool,
) -> Result<Value> {
    body["stream"] = json!(true);
    let started = Instant::now();
    let mut request = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&body);
    if let Ok(key) = std::env::var("GATEWAY_API_KEY") {
        request = request.bearer_auth(key);
    }
    let mut response = request.send().await?.error_for_status()?;
    let mut pending = Vec::new();
    let mut first = None;
    let mut text = String::new();
    while let Some(chunk) = response.chunk().await? {
        pending.extend_from_slice(&chunk);
        if cancel && pending.windows(5).any(|bytes| bytes == b"data:") {
            tokio::time::sleep(Duration::from_millis(100)).await;
            drop(response);
            return Ok(json!({"closed_seconds":started.elapsed().as_secs_f64()}));
        }
        while let Some(end) = pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<_> = pending.drain(..=end).collect();
            let Some(data) = std::str::from_utf8(&line)?
                .trim()
                .strip_prefix("data:")
                .map(str::trim)
            else {
                continue;
            };
            if data == "[DONE]" {
                return Ok(
                    json!({"first_content_seconds":first,"seconds":started.elapsed().as_secs_f64(),"text":text}),
                );
            }
            let event: Value = serde_json::from_str(data)?;
            ensure!(event.get("error").is_none(), "erreur SSE : {event}");
            if let Some(fragment) = event["choices"][0]["delta"]["content"].as_str() {
                if !fragment.is_empty() && first.is_none() {
                    first = Some(started.elapsed().as_secs_f64());
                }
                text.push_str(fragment);
            }
        }
    }
    anyhow::bail!("flux interrompu sans [DONE]")
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = Options::parse();
    let base = options.url.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(240))
        .build()?;
    let models: Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let prompt = "Rédige en français un récit de voyage de plus de cinq cents mots. Décris les paysages, les rencontres et les sensations avec beaucoup de détails. Continue le récit sans introduction ni conclusion.";
    ensure!(
        request(&client, base, &payload(prompt, 0., 16)).await?["status"] == 200,
        "échauffement impossible"
    );
    let monitor = Monitor::start()?;
    let work=async {
        eprintln!("Décodage déterministe");
        let greedy=repeated(&client,base,&payload(prompt,0.,128),3).await?;
        ensure!(greedy["outputs_identical"] == true,"sorties déterministes instables");
        eprintln!("Décodage température 0,7");
        let sampling=repeated(&client,base,&payload(prompt,0.7,128),3).await?;
        let mut report=json!({"label":options.label,"unix_seconds":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),"endpoint":base,"models":models,"greedy":greedy,"sampling":sampling,"build":options.build_description,"method":"One warm-up, three sequential identical requests per decode workload; same requests and FP16 weights for each version."});
        if !options.quick {
            let sse=stream(&client,base,payload(prompt,0.,128),false).await?;
            ensure!(sse["text"] == report["greedy"]["runs"][0]["text"],"réponse SSE différente du JSON déterministe");
            report["greedy_sse"]=sse;
            let context="Fait neutre : le ciel est bleu et la forêt est verte.\n".repeat(200);
            let mut cold=Vec::new();
            for label in ["A","B","C"] {
                let question=format!("Document {label}.\n{context}\nRéponds uniquement par OK.");
                cold.push(request(&client,base,&payload(&question,0.,16)).await?);
            }
            report["cold_prefill"]=json!({"template":"Document <A|B|C>. + 200 neutral facts + answer OK","runs":cold});
            eprintln!("Préfixe commun");
            report["prefix_reuse"]=repeated(&client,base,&payload(&format!("Document commun.\n{context}\nRéponds uniquement par OK."),0.,16),4).await?;
            let long=format!("Code au début : PARIS7.\n{}\nCode au milieu : LYON8.\n{}\nCode à la fin : LILLE9.\nRecopie uniquement les trois codes dans leur ordre d'apparition.","Fait neutre : le ciel est bleu et la forêt est verte.\n".repeat(375),"Fait neutre : le ciel est bleu et la forêt est verte.\n".repeat(375));
            eprintln!("Long contexte et rappel de codes");
            let result=request(&client,base,&payload(&long,0.,32)).await?;
            let output=result["text"].as_str().unwrap_or_default();
            let codes_recalled=["PARIS7","LYON8","LILLE9"].iter().all(|code|output.contains(code));
            report["long_context"]=json!({"result":result,"codes_recalled":codes_recalled});
            report["after_long_context"]=request(&client,base,&payload("Réponds uniquement par 42.",0.,16)).await?;
            eprintln!("Annulation du préremplissage");
            report["cancel"]=stream(&client,base,payload(&long,0.,128),true).await?;
            report["after_cancel"]=request(&client,base,&payload("Réponds uniquement par 42.",0.,16)).await?;
        }
        Ok::<_,anyhow::Error>(report)
    }.await;
    let samples = monitor.finish().await;
    let mut report = work?;
    report["gpu_samples"] = json!(samples);
    report["ac_online"] = json!(
        std::fs::read_to_string("/sys/class/power_supply/AC/online")
            .ok()
            .map(|value| value.trim().to_owned())
    );
    std::fs::write(&options.output, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("Rapport : {}", options.output.display());
    Ok(())
}
