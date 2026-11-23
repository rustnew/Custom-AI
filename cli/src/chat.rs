use std::{
    io::{self, Write},
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use serde_json::{Value, json};

#[derive(Args)]
pub struct Chat {
    /// Question à envoyer ; sans argument, démarre une conversation interactive
    #[arg(conflicts_with = "file")]
    prompt: Option<String>,
    /// Lit un prompt long ou multiligne depuis un fichier UTF-8
    #[arg(long)]
    file: Option<PathBuf>,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    url: String,
    #[arg(long, default_value = "qwen3-4b-instruct")]
    model: String,
    /// Limite de sortie ; par défaut, le moteur utilise le contexte restant
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    max_tokens: Option<u32>,
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
}

pub async fn run(options: &Chat) -> Result<()> {
    ensure!(
        options.temperature.is_finite() && (0.0..=2.0).contains(&options.temperature),
        "temperature doit être comprise entre 0 et 2"
    );
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let mut messages = Vec::new();
    let file_prompt = options
        .file
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path)
                .with_context(|| format!("lecture du prompt impossible : {}", path.display()))
        })
        .transpose()?;
    if let Some(prompt) = options.prompt.as_deref().or(file_prompt.as_deref()) {
        ensure!(!prompt.trim().is_empty(), "le prompt ne doit pas être vide");
        messages.push(json!({"role":"user","content":prompt}));
        complete(&client, options, &messages).await?;
        return Ok(());
    }
    println!("Écrivez votre question puis Entrée. /reset efface la conversation ; /quit quitte.");
    loop {
        print!("Vous > ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let prompt = line.trim();
        match prompt {
            "/quit" | "/exit" => break,
            "/reset" => {
                messages.clear();
                println!("Conversation effacée.");
                continue;
            }
            "" => continue,
            _ => {}
        }
        messages.push(json!({"role":"user","content":prompt}));
        print!("Modèle > ");
        io::stdout().flush()?;
        match complete(&client, options, &messages).await {
            Ok(answer) => messages.push(json!({"role":"assistant","content":answer})),
            Err(error) => {
                messages.pop();
                eprintln!("\n{error:#}");
            }
        }
    }
    Ok(())
}

async fn complete(client: &reqwest::Client, options: &Chat, messages: &[Value]) -> Result<String> {
    let endpoint = format!("{}/v1/chat/completions", options.url.trim_end_matches('/'));
    let mut body = json!({
        "model":options.model,"messages":messages,
        "temperature":options.temperature,"stream":true
    });
    if let Some(max_tokens) = options.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    let mut request = client.post(endpoint).json(&body);
    if let Ok(key) = std::env::var("GATEWAY_API_KEY") {
        request = request.bearer_auth(key);
    }
    let mut response = request
        .send()
        .await
        .context("serveur inaccessible ; lancer ./target/release/inference")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await?;
        bail!("API {status} : {body}");
    }
    let mut parser = EventParser::default();
    let mut answer = String::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("lecture du flux interrompue")?
    {
        for event in parser.feed(&chunk)? {
            match event {
                ChatEvent::Content(text) => {
                    print!("{text}");
                    io::stdout().flush()?;
                    answer.push_str(&text);
                }
                ChatEvent::Done => {
                    println!();
                    return Ok(answer);
                }
            }
        }
    }
    bail!("réponse interrompue avant [DONE] ; réessayer ou utiliser /reset")
}

#[derive(Debug, PartialEq)]
enum ChatEvent {
    Content(String),
    Done,
}

#[derive(Default)]
struct EventParser {
    pending: Vec<u8>,
}

impl EventParser {
    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ChatEvent>> {
        self.pending.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let bytes: Vec<_> = self.pending.drain(..=end).collect();
            let line = std::str::from_utf8(&bytes)?.trim_end();
            let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                events.push(ChatEvent::Done);
                break;
            }
            let event: Value = serde_json::from_str(data).context("événement SSE invalide")?;
            if let Some(error) = event.get("error") {
                bail!("erreur du moteur : {error}");
            }
            if let Some(content) = event["choices"][0]["delta"]["content"].as_str() {
                events.push(ChatEvent::Content(content.to_owned()));
            }
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_french_characters_split_across_http_chunks_and_handles_done() {
        let data = "data: {\"choices\":[{\"delta\":{\"content\":\"ça\"}}]}\r\n\r\ndata: [DONE]\n\n"
            .as_bytes();
        let mut parser = EventParser::default();
        let mut events = Vec::new();
        for byte in data {
            events.extend(parser.feed(&[*byte]).unwrap());
        }
        assert_eq!(
            events,
            vec![ChatEvent::Content("ça".into()), ChatEvent::Done]
        );
    }

    #[test]
    fn reports_server_errors_inside_an_sse_stream() {
        let mut parser = EventParser::default();
        assert!(
            parser
                .feed(b"data: {\"error\":{\"message\":\"GPU unavailable\"}}\n\n")
                .unwrap_err()
                .to_string()
                .contains("GPU unavailable")
        );
    }
}
