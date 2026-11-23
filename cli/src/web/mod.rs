pub mod webdriver;

use anyhow::{Context, Result, ensure};
use clap::Args;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tokio::time::{Duration, Instant, sleep};

use webdriver::{Browser, BrowserOptions, parse_public_url};

#[derive(Args)]
pub struct Search {
    /// Recherche effectuée sur Google et Bing
    pub query: String,
    #[command(flatten)]
    pub browser: BrowserOptions,
}

#[derive(Args)]
pub struct Read {
    /// URL publique HTTP(S) à lire
    pub url: String,
    #[command(flatten)]
    pub browser: BrowserOptions,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Google,
    Bing,
}

impl Engine {
    pub fn url(self, query: &str) -> Url {
        let mut url = Url::parse(match self {
            Self::Google => "https://www.google.com/search",
            Self::Bing => "https://www.bing.com/search",
        })
        .expect("URL constante valide");
        url.query_pairs_mut().append_pair("q", query);
        match self {
            Self::Google => {
                url.query_pairs_mut()
                    .append_pair("num", "5")
                    .append_pair("hl", "fr");
            }
            Self::Bing => {
                url.query_pairs_mut()
                    .append_pair("count", "5")
                    .append_pair("setlang", "fr");
            }
        }
        url
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Link {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Page {
    pub url: String,
    pub title: String,
    pub text: String,
    #[serde(default)]
    pub links: Vec<Link>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub engine: Engine,
    pub results: Vec<Link>,
    pub error: Option<String>,
}

pub async fn search_both(browser: &Browser, query: &str) -> Result<Vec<SearchResult>> {
    ensure!(
        !query.trim().is_empty() && query.len() <= 1600,
        "recherche vide ou trop longue (1600 octets maximum)"
    );
    let mut reports = Vec::new();
    for engine in [Engine::Google, Engine::Bing] {
        eprintln!("Recherche {engine:?} : {query}");
        match search(browser, engine, query).await {
            Ok(results) => reports.push(SearchResult {
                engine,
                results,
                error: None,
            }),
            Err(error) => reports.push(SearchResult {
                engine,
                results: Vec::new(),
                error: Some(format!("{error:#}")),
            }),
        }
    }
    Ok(reports)
}

async fn search(browser: &Browser, engine: Engine, query: &str) -> Result<Vec<Link>> {
    browser.navigate(engine.url(query).as_str()).await?;
    let deadline = Instant::now() + Duration::from_secs(if browser.visible { 90 } else { 5 });
    let mut manual_notice = false;
    loop {
        let page: Page = serde_json::from_value(browser.snapshot().await?)
            .context("lecture des résultats impossible")?;
        let links = clean_links(page.links);
        if !links.is_empty() {
            return Ok(links);
        }
        let text = page.text.to_lowercase();
        let url = page.url.to_lowercase();
        if url.contains("/sorry/")
            || url.contains("consent.google.")
            || text.contains("captcha")
            || text.contains("unusual traffic")
            || text.contains("trafic inhabituel")
        {
            if !browser.visible || Instant::now() >= deadline {
                anyhow::bail!(
                    "{engine:?} demande un CAPTCHA ou un consentement ; relancer avec --visible pour le traiter manuellement"
                );
            }
            if !manual_notice {
                eprintln!(
                    "{engine:?} : validez manuellement le CAPTCHA/consentement dans la fenêtre (attente maximale de 90 s)."
                );
                manual_notice = true;
            }
        } else if text.contains("there are no results for") || text.contains("aucun résultat pour")
        {
            return Ok(Vec::new());
        } else if Instant::now() >= deadline {
            anyhow::bail!(
                "{engine:?} : aucun résultat lisible (page bloquée ou sélecteurs modifiés), titre : {}",
                page.title
            );
        }
        sleep(Duration::from_millis(300)).await;
    }
}

pub fn clean_links(links: Vec<Link>) -> Vec<Link> {
    let mut seen = HashSet::new();
    links
        .into_iter()
        .filter_map(|mut link| {
            let mut url = parse_public_url(&link.url).ok()?;
            let host = url.host_str()?;
            if host == "google.com"
                || host.ends_with(".google.com")
                || host == "bing.com"
                || host.ends_with(".bing.com")
            {
                return None;
            }
            url.set_fragment(None);
            link.url = url.to_string();
            link.title = clipped(&link.title, 250);
            link.snippet = clipped(&link.snippet, 500);
            (!link.title.trim().is_empty() && seen.insert(link.url.clone())).then_some(link)
        })
        .take(5)
        .collect()
}

pub async fn read(browser: &Browser, url: &str) -> Result<Page> {
    eprintln!("Lecture : {url}");
    browser.navigate(url).await?;
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let mut page: Page = serde_json::from_value(browser.snapshot().await?)?;
        if !page.text.trim().is_empty() {
            page.links.clear();
            page.text = clipped(&page.text, 4000);
            return Ok(page);
        }
        ensure!(Instant::now() < deadline, "page sans texte lisible");
        sleep(Duration::from_millis(300)).await;
    }
}

pub fn clipped(text: &str, bytes: usize) -> String {
    let mut end = text.len().min(bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

pub async fn run(options: &Search) -> Result<()> {
    ensure!(
        !options.query.trim().is_empty() && options.query.len() <= 1600,
        "recherche vide ou trop longue"
    );
    let browser = Browser::start(&options.browser).await?;
    let result = tokio::select! {
        result = search_both(&browser, &options.query) => result,
        _ = tokio::signal::ctrl_c() => Err(anyhow::anyhow!("recherche annulée")),
    };
    let closed = browser.close().await;
    let reports = result?;
    println!("{}", serde_json::to_string_pretty(&reports)?);
    closed?;
    ensure!(
        reports.iter().any(|report| !report.results.is_empty()),
        "aucun résultat sur les deux moteurs ; consulter les erreurs éventuelles ci-dessus"
    );
    Ok(())
}

pub async fn run_read(options: &Read) -> Result<()> {
    parse_public_url(&options.url)?;
    let browser = Browser::start(&options.browser).await?;
    let result = tokio::select! {
        result = read(&browser, &options.url) => result,
        _ = tokio::signal::ctrl_c() => Err(anyhow::anyhow!("lecture annulée")),
    };
    let closed = browser.close().await;
    println!("{}", serde_json::to_string_pretty(&result?)?);
    closed?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_are_encoded_without_becoming_url_parameters() {
        for engine in [Engine::Google, Engine::Bing] {
            let url = engine.url("Rust & français + #code ? q=secret");
            assert_eq!(
                url.query_pairs().find(|(key, _)| key == "q").unwrap().1,
                "Rust & français + #code ? q=secret"
            );
            assert!(url.fragment().is_none());
        }
    }

    #[test]
    fn search_links_reject_unsafe_destinations_and_duplicates() {
        let links = [
            "https://www.rust-lang.org/#a",
            "https://www.rust-lang.org/#b",
            "file:///etc/passwd",
            "http://127.0.0.1/",
            "https://www.google.com/preferences",
            "https://user:password@example.com/",
        ]
        .into_iter()
        .map(|url| Link {
            title: "Rust".into(),
            url: url.into(),
            snippet: String::new(),
        })
        .collect();
        let result = clean_links(links);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url, "https://www.rust-lang.org/");
    }

    #[test]
    fn clipping_preserves_utf8() {
        assert_eq!(clipped("école", 1), "");
        assert_eq!(clipped("école", 3), "éc");
    }

    #[test]
    fn refuses_local_and_non_http_urls() {
        for url in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "http://localhost/",
            "http://127.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://10.0.0.2/",
            "http://169.254.169.254/",
            "http://100.64.0.1/",
            "http://192.168.1.1/",
            "http://user:pass@example.com/",
        ] {
            assert!(parse_public_url(url).is_err(), "{url}");
        }
        assert!(parse_public_url("https://www.rust-lang.org/").is_ok());
    }
}
