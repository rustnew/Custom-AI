mod providers;
pub mod webdriver;
mod wiki;

use anyhow::{Context, Result, ensure};
use clap::Args;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tokio::time::{Duration, Instant, sleep};

use webdriver::{Browser, BrowserOptions, parse_public_url};

#[derive(Args)]
pub struct Search {
    /// Recherche multi-moteurs avec secours automatique
    pub query: String,
    #[command(flatten)]
    pub browser: BrowserOptions,
}

#[derive(Args)]
pub struct Read {
    /// URL publique HTTP(S) à lire
    pub url: String,
    /// Position en octets UTF-8 dans le texte disponible
    #[arg(long, default_value_t = 0)]
    pub offset: usize,
    /// Expression exacte à rechercher avant de sélectionner un passage
    #[arg(long)]
    pub find: Option<String>,
    #[command(flatten)]
    pub browser: BrowserOptions,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Google,
    Bing,
    DuckDuckGo,
    Brave,
    BraveApi,
    WikipediaFr,
    WikipediaEn,
}

impl Engine {
    pub fn url(self, query: &str) -> Url {
        let mut url = Url::parse(match self {
            Self::Google => "https://www.google.com/search",
            Self::Bing => "https://www.bing.com/search",
            Self::DuckDuckGo => "https://html.duckduckgo.com/html/",
            Self::Brave => "https://search.brave.com/search",
            Self::BraveApi => "https://api.search.brave.com/res/v1/web/search",
            Self::WikipediaFr => "https://fr.wikipedia.org/w/index.php",
            Self::WikipediaEn => "https://en.wikipedia.org/w/index.php",
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
            _ => {}
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

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
pub struct Page {
    pub url: String,
    pub title: String,
    pub text: String,
    #[serde(default)]
    pub links: Vec<Link>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub next_offset: Option<usize>,
    #[serde(default)]
    pub total_bytes: usize,
    #[serde(default)]
    pub content_truncated: bool,
    #[serde(default)]
    pub reader: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub engine: Engine,
    pub results: Vec<Link>,
    pub error: Option<String>,
}

pub async fn search_resilient(browser: &Browser, query: &str) -> Result<Vec<SearchResult>> {
    ensure!(
        !query.trim().is_empty() && query.len() <= 1600,
        "recherche vide ou trop longue (1600 octets maximum)"
    );
    providers::search(browser, query).await
}

pub(super) async fn search_browser(
    browser: &Browser,
    engine: Engine,
    query: &str,
) -> Result<Vec<Link>> {
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
            || url.contains("/captcha")
            || url.contains("consent.google.")
            || page.title.to_lowercase().contains("captcha")
            || text.contains("captcha")
            || text.contains("unusual traffic")
            || text.contains("trafic inhabituel")
            || text.contains("confirm this search was made by a human")
            || text.contains("verify you are human")
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
                || host == "duckduckgo.com"
                || host.ends_with(".duckduckgo.com")
                || host == "search.brave.com"
                || host == "api.search.brave.com"
            {
                return None;
            }
            url.set_fragment(None);
            link.url = url.to_string();
            link.title = clipped(&link.title, 250);
            link.snippet = clipped(&link.snippet, 500);
            (!link.title.trim().is_empty() && seen.insert(link.url.clone())).then_some(link)
        })
        .take(16)
        .collect()
}

pub async fn read(browser: &Browser, url: &str) -> Result<Page> {
    read_at(browser, url, 0, None).await
}

pub async fn read_at(
    browser: &Browser,
    url: &str,
    offset: usize,
    find: Option<&str>,
) -> Result<Page> {
    eprintln!("Lecture : {url}");
    if let Some(request) = wiki::request_url(url)? {
        match wiki::read(request).await {
            Ok(mut page) => {
                page.links = clean_links(page.links);
                let (text, start, next) = passage(&page.text, offset, find)?;
                page.text = text;
                page.offset = start;
                page.next_offset = next;
                return Ok(page);
            }
            Err(error) => eprintln!(
                "Lecture API Wikipédia indisponible ({error:#}) ; essai dans le navigateur."
            ),
        }
    }
    browser.navigate(url).await?;
    let deadline = Instant::now() + Duration::from_secs(if browser.visible { 90 } else { 3 });
    let mut manual_notice = false;
    loop {
        let mut page: Page = serde_json::from_value(browser.snapshot().await?)?;
        if is_challenge(&page) {
            if !browser.visible || Instant::now() >= deadline {
                anyhow::bail!(
                    "la page demande une vérification humaine ; ce contenu n'est pas une source lue"
                );
            }
            if !manual_notice {
                eprintln!(
                    "Vérification humaine : intervenez dans le navigateur (attente maximale de 90 s)."
                );
                manual_notice = true;
            }
            sleep(Duration::from_millis(300)).await;
            continue;
        }
        if !page.text.trim().is_empty() {
            page.links = clean_links(page.links);
            page.total_bytes = page.total_bytes.max(page.text.len());
            let (text, start, next) = passage(&page.text, offset, find)?;
            page.text = text;
            page.offset = start;
            page.next_offset = next;
            return Ok(page);
        }
        ensure!(Instant::now() < deadline, "page sans texte lisible");
        sleep(Duration::from_millis(300)).await;
    }
}

fn is_challenge(page: &Page) -> bool {
    let title = page.title.to_lowercase();
    let text = page.text.to_lowercase();
    page.url.contains("/sorry/")
        || page.url.contains("consent.google.")
        || matches!(
            title.trim(),
            "captcha" | "captcha - brave search" | "just a moment..." | "access denied"
        )
        || (text.len() < 3000
            && (text.contains("verify you are human")
                || text.contains("confirm this search was made by a human")
                || text.contains("checking your browser before accessing")))
}

fn passage(
    text: &str,
    offset: usize,
    find: Option<&str>,
) -> Result<(String, usize, Option<usize>)> {
    ensure!(
        offset < text.len(),
        "offset hors du texte disponible ({} octets)",
        text.len()
    );
    let mut start = offset;
    if let Some(needle) = find {
        ensure!(
            !needle.is_empty() && needle.len() <= 200,
            "find doit contenir de 1 à 200 octets"
        );
        let found = text.match_indices(needle).find(|(index,_)| *index >= offset)
            .map(|(index,_)|index).context("expression absente du texte disponible ; essayer une autre expression ou un autre offset")?;
        start = found.saturating_sub(400).max(offset);
    }
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let excerpt = clipped(&text[start..], 4000);
    let end = start + excerpt.len();
    Ok((excerpt, start, (end < text.len()).then_some(end)))
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
        result = search_resilient(&browser, &options.query) => result,
        _ = tokio::signal::ctrl_c() => Err(anyhow::anyhow!("recherche annulée")),
    };
    let closed = browser.close().await;
    let reports = result?;
    println!("{}", serde_json::to_string_pretty(&reports)?);
    closed?;
    ensure!(
        reports.iter().any(|report| !report.results.is_empty()),
        "aucun résultat parmi les moteurs et secours ; consulter les erreurs éventuelles ci-dessus"
    );
    Ok(())
}

pub async fn run_read(options: &Read) -> Result<()> {
    parse_public_url(&options.url)?;
    let browser = Browser::start(&options.browser).await?;
    let result = tokio::select! {
        result = async {
            if options.offset == 0 && options.find.is_none() {read(&browser,&options.url).await}
            else {read_at(&browser,&options.url,options.offset,options.find.as_deref()).await}
        } => result,
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
    fn pages_are_read_in_utf8_aligned_passages_and_can_be_searched_deep_in_the_text() {
        let text = format!("{}CIBLE:42 {}", "é".repeat(3000), "texte ".repeat(1200));
        let (first, _, next) = passage(&text, 0, None).unwrap();
        assert_eq!(first.len(), 4000);
        assert_eq!(next, Some(4000));
        assert!(!first.contains("CIBLE:42"));
        let (found, start, _) = passage(&text, 0, Some("CIBLE:42")).unwrap();
        assert!(found.contains("CIBLE:42"));
        assert!(start >= 5000);
        assert!(text.is_char_boundary(start));
        assert!(passage(&text, usize::MAX, None).is_err());
        assert!(passage(&text, text.len(), None).is_err());
        assert!(passage(&text, 0, Some("ABSENT")).is_err());
        assert!(passage(&text, 0, Some("")).is_err());
    }

    #[test]
    fn a_challenge_page_is_not_accepted_as_a_read_source() {
        let challenge = Page {
            url: "https://search.brave.com/".into(),
            title: "Captcha - Brave Search".into(),
            text: "Verify you are human".into(),
            ..Default::default()
        };
        assert!(is_challenge(&challenge));
        let article = Page {
            url: "https://fr.wikipedia.org/wiki/CAPTCHA".into(),
            title: "CAPTCHA — Wikipédia".into(),
            text: "Un CAPTCHA est un test de distinction entre humains et ordinateurs.".into(),
            ..Default::default()
        };
        assert!(!is_challenge(&article));
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
