use anyhow::{Context, Result, ensure};
use reqwest::{Client, Url};
use serde_json::Value;
use std::time::Duration;

use super::{Engine, Link, SearchResult, clean_links, webdriver::Browser};

trait Provider {
    async fn query(&self, engine: Engine, query: &str) -> Result<Vec<Link>>;
}

struct Providers<'a> {
    browser: &'a Browser,
    client: Client,
    brave_key: Option<String>,
}

impl Provider for Providers<'_> {
    async fn query(&self, engine: Engine, query: &str) -> Result<Vec<Link>> {
        match engine {
            Engine::WikipediaFr | Engine::WikipediaEn => {
                let language = if matches!(engine, Engine::WikipediaFr) {
                    "fr"
                } else {
                    "en"
                };
                let url = Url::parse_with_params(
                    &format!("https://{language}.wikipedia.org/w/api.php"),
                    &[
                        ("action", "query"),
                        ("list", "search"),
                        ("srsearch", query),
                        ("format", "json"),
                        ("srlimit", "5"),
                        ("utf8", "1"),
                    ],
                )?;
                wikipedia_results(&json_response(self.client.get(url)).await?, language)
            }
            Engine::BraveApi => {
                ensure!(
                    query.chars().count() <= 600 && query.split_whitespace().count() <= 75,
                    "Brave API : requête limitée à 600 caractères et 75 mots"
                );
                let key = self
                    .brave_key
                    .as_ref()
                    .context("BRAVE_SEARCH_API_KEY absente")?;
                let url = Url::parse_with_params(
                    "https://api.search.brave.com/res/v1/web/search",
                    &[("q", query), ("count", "5")],
                )?;
                brave_results(
                    &json_response(self.client.get(url).header("X-Subscription-Token", key))
                        .await?,
                )
            }
            _ => {
                if let Some(error) = self.browser.blocked_engine(engine) {
                    anyhow::bail!("moteur déjà bloqué pendant cette session : {error}");
                }
                let result = super::search_browser(self.browser, engine, query).await;
                if let Err(error) = &result {
                    let message = error.to_string();
                    if message.contains("CAPTCHA") || message.contains("429") {
                        self.browser.block_engine(engine, &message);
                    }
                }
                result
            }
        }
    }
}

pub async fn search(browser: &Browser, query: &str) -> Result<Vec<SearchResult>> {
    let brave_key = std::env::var("BRAVE_SEARCH_API_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let use_brave = brave_key.is_some();
    let providers = Providers {
        browser,
        brave_key,
        client: Client::builder()
            .user_agent("Custom-AI/0.1 (https://github.com/rustnew/Custom-AI)")
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .build()?,
    };
    search_with(&providers, query, use_brave).await
}

async fn attempt<P: Provider>(provider: &P, engine: Engine, query: &str) -> SearchResult {
    eprintln!("Recherche {engine:?} : {query}");
    match provider.query(engine, query).await {
        Ok(results) => SearchResult {
            engine,
            results: clean_links(results).into_iter().take(5).collect(),
            error: None,
        },
        Err(error) => SearchResult {
            engine,
            results: vec![],
            error: Some(format!("{error:#}")),
        },
    }
}

async fn search_with<P: Provider>(
    provider: &P,
    query: &str,
    use_brave: bool,
) -> Result<Vec<SearchResult>> {
    let mut reports = Vec::new();
    if use_brave {
        let report = attempt(provider, Engine::BraveApi, query).await;
        let useful = !report.results.is_empty();
        reports.push(report);
        if useful {
            return Ok(reports);
        }
    }
    for engine in [Engine::Google, Engine::Bing] {
        reports.push(attempt(provider, engine, query).await);
    }
    if !reports.iter().any(|report| !report.results.is_empty()) {
        for engine in [Engine::DuckDuckGo, Engine::Brave] {
            let report = attempt(provider, engine, query).await;
            let useful = !report.results.is_empty();
            reports.push(report);
            if useful {
                break;
            }
        }
    }
    // Ce secours est encyclopédique : il ne remplace pas un index du Web général.
    if !reports.iter().any(|report| !report.results.is_empty()) {
        for engine in [Engine::WikipediaFr, Engine::WikipediaEn] {
            reports.push(attempt(provider, engine, query).await);
        }
    }
    Ok(reports)
}

pub(super) async fn json_response(request: reqwest::RequestBuilder) -> Result<Value> {
    let mut response = request
        .send()
        .await
        .context("API de recherche inaccessible")?;
    ensure!(
        response.status().is_success(),
        "API de recherche HTTP {}",
        response.status()
    );
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            bytes.len() + chunk.len() <= 1024 * 1024,
            "réponse de recherche trop volumineuse"
        );
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).context("réponse JSON de recherche invalide")
}

fn wikipedia_results(value: &Value, language: &str) -> Result<Vec<Link>> {
    ensure!(
        value.get("error").is_none(),
        "erreur de recherche Wikipédia"
    );
    value["query"]["search"]
        .as_array()
        .context("résultats Wikipédia absents")?
        .iter()
        .take(5)
        .map(|item| {
            let title = item["title"].as_str().context("titre Wikipédia absent")?;
            let id = item["pageid"].as_u64().context("pageid Wikipédia absent")?;
            Ok(Link {
                title: title.into(),
                url: format!("https://{language}.wikipedia.org/?curid={id}"),
                snippet: plain_snippet(item["snippet"].as_str().unwrap_or_default()),
            })
        })
        .collect()
}

fn brave_results(value: &Value) -> Result<Vec<Link>> {
    value["web"]["results"]
        .as_array()
        .context("résultats Web Brave absents")?
        .iter()
        .take(5)
        .map(|item| {
            Ok(Link {
                title: item["title"].as_str().context("titre Brave absent")?.into(),
                url: item["url"].as_str().context("URL Brave absente")?.into(),
                snippet: plain_snippet(item["description"].as_str().unwrap_or_default()),
            })
        })
        .collect()
}

fn plain_snippet(value: &str) -> String {
    let mut in_tag = false;
    value
        .chars()
        .filter(|character| match character {
            '<' => {
                in_tag = true;
                false
            }
            '>' => {
                in_tag = false;
                false
            }
            _ => !in_tag,
        })
        .collect::<String>()
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#039;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    struct Fake {
        calls: Mutex<Vec<String>>,
        success: &'static str,
    }
    impl Provider for Fake {
        async fn query(&self, engine: Engine, _: &str) -> Result<Vec<Link>> {
            let name = format!("{engine:?}");
            self.calls.lock().unwrap().push(name.clone());
            if name == self.success {
                Ok(vec![Link {
                    title: "Source".into(),
                    url: "https://example.com/".into(),
                    snippet: String::new(),
                }])
            } else {
                anyhow::bail!("moteur bloqué")
            }
        }
    }

    #[tokio::test]
    async fn blocked_engines_fall_back_to_an_independent_engine() {
        let provider = Fake {
            calls: Mutex::new(vec![]),
            success: "DuckDuckGo",
        };
        let reports = search_with(&provider, "Rust", false).await.unwrap();
        assert_eq!(
            *provider.calls.lock().unwrap(),
            vec!["Google", "Bing", "DuckDuckGo"]
        );
        assert!(reports[0].error.is_some());
        assert_eq!(reports[2].results.len(), 1);
    }

    #[tokio::test]
    async fn encyclopedia_fallback_is_used_only_when_general_search_failed() {
        let provider = Fake {
            calls: Mutex::new(vec![]),
            success: "WikipediaFr",
        };
        let reports = search_with(&provider, "Rust", false).await.unwrap();
        assert_eq!(provider.calls.lock().unwrap().len(), 6);
        assert!(matches!(reports[4].engine, Engine::WikipediaFr));
        assert_eq!(reports[4].results.len(), 1);
    }

    #[tokio::test]
    async fn configured_brave_api_avoids_browser_captcha_when_it_succeeds() {
        let provider = Fake {
            calls: Mutex::new(vec![]),
            success: "BraveApi",
        };
        assert_eq!(search_with(&provider, "Rust", true).await.unwrap().len(), 1);
        assert_eq!(*provider.calls.lock().unwrap(), vec!["BraveApi"]);
    }

    #[test]
    fn parses_real_api_shapes_without_treating_snippets_as_read_pages() {
        let links=wikipedia_results(&json!({"query":{"search":[{"pageid":42,"title":"Rust","snippet":"Langage <span>Rust</span> &amp; mémoire"}]}}),"fr").unwrap();
        assert_eq!(links[0].url, "https://fr.wikipedia.org/?curid=42");
        assert_eq!(links[0].snippet, "Langage Rust & mémoire");
        assert!(brave_results(&json!({"error":"rate limit"})).is_err());
        assert_eq!(brave_results(&json!({"web":{"results":[{"title":"Rust","url":"https://example.com/","description":"Source"}]}})).unwrap().len(),1);
    }
}
