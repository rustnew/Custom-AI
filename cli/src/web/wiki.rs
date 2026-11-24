use anyhow::{Context, Result, ensure};
use percent_encoding::percent_decode_str;
use reqwest::{Client, Url};
use serde_json::Value;
use std::time::Duration;

use super::{Link, Page, clipped, webdriver::parse_public_url};

pub fn request_url(value: &str) -> Result<Option<Url>> {
    let source = parse_public_url(value)?;
    let Some(prefix) = source
        .host_str()
        .and_then(|host| host.strip_suffix(".wikipedia.org"))
    else {
        return Ok(None);
    };
    let language = prefix.strip_suffix(".m").unwrap_or(prefix);
    if language.is_empty()
        || language.len() > 20
        || !language
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte == b'-')
    {
        return Ok(None);
    }
    let target = if let Some((_, id)) = source.query_pairs().find(|(key, _)| key == "curid") {
        ensure!(
            id.parse::<u64>().is_ok(),
            "identifiant d'article Wikipédia invalide"
        );
        ("pageids", id.into_owned())
    } else if let Some((_, title)) = source.query_pairs().find(|(key, _)| key == "title") {
        ("titles", title.into_owned())
    } else if let Some(title) = source.path().strip_prefix("/wiki/") {
        (
            "titles",
            percent_decode_str(title).decode_utf8()?.into_owned(),
        )
    } else {
        return Ok(None);
    };
    let mut url = Url::parse_with_params(
        &format!("https://{language}.wikipedia.org/w/api.php"),
        &[
            ("action", "query"),
            ("prop", "extracts|info|links"),
            ("explaintext", "1"),
            ("exsectionformat", "plain"),
            ("inprop", "url"),
            ("pllimit", "16"),
            ("format", "json"),
            ("formatversion", "2"),
            ("redirects", "1"),
        ],
    )?;
    url.query_pairs_mut().append_pair(target.0, &target.1);
    Ok(Some(url))
}

pub async fn read(request: Url) -> Result<Page> {
    let client = Client::builder()
        .user_agent("Custom-AI/0.1 (https://github.com/rustnew/Custom-AI)")
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let value = super::providers::json_response(client.get(request.clone())).await?;
    parse_page(&value, &request)
}

fn parse_page(value: &Value, request: &Url) -> Result<Page> {
    ensure!(
        value.get("error").is_none(),
        "erreur API de lecture Wikipédia"
    );
    let page = &value["query"]["pages"][0];
    ensure!(page.get("missing").is_none(), "article Wikipédia absent");
    let text = page["extract"]
        .as_str()
        .filter(|text| !text.trim().is_empty())
        .context("article sans extrait texte disponible")?;
    let title = page["title"].as_str().context("titre d'article absent")?;
    let url = page["canonicalurl"]
        .as_str()
        .context("URL d'article absente")?;
    let canonical = parse_public_url(url)?;
    ensure!(
        canonical.host_str() == request.host_str(),
        "l'article renvoie vers un autre site"
    );
    let origin = format!("https://{}", request.host_str().context("hôte API absent")?);
    let mut links = Vec::new();
    if let Some(items) = page["links"].as_array() {
        for item in items
            .iter()
            .filter(|item| item["ns"].as_i64() == Some(0))
            .take(16)
        {
            if let Some(title) = item["title"].as_str() {
                let url =
                    Url::parse_with_params(&format!("{origin}/w/index.php"), &[("title", title)])?;
                links.push(Link {
                    title: title.into(),
                    url: url.into(),
                    snippet: String::new(),
                });
            }
        }
    }
    Ok(Page {
        url: canonical.into(),
        title: title.into(),
        text: clipped(text, 160000),
        links,
        total_bytes: text.len(),
        content_truncated: text.len() > 160000,
        reader: "mediawiki-api".into(),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wikipedia_urls_map_to_official_read_only_api_requests() {
        for source in [
            "https://fr.wikipedia.org/?curid=6719375",
            "https://fr.wikipedia.org/wiki/Rust_%28langage%29",
            "https://fr.m.wikipedia.org/wiki/Rust_(langage)",
        ] {
            let request = request_url(source).unwrap().unwrap();
            assert_eq!(request.host_str(), Some("fr.wikipedia.org"));
            assert!(
                request
                    .query_pairs()
                    .any(|(key, value)| key == "action" && value == "query")
            );
            assert!(
                request
                    .query_pairs()
                    .any(|(key, value)| (key == "pageids" && value == "6719375")
                        || (key == "titles" && value == "Rust_(langage)"))
            );
        }
        assert!(
            request_url("https://example.com/wiki/Rust")
                .unwrap()
                .is_none()
        );
        assert!(
            request_url("https://fr.wikipedia.org.evil.example/wiki/Rust")
                .unwrap()
                .is_none()
        );
        assert!(request_url("https://fr.wikipedia.org/?curid=delete").is_err());
    }

    #[test]
    fn real_article_text_and_links_are_distinct_from_search_snippets() {
        let request = request_url("https://fr.wikipedia.org/?curid=1")
            .unwrap()
            .unwrap();
        let page=parse_page(&json!({"query":{"pages":[{"pageid":1,"title":"Rust","canonicalurl":"https://fr.wikipedia.org/wiki/Rust","extract":"Contenu réel de l'article.","links":[{"ns":0,"title":"Programmation"},{"ns":6,"title":"File:Image.png"}]}]}}),&request).unwrap();
        assert_eq!(page.reader, "mediawiki-api");
        assert_eq!(page.text, "Contenu réel de l'article.");
        assert_eq!(page.links.len(), 1);
        assert!(parse_page(&json!({"query":{"pages":[{"missing":true}]}}), &request).is_err());
    }
}
