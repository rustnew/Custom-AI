use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    path::PathBuf,
    process::Stdio,
    sync::Mutex,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use clap::{Args, ValueEnum};
use reqwest::{Client, Method, Url};
use serde_json::{Value, json};
use tokio::{
    net::TcpListener,
    process::{Child, Command},
    time::{Instant, sleep},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BrowserKind {
    Chrome,
    Firefox,
}

#[derive(Args)]
pub struct BrowserOptions {
    /// Navigateur à piloter ; les moteurs de secours sont utilisés si nécessaire
    #[arg(long, value_enum, default_value = "chrome")]
    pub browser: BrowserKind,
    /// Pilote WebDriver ; sinon .local/bin/<pilote>, puis PATH
    #[arg(long)]
    pub driver: Option<PathBuf>,
    /// Chemin du navigateur, si sa détection automatique ne convient pas
    #[arg(long)]
    pub browser_binary: Option<PathBuf>,
    /// Affiche la fenêtre du navigateur (une session graphique est nécessaire)
    #[arg(long)]
    pub visible: bool,
    /// Délai maximal par navigation, en secondes
    #[arg(long, default_value_t = 25, value_parser = clap::value_parser!(u64).range(5..=120))]
    pub navigation_timeout: u64,
}

pub struct Browser {
    client: Client,
    base: String,
    session: String,
    child: Child,
    pub visible: bool,
    blocked_engines: Mutex<HashMap<String, String>>,
}

impl Browser {
    pub async fn start(options: &BrowserOptions) -> Result<Self> {
        let name = match options.browser {
            BrowserKind::Chrome => "chromedriver",
            BrowserKind::Firefox => "geckodriver",
        };
        let local = PathBuf::from(".local/bin").join(name);
        let driver = options
            .driver
            .clone()
            .unwrap_or_else(|| if local.is_file() { local } else { name.into() });
        let port_socket = TcpListener::bind("127.0.0.1:0").await?;
        let port = port_socket.local_addr()?.port();
        drop(port_socket);
        let mut command = Command::new(&driver);
        match options.browser {
            BrowserKind::Chrome => {
                command
                    .arg(format!("--port={port}"))
                    .arg("--allowed-ips=127.0.0.1");
            }
            BrowserKind::Firefox => {
                command.args(["--host", "127.0.0.1", "--port", &port.to_string()]);
            }
        }
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "pilote {} inaccessible ; voir les instructions WebDriver du README",
                    driver.display()
                )
            })?;
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(options.navigation_timeout + 10))
            .build()?;
        let mut browser = Self {
            client,
            base: format!("http://127.0.0.1:{port}"),
            session: String::new(),
            child,
            visible: options.visible,
            blocked_engines: Mutex::new(HashMap::new()),
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = browser.child.try_wait()? {
                bail!("{name} s'est arrêté avant le démarrage : {status}");
            }
            if let Ok(value) = browser.command(Method::GET, "/status", None).await
                && value["ready"].as_bool() == Some(true)
            {
                break;
            }
            ensure!(
                Instant::now() < deadline,
                "{name} ne devient pas disponible"
            );
            sleep(Duration::from_millis(100)).await;
        }
        let capabilities = capabilities(options);
        let value = browser
            .command(
                Method::POST,
                "/session",
                Some(json!({"capabilities":{"alwaysMatch":capabilities}})),
            )
            .await
            .with_context(|| {
                format!(
                    "impossible de démarrer {:?} ; vérifier la version du pilote et le navigateur",
                    options.browser
                )
            })?;
        browser.session = value["sessionId"]
            .as_str()
            .context("WebDriver n'a pas renvoyé de sessionId")?
            .to_owned();
        Ok(browser)
    }

    async fn command(&self, method: Method, path: &str, body: Option<Value>) -> Result<Value> {
        let mut request = self.client.request(method, format!("{}{path}", self.base));
        if let Some(body) = body {
            request = request.json(&body);
        }
        let mut response = request.send().await.context("WebDriver ne répond pas")?;
        let status = response.status();
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            ensure!(
                bytes.len() + chunk.len() <= 2 * 1024 * 1024,
                "réponse WebDriver trop volumineuse"
            );
            bytes.extend_from_slice(&chunk);
        }
        let result: Value = serde_json::from_slice(&bytes).context("réponse WebDriver invalide")?;
        let value = &result["value"];
        if !status.is_success() || value.get("error").is_some() {
            bail!(
                "WebDriver {status} : {}",
                value["message"].as_str().unwrap_or("erreur de protocole")
            );
        }
        Ok(value.clone())
    }

    pub async fn navigate(&self, url: &str) -> Result<()> {
        validate_public_url(url).await?;
        self.command(
            Method::POST,
            &format!("/session/{}/url", self.session),
            Some(json!({"url":url})),
        )
        .await?;
        let current = self
            .command(Method::GET, &format!("/session/{}/url", self.session), None)
            .await?;
        validate_public_url(current.as_str().context("URL du navigateur absente")?).await?;
        Ok(())
    }

    /// Lecture du DOM avec les commandes WebDriver natives, sans script JavaScript.
    pub async fn snapshot(&self) -> Result<Value> {
        let url = self
            .command(Method::GET, &format!("/session/{}/url", self.session), None)
            .await?;
        let url = Url::parse(url.as_str().context("URL du navigateur absente")?)?;
        let title = self
            .command(
                Method::GET,
                &format!("/session/{}/title", self.session),
                None,
            )
            .await?;
        let mut roots = self.elements("main, article, [role=main]").await?;
        if roots.is_empty() {
            roots = self.elements("body").await?;
        }
        let text = if let Some(root) = roots.first() {
            self.element_value(root, "text").await?
        } else {
            String::new()
        };
        let host = url.host_str().unwrap_or_default();
        let selector = if host == "www.google.com" || host == "google.com" {
            Some("a:has(h3)")
        } else if host == "www.bing.com" || host == "bing.com" {
            Some("li.b_algo h2 a")
        } else if host == "duckduckgo.com" || host.ends_with(".duckduckgo.com") {
            Some("a.result__a, a[data-testid=result-title-a]")
        } else if host == "search.brave.com" {
            Some("a:has(.snippet-title), a.heading-serpresult, a.result-header")
        } else if host.ends_with(".wikipedia.org") {
            Some(".mw-parser-output a[href]")
        } else {
            Some("main a[href], article a[href], [role=main] a[href]")
        };
        let mut links = Vec::new();
        if let Some(selector) = selector {
            for element in self.elements(selector).await?.into_iter().take(16) {
                let href = self.element_value(&element, "attribute/href").await?;
                let Some(destination) = destination(&url, &href) else {
                    continue;
                };
                let title = self.element_value(&element, "text").await?;
                links.push(
                    json!({"url":destination,"title":super::clipped(&title,250),"snippet":""}),
                );
            }
        }
        Ok(
            json!({"url":url.as_str(),"title":title.as_str().unwrap_or_default(),"text":super::clipped(&text,160000),"links":links,"total_bytes":text.len(),"content_truncated":text.len() > 160000,"reader":"webdriver"}),
        )
    }

    pub fn blocked_engine(&self, engine: super::Engine) -> Option<String> {
        self.blocked_engines
            .lock()
            .ok()?
            .get(&format!("{engine:?}"))
            .cloned()
    }

    pub fn block_engine(&self, engine: super::Engine, message: &str) {
        if let Ok(mut blocked) = self.blocked_engines.lock() {
            blocked.insert(format!("{engine:?}"), super::clipped(message, 300));
        }
    }

    async fn elements(&self, selector: &str) -> Result<Vec<String>> {
        let value = self
            .command(
                Method::POST,
                &format!("/session/{}/elements", self.session),
                Some(json!({"using":"css selector","value":selector})),
            )
            .await?;
        value
            .as_array()
            .context("liste d'éléments WebDriver absente")?
            .iter()
            .map(|element| {
                element["element-6066-11e4-a52e-4f735466cecf"]
                    .as_str()
                    .map(str::to_owned)
                    .context("identifiant d'élément WebDriver absent")
            })
            .collect()
    }

    async fn element_value(&self, id: &str, property: &str) -> Result<String> {
        ensure!(
            id.bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_')),
            "identifiant d'élément invalide"
        );
        let value = self
            .command(
                Method::GET,
                &format!("/session/{}/element/{id}/{property}", self.session),
                None,
            )
            .await?;
        Ok(value.as_str().unwrap_or_default().to_owned())
    }

    pub async fn close(mut self) -> Result<()> {
        let result = if self.session.is_empty() {
            Ok(Value::Null)
        } else {
            tokio::time::timeout(
                Duration::from_secs(5),
                self.command(Method::DELETE, &format!("/session/{}", self.session), None),
            )
            .await
            .context("fermeture du navigateur trop lente")
            .and_then(|result| result)
        };
        self.child.kill().await.ok();
        self.child.wait().await.ok();
        result.map(|_| ())
    }
}

fn destination(base: &Url, href: &str) -> Option<String> {
    let url = base.join(href).ok()?;
    let host = url.host_str()?;
    if (host == "duckduckgo.com" || host.ends_with(".duckduckgo.com"))
        && let Some((_, destination)) = url.query_pairs().find(|(key, _)| key == "uddg")
    {
        return Some(destination.into_owned());
    }
    if (host == "www.google.com" || host == "google.com") && url.path() == "/url" {
        return url
            .query_pairs()
            .find(|(key, _)| key == "q" || key == "url")
            .map(|(_, value)| value.into_owned());
    }
    if (host == "www.bing.com" || host == "bing.com") && url.path().starts_with("/ck/") {
        let encoded = url.query_pairs().find(|(key, _)| key == "u")?.1;
        let decoded = URL_SAFE_NO_PAD
            .decode(encoded.strip_prefix("a1")?.trim_end_matches('='))
            .ok()?;
        return String::from_utf8(decoded).ok();
    }
    Some(url.to_string())
}

fn capabilities(options: &BrowserOptions) -> Value {
    let mut caps = json!({"pageLoadStrategy":"eager", "timeouts":{"pageLoad":options.navigation_timeout * 1000,"script":5000,"implicit":0}});
    match options.browser {
        BrowserKind::Chrome => {
            caps["browserName"] = json!("chrome");
            let mut args = vec!["--disable-dev-shm-usage", "--no-first-run"];
            if !options.visible {
                args.push("--headless=new");
            }
            caps["goog:chromeOptions"] = json!({"args":args,"prefs":{"download_restrictions":3,"profile.default_content_setting_values.notifications":2}});
            if let Some(binary) = &options.browser_binary {
                caps["goog:chromeOptions"]["binary"] = json!(binary);
            }
        }
        BrowserKind::Firefox => {
            caps["browserName"] = json!("firefox");
            let args: Vec<&str> = if options.visible {
                vec![]
            } else {
                vec!["-headless"]
            };
            caps["moz:firefoxOptions"] = json!({"args":args,"prefs":{"browser.download.always_ask_before_handling_new_types":true,"dom.webnotifications.enabled":false}});
            if let Some(binary) = &options.browser_binary {
                caps["moz:firefoxOptions"]["binary"] = json!(binary);
            }
        }
    }
    caps
}

pub fn parse_public_url(value: &str) -> Result<Url> {
    ensure!(value.len() <= 4096, "URL trop longue");
    let url = Url::parse(value).context("URL invalide")?;
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "seuls HTTP et HTTPS sont autorisés"
    );
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "identifiants interdits dans l'URL"
    );
    let host = url.host_str().context("hôte absent")?;
    ensure!(
        !host.eq_ignore_ascii_case("localhost")
            && !host.ends_with(".localhost")
            && !host.ends_with(".local"),
        "adresse locale interdite"
    );
    if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        ensure!(public_ip(ip), "adresse privée interdite");
    }
    Ok(url)
}

fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_v4(ip),
        IpAddr::V6(ip) => public_v6(ip),
    }
}

fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, _, _] = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_multicast()
        || ip.is_broadcast()
        || ip.is_documentation()
        || a == 0
        || a >= 240
        || (a == 100 && (64..=127).contains(&b))
        || (a == 198 && (18..=19).contains(&b)))
}

fn public_v6(ip: Ipv6Addr) -> bool {
    if let Some(ip) = ip.to_ipv4_mapped() {
        return public_v4(ip);
    }
    let segments = ip.segments();
    // Les adresses publiques unicast utilisées ici appartiennent à 2000::/3.
    (segments[0] & 0xe000) == 0x2000 && !(segments[0] == 0x2001 && segments[1] == 0xdb8)
}

async fn validate_public_url(value: &str) -> Result<()> {
    let url = parse_public_url(value)?;
    let host = url
        .host_str()
        .context("hôte absent")?
        .trim_matches(['[', ']']);
    let addresses = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::net::lookup_host((host, url.port_or_known_default().unwrap_or(443))),
    )
    .await
    .context("résolution DNS trop lente")??;
    let mut found = false;
    for address in addresses {
        ensure!(
            public_ip(address.ip()),
            "l'hôte résout vers une adresse privée"
        );
        found = true;
    }
    ensure!(found, "aucune adresse publique pour cet hôte");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        extract::State,
        routing::{delete, get, post},
    };
    use std::sync::{Arc, Mutex};

    fn options(browser: BrowserKind) -> BrowserOptions {
        BrowserOptions {
            browser,
            driver: None,
            browser_binary: None,
            visible: false,
            navigation_timeout: 25,
        }
    }

    #[test]
    fn uses_native_capabilities_and_fresh_profiles_for_each_browser() {
        let chrome = capabilities(&options(BrowserKind::Chrome));
        assert_eq!(chrome["browserName"], "chrome");
        assert!(
            chrome["goog:chromeOptions"]["args"]
                .as_array()
                .unwrap()
                .contains(&json!("--headless=new"))
        );
        assert!(!chrome.to_string().contains("user-data-dir"));
        let firefox = capabilities(&options(BrowserKind::Firefox));
        assert_eq!(firefox["browserName"], "firefox");
        assert_eq!(firefox["moz:firefoxOptions"]["args"], json!(["-headless"]));
        assert_eq!(firefox["timeouts"]["pageLoad"], 25000);
    }

    #[tokio::test]
    async fn webdriver_reads_dom_detects_errors_and_closes_its_own_session() {
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let calls_for_close = calls.clone();
        let app = Router::new()
            .route(
                "/status",
                get(|| async { Json(json!({"value":{"ready":true}})) }),
            )
            .route(
                "/error",
                get(|| async {
                    (
                        axum::http::StatusCode::BAD_REQUEST,
                        Json(json!({"value":{"error":"invalid argument","message":"test error"}})),
                    )
                }),
            )
            .route(
                "/session/test/url",
                get(|| async { Json(json!({"value":"https://example.com/"})) }),
            )
            .route(
                "/session/test/title",
                get(|| async { Json(json!({"value":"Fixture"})) }),
            )
            .route(
                "/session/test/elements",
                post(|Json(body): Json<Value>| async move {
                    assert_eq!(body["using"], "css selector");
                    if body["value"] == "main a[href], article a[href], [role=main] a[href]" {
                        return Json(json!({"value":[]}));
                    }
                    assert_eq!(body["value"], "main, article, [role=main]");
                    Json(json!({"value":[{"element-6066-11e4-a52e-4f735466cecf":"root"}]}))
                }),
            )
            .route(
                "/session/test/element/root/text",
                get(|| async { Json(json!({"value":"Texte du DOM"})) }),
            )
            .route(
                "/session/test",
                delete(|State(calls): State<Arc<Mutex<Vec<String>>>>| async move {
                    calls.lock().unwrap().push("delete test".into());
                    Json(json!({"value":null}))
                }),
            )
            .with_state(calls_for_close);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let child = Command::new("sleep")
            .arg("30")
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let browser = Browser {
            client: Client::new(),
            base: format!("http://{address}"),
            session: "test".into(),
            child,
            visible: false,
            blocked_engines: Mutex::new(HashMap::new()),
        };
        assert_eq!(
            browser.command(Method::GET, "/status", None).await.unwrap()["ready"],
            true
        );
        assert!(
            browser
                .command(Method::GET, "/error", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("test error")
        );
        assert_eq!(browser.snapshot().await.unwrap()["text"], "Texte du DOM");
        browser.close().await.unwrap();
        assert_eq!(*calls.lock().unwrap(), vec!["delete test"]);
        task.abort();
    }

    #[test]
    fn resolves_relative_links_and_search_engine_redirects_in_rust() {
        let google = Url::parse("https://www.google.com/search?q=rust").unwrap();
        assert_eq!(
            destination(&google, "/url?q=https%3A%2F%2Fexample.com%2F").unwrap(),
            "https://example.com/"
        );
        let bing = Url::parse("https://www.bing.com/search?q=rust").unwrap();
        let encoded = URL_SAFE_NO_PAD.encode("https://example.com/école");
        assert_eq!(
            destination(&bing, &format!("/ck/a?u=a1{encoded}")).unwrap(),
            "https://example.com/école"
        );
        assert!(destination(&bing, "/ck/a?u=a1!!!!").is_none());
        assert_eq!(
            destination(&google, "https://example.com/").unwrap(),
            "https://example.com/"
        );
        let duck = Url::parse("https://html.duckduckgo.com/html/").unwrap();
        assert_eq!(
            destination(
                &duck,
                "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F"
            )
            .unwrap(),
            "https://example.com/"
        );
    }
}
