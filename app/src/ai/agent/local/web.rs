use std::{
    io::{Read, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs},
    time::Duration,
};

use anyhow::{anyhow, bail};
use regex::Regex;
use reqwest::{blocking::Client, header::CONTENT_TYPE, redirect::Policy, Url};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::LazyLock;

const SEARXNG_ENDPOINT_ENV: &str = "WARP_LOCAL_AGENT_SEARXNG_URL";
const MAX_SEARCH_QUERY_CHARS: usize = 512;
const MAX_SEARCH_RESULTS: usize = 5;
const MAX_SEARCH_RESULT_CHARS: usize = 2_000;
const MAX_FETCH_BYTES: usize = 512 * 1024;
const MAX_FETCH_CHARS: usize = 64 * 1024;
const DEFAULT_FETCH_CHARS: usize = 32 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_REDIRECTS: usize = 5;

static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<[^>]+>").unwrap());
static SCRIPT_STYLE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<noscript[^>]*>.*?</noscript>",
    )
    .unwrap()
});
static WHITESPACE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum LocalWebUiStatus {
    SearchSuccess {
        query: String,
        pages: Vec<(String, String)>,
    },
    FetchSuccess {
        pages: Vec<(String, String, bool)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LocalWebToolOutput {
    pub text: String,
    pub ui_status: LocalWebUiStatus,
}

#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct WebFetchArgs {
    url: String,
    max_bytes: Option<usize>,
    max_chars: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SearxngResponse {
    #[serde(default)]
    results: Vec<SearxngResult>,
}

#[derive(Debug, Deserialize)]
struct SearxngResult {
    title: Option<String>,
    url: Option<String>,
    content: Option<String>,
    engine: Option<String>,
    published_date: Option<String>,
}

pub(super) fn web_search_tool_definition() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "Non-empty search query to send to the configured local SearXNG endpoint." },
            "max_results": { "type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS, "description": "Optional maximum number of results. Clamped to a small local limit." }
        },
        "required": ["query"]
    })
}

pub(super) fn web_fetch_tool_definition() -> Value {
    json!({
        "type": "object",
        "properties": {
            "url": { "type": "string", "description": "Absolute HTTP(S) URL to fetch with a single bounded GET request." },
            "max_bytes": { "type": "integer", "minimum": 1, "maximum": MAX_FETCH_BYTES, "description": "Optional response byte cap. Clamped to the implementation maximum." },
            "max_chars": { "type": "integer", "minimum": 1, "maximum": MAX_FETCH_CHARS, "description": "Optional returned text character cap. Clamped to the implementation maximum." }
        },
        "required": ["url"]
    })
}

pub(super) fn execute_web_search_tool(arguments: &str) -> anyhow::Result<LocalWebToolOutput> {
    let args: WebSearchArgs =
        serde_json::from_str(arguments).map_err(|_| anyhow!("Invalid web_search arguments"))?;
    let query = bounded_non_empty(&args.query, MAX_SEARCH_QUERY_CHARS, "query")?;
    let endpoint = configured_searxng_endpoint()
        .ok_or_else(|| anyhow!("web_search unavailable: set WARP_LOCAL_AGENT_SEARXNG_URL to your local SearXNG endpoint URL. No third-party search API is called by default."))?;
    let max_results = args
        .max_results
        .unwrap_or(MAX_SEARCH_RESULTS)
        .clamp(1, MAX_SEARCH_RESULTS);

    let mut url = endpoint;
    {
        let mut pairs = url.query_pairs_mut();
        pairs.append_pair("q", &query);
        pairs.append_pair("format", "json");
    }
    let response = local_web_client()?.get(url.clone()).send()?;
    if !response.status().is_success() {
        bail!(
            "web_search unavailable: SearXNG returned HTTP {}",
            response.status()
        );
    }
    let searxng = response.json::<SearxngResponse>()?;
    let (text, pages) = searxng_provider_text(query.clone(), searxng, max_results);

    Ok(LocalWebToolOutput {
        text,
        ui_status: LocalWebUiStatus::SearchSuccess { query, pages },
    })
}

fn searxng_provider_text(
    query: String,
    searxng: SearxngResponse,
    max_results: usize,
) -> (String, Vec<(String, String)>) {
    let results = searxng
        .results
        .into_iter()
        .filter_map(|result| {
            let url = result.url?;
            let title = result.title.unwrap_or_else(|| url.clone());
            Some((
                url,
                title,
                result.content,
                result.engine,
                result.published_date,
            ))
        })
        .take(max_results)
        .collect::<Vec<_>>();

    let mut text = format!(
        "Status: success\nQuery: {query}\nResults: {}\n",
        results.len()
    );
    let mut pages = Vec::new();
    for (index, (url, title, snippet, engine, published_date)) in results.into_iter().enumerate() {
        pages.push((url.clone(), title.clone()));
        text.push_str(&format!(
            "\n{}. {}\nURL: {}\n",
            index + 1,
            truncate_chars(&title, 200),
            url
        ));
        if let Some(engine) = engine.filter(|value| !value.is_empty()) {
            text.push_str(&format!("Engine: {}\n", truncate_chars(&engine, 80)));
        }
        if let Some(date) = published_date.filter(|value| !value.is_empty()) {
            text.push_str(&format!("Published: {}\n", truncate_chars(&date, 80)));
        }
        if let Some(snippet) = snippet.filter(|value| !value.is_empty()) {
            text.push_str(&format!("Snippet: {}\n", truncate_chars(&snippet, 500)));
        }
    }

    (truncate_chars(&text, MAX_SEARCH_RESULT_CHARS), pages)
}

pub(super) fn execute_web_fetch_tool(arguments: &str) -> anyhow::Result<LocalWebToolOutput> {
    let args: WebFetchArgs =
        serde_json::from_str(arguments).map_err(|_| anyhow!("Invalid web_fetch arguments"))?;
    let mut url = parse_http_url(&args.url)?;
    let max_bytes = args
        .max_bytes
        .unwrap_or(MAX_FETCH_BYTES)
        .clamp(1, MAX_FETCH_BYTES);
    let max_chars = args
        .max_chars
        .unwrap_or(DEFAULT_FETCH_CHARS)
        .clamp(1, MAX_FETCH_CHARS);

    for redirect_count in 0..=MAX_REDIRECTS {
        validate_public_url(&url)?;
        let client = checked_client_for_url(&url)?;
        let response = client.get(url.clone()).send()?;
        if response.status().is_redirection() {
            if redirect_count == MAX_REDIRECTS {
                bail!("web_fetch denied: redirect limit exceeded");
            }
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| anyhow!("web_fetch failed: redirect missing Location header"))?;
            url = url.join(location)?;
            continue;
        }
        let final_url = response.url().clone();
        validate_public_url(&final_url)?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        if !status.is_success() {
            bail!("web_fetch failed: HTTP {status}");
        }
        if !is_allowed_content_type(&content_type) {
            return Ok(unsupported_content_output(&final_url, &content_type));
        }

        let (bytes, byte_truncated) = read_bounded(response, max_bytes)?;
        let mut text = String::from_utf8_lossy(&bytes).to_string();
        if is_html_content_type(&content_type) {
            text = html_to_text(&text);
        }
        let (text, char_truncated) = truncate_chars_with_flag(&text, max_chars);
        let title = page_title(&text).unwrap_or_else(|| final_url.to_string());
        let insecure = if final_url.scheme() == "http" {
            "Scheme: http (insecure transport; SSRF checks passed)\n"
        } else {
            ""
        };
        let truncation = if byte_truncated || char_truncated {
            "\n[web_fetch result truncated]"
        } else {
            ""
        };
        let provider_text = format!(
            "Status: success\nURL: {final_url}\n{insecure}Content-Type: {content_type}\n\nUntrusted fetched page text:\n{text}{truncation}"
        );
        return Ok(LocalWebToolOutput {
            text: provider_text,
            ui_status: LocalWebUiStatus::FetchSuccess {
                pages: vec![(final_url.to_string(), truncate_chars(&title, 120), true)],
            },
        });
    }

    unreachable!("redirect loop exits via return or error")
}

fn configured_searxng_endpoint() -> Option<Url> {
    std::env::var(SEARXNG_ENDPOINT_ENV)
        .ok()
        .and_then(|value| parse_http_url(&value).ok())
}

fn parse_http_url(value: &str) -> anyhow::Result<Url> {
    let url = Url::parse(value.trim()).map_err(|_| anyhow!("URL must be absolute HTTP(S) URL"))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        _ => bail!("URL must use http or https"),
    }
}

fn validate_public_url(url: &Url) -> anyhow::Result<Vec<SocketAddr>> {
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("URL must include a host"))?;
    let host_lower = host.to_ascii_lowercase();
    if host_lower == "localhost"
        || host_lower.ends_with(".localhost")
        || host_lower.ends_with(".local")
        || host_lower.ends_with(".lan")
        || host_lower.ends_with(".home")
        || host_lower.ends_with(".internal")
    {
        bail!("URL host is denied by local-network SSRF policy");
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        deny_private_ip(ip)?;
        let port = url.port_or_known_default().unwrap_or(443);
        return Ok(vec![SocketAddr::new(ip, port)]);
    }

    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow!("URL must include a valid port"))?;
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|_| anyhow!("URL host could not be resolved"))?
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        bail!("URL host resolved to no addresses");
    }
    for addr in &addrs {
        deny_private_ip(addr.ip())?;
    }
    Ok(addrs)
}

fn checked_client_for_url(url: &Url) -> anyhow::Result<Client> {
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("URL must include a host"))?;
    let addrs = validate_public_url(url)?;
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .redirect(Policy::none())
        .resolve_to_addrs(host, &addrs)
        .build()
        .map_err(Into::into)
}

fn local_web_client() -> anyhow::Result<Client> {
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .redirect(Policy::none())
        .build()
        .map_err(Into::into)
}

fn deny_private_ip(ip: IpAddr) -> anyhow::Result<()> {
    match ip {
        IpAddr::V4(ip) => deny_private_ipv4(ip),
        IpAddr::V6(ip) => deny_private_ipv6(ip),
    }
}

fn deny_private_ipv4(ip: Ipv4Addr) -> anyhow::Result<()> {
    if ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.octets()[0] == 0
        || ip == Ipv4Addr::new(169, 254, 169, 254)
    {
        bail!("URL resolves to a denied local/private address");
    }
    Ok(())
}

fn deny_private_ipv6(ip: Ipv6Addr) -> anyhow::Result<()> {
    if let Some(mapped) = ip.to_ipv4_mapped() {
        return deny_private_ipv4(mapped);
    }
    let segments = ip.segments();
    let first = segments[0];
    if ip.is_unspecified()
        || ip.is_loopback()
        || (first & 0xffc0) == 0xfe80
        || (first & 0xfe00) == 0xfc00
    {
        bail!("URL resolves to a denied local/private address");
    }
    Ok(())
}

fn is_allowed_content_type(content_type: &str) -> bool {
    let lower = content_type.to_ascii_lowercase();
    lower.starts_with("text/html")
        || lower.starts_with("text/plain")
        || lower.starts_with("application/json")
        || lower.starts_with("application/xml")
        || lower.starts_with("text/xml")
        || lower.is_empty()
}

fn unsupported_content_output(url: &Url, content_type: &str) -> LocalWebToolOutput {
    let content_type = if content_type.is_empty() {
        "unknown"
    } else {
        content_type
    };
    LocalWebToolOutput {
        text: format!(
            "Status: unsupported-content\nURL: {url}\nContent-Type: {content_type}\nNo raw binary/media content was returned to the provider."
        ),
        ui_status: LocalWebUiStatus::FetchSuccess {
            pages: vec![(url.to_string(), "Unsupported content".to_string(), false)],
        },
    }
}

fn is_html_content_type(content_type: &str) -> bool {
    content_type.to_ascii_lowercase().starts_with("text/html")
}

fn read_bounded(
    mut response: reqwest::blocking::Response,
    max_bytes: usize,
) -> anyhow::Result<(Vec<u8>, bool)> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = response.read(&mut buffer)?;
        if read == 0 {
            return Ok((bytes, false));
        }
        let remaining = max_bytes.saturating_sub(bytes.len());
        if read > remaining {
            bytes.write_all(&buffer[..remaining])?;
            return Ok((bytes, true));
        }
        bytes.write_all(&buffer[..read])?;
    }
}

fn html_to_text(html: &str) -> String {
    let without_script = SCRIPT_STYLE_RE.replace_all(html, " ");
    let without_tags = TAG_RE.replace_all(&without_script, " ");
    normalize_whitespace(&html_unescape(&without_tags))
}

fn html_unescape(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn normalize_whitespace(text: &str) -> String {
    WHITESPACE_RE.replace_all(text, " ").trim().to_string()
}

fn page_title(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| truncate_chars(line, 120))
}

fn bounded_non_empty(value: &str, max_chars: usize, name: &str) -> anyhow::Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{name} must be non-empty");
    }
    Ok(truncate_chars(value, max_chars))
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    truncate_chars_with_flag(text, max_chars).0
}

fn truncate_chars_with_flag(text: &str, max_chars: usize) -> (String, bool) {
    if text.chars().count() <= max_chars {
        return (text.to_string(), false);
    }
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str("\n[truncated]");
    (truncated, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::{
        io::{BufRead, BufReader},
        net::TcpListener,
        thread,
    };

    fn local_http_server(body: &'static str, content_type: &'static str) -> Url {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 0 {
                    if line == "\r\n" {
                        break;
                    }
                    line.clear();
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        Url::parse(&format!("http://{addr}/")).unwrap()
    }

    #[test]
    fn denies_loopback_private_link_local_local_and_metadata_hosts() {
        for url in [
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
            "http://169.254.1.1/",
            "http://169.254.169.254/",
            "http://[::1]/",
            "http://[fc00::1]/",
            "http://[fe80::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://0.0.0.0/",
            "http://printer.local/",
        ] {
            let url = parse_http_url(url).unwrap();
            assert!(validate_public_url(&url).is_err(), "{url}");
        }
    }

    #[test]
    fn unsupported_content_returns_metadata_only() {
        let url = Url::parse("https://example.com/file.png").unwrap();
        let output = unsupported_content_output(&url, "image/png");

        assert!(output.text.contains("unsupported-content"));
        assert!(!output.text.contains("PNG_BYTES"));
    }

    #[test]
    fn redirect_target_is_rechecked_by_public_url_policy() {
        let base = Url::parse("https://example.com/search").unwrap();
        let redirect = base.join("http://127.0.0.1/private").unwrap();

        assert!(validate_public_url(&redirect).is_err());
    }

    #[test]
    fn truncates_by_character_limit() {
        let (text, truncated) = truncate_chars_with_flag("abcdef", 3);

        assert!(truncated);
        assert_eq!(text, "abc\n[truncated]");
    }

    #[test]
    fn read_bounded_marks_byte_truncation() {
        let url = local_http_server("abcdef", "text/plain");
        let response = local_web_client().unwrap().get(url).send().unwrap();

        let (bytes, truncated) = read_bounded(response, 3).unwrap();

        assert!(truncated);
        assert_eq!(bytes, b"abc");
    }

    #[test]
    fn search_response_is_bounded_provider_text() {
        let response = SearxngResponse {
            results: vec![SearxngResult {
                title: Some("Example".to_string()),
                url: Some("https://example.com".to_string()),
                content: Some("x".repeat(1000)),
                engine: Some("test".to_string()),
                published_date: Some("2026-05-25".to_string()),
            }],
        };
        let (text, pages) = searxng_provider_text("rust".to_string(), response, 5);

        assert!(text.contains("https://example.com"));
        assert_eq!(
            pages,
            vec![("https://example.com".to_string(), "Example".to_string())]
        );
        assert!(text.len() < 900);
    }

    #[test]
    #[serial]
    fn missing_searxng_endpoint_returns_unavailable_error() {
        let previous = std::env::var(SEARXNG_ENDPOINT_ENV).ok();
        std::env::remove_var(SEARXNG_ENDPOINT_ENV);

        let error = execute_web_search_tool(r#"{"query":"rust"}"#).unwrap_err();

        if let Some(previous) = previous {
            std::env::set_var(SEARXNG_ENDPOINT_ENV, previous);
        }
        assert!(error.to_string().contains("web_search unavailable"));
        assert!(error.to_string().contains(SEARXNG_ENDPOINT_ENV));
    }
}
