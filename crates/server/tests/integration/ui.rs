//! `GET /v1/scopes`, `GET /v1/facts` and the `memoryos ui` page server, against real Postgres.

use memoryos_core::Trust;
use memoryos_server::ui::{router, UiState};
use reqwest::{header, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::common::{assert_error, Key, Server};

const TOKEN: &str = "page-token";

async fn commit(server: &Server, key: &Key, scope: &str, facts: Value) -> Value {
    let body =
        json!({"scope": scope, "source": "test", "content": "said in a test", "facts": facts});
    let (status, value) = server
        .call("POST", "/v1/commits", Some(key), Some(body))
        .await;
    assert_eq!(status, StatusCode::OK, "{value}");
    value
}

fn fact(statement: &str, subject: &str, predicate: &str, object: Option<&str>) -> Value {
    json!({
        "statement": statement,
        "subject": {"name": subject, "kind": "person"},
        "predicate": predicate,
        "object": object.map(|name| json!({"name": name, "kind": "place"})),
    })
}

fn statements(body: &Value) -> Vec<&str> {
    body["facts"]
        .as_array()
        .expect("facts")
        .iter()
        .map(|fact| fact["statement"].as_str().expect("statement"))
        .collect()
}

fn summary<'a>(body: &'a Value, scope: &str) -> Option<&'a Value> {
    body["scopes"]
        .as_array()
        .expect("scopes")
        .iter()
        .find(|item| item["scope"] == scope)
}

#[tokio::test]
async fn scopes_count_what_each_scope_holds_within_the_key() {
    let server = Server::start().await;
    let root = server.root("acme").await;
    let first = commit(
        &server,
        &root,
        "agent:a",
        json!([fact(
            "Ada lives in Jakarta",
            "Ada",
            "lives_in",
            Some("Jakarta")
        )]),
    )
    .await;
    let replaces = first["facts"][0]["id"].as_str().expect("id");
    let mut newer = fact("Ada lives in Bandung", "Ada", "lives_in", Some("Bandung"));
    newer["supersedes"] = json!(replaces);
    commit(&server, &root, "agent:a", json!([newer])).await;
    commit(
        &server,
        &root,
        "user:u",
        json!([fact("Held back", "Ada", "note", None)]),
    )
    .await;
    let (status, _) = server
        .call(
            "PUT",
            "/v1/agents/notes/blocks/plan",
            Some(&root),
            Some(json!({"content": "x"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = server.call("GET", "/v1/scopes", Some(&root), None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["truncated"], false);
    let agent = summary(&body, "agent:a").expect("agent scope");
    assert_eq!(agent["current"], 1);
    assert_eq!(agent["past"], 1);
    assert_eq!(agent["quarantined"], 0);
    assert_eq!(agent["entities"], 3);
    assert_eq!(
        summary(&body, "user:u").expect("user scope")["quarantined"],
        1
    );
    assert_eq!(
        summary(&body, "agent:notes").expect("notes scope")["blocks"],
        1
    );

    // A key sees only its own scopes, and the quarantine count only as an admin.
    let narrow = server.key("acme", &["user:u"], Trust::Agent, false).await;
    let (_, body) = server.call("GET", "/v1/scopes", Some(&narrow), None).await;
    assert!(summary(&body, "agent:a").is_none(), "{body}");
    assert_eq!(
        summary(&body, "user:u").expect("user scope")["quarantined"],
        0
    );

    // Another tenant sees nothing.
    let other = server.root("other").await;
    let (_, body) = server.call("GET", "/v1/scopes", Some(&other), None).await;
    assert_eq!(body["scopes"], json!([]));
    server.stop().await.expect("stop");
}

#[tokio::test]
async fn facts_list_by_state_with_names_and_pages() {
    let server = Server::start().await;
    let root = server.root("acme").await;
    let first = commit(
        &server,
        &root,
        "agent:a",
        json!([fact(
            "Ada lives in Jakarta",
            "Ada",
            "lives_in",
            Some("Jakarta")
        )]),
    )
    .await;
    let mut newer = fact("Ada lives in Bandung", "Ada", "lives_in", Some("Bandung"));
    newer["supersedes"] = first["facts"][0]["id"].clone();
    commit(&server, &root, "agent:a", json!([newer])).await;
    commit(
        &server,
        &root,
        "agent:a",
        json!([fact("Ada likes tea", "Ada", "likes", None)]),
    )
    .await;

    let list = |query: String| {
        let server = &server;
        let root = &root;
        async move {
            server
                .call("GET", &format!("/v1/facts?{query}"), Some(root), None)
                .await
                .1
        }
    };

    let current = list("scope=agent:a".to_owned()).await;
    assert_eq!(
        statements(&current),
        ["Ada likes tea", "Ada lives in Bandung"]
    );
    assert_eq!(current["facts"][1]["subject_name"], "Ada");
    assert_eq!(current["facts"][1]["object_name"], "Bandung");
    assert_eq!(current["facts"][1]["object_kind"], "place");
    assert_eq!(current["facts"][0]["object_name"], Value::Null);

    let past = list("scope=agent:a&state=past".to_owned()).await;
    assert_eq!(statements(&past), ["Ada lives in Jakarta"]);
    assert!(past["facts"][0]["superseded_by"].is_string());

    let page = list("scope=agent:a&state=all&limit=2".to_owned()).await;
    assert_eq!(statements(&page), ["Ada likes tea", "Ada lives in Bandung"]);
    let last = page["facts"][1]["id"].as_str().expect("id");
    let rest = list(format!("scope=agent:a&state=all&limit=2&before={last}")).await;
    assert_eq!(statements(&rest), ["Ada lives in Jakarta"]);
    server.stop().await.expect("stop");
}

#[tokio::test]
async fn facts_list_keeps_to_the_key_and_never_shows_quarantine() {
    let server = Server::start().await;
    let root = server.root("acme").await;
    commit(
        &server,
        &root,
        "user:u",
        json!([fact("Held back", "Ada", "note", None)]),
    )
    .await;

    for state in ["current", "past", "all"] {
        let (status, body) = server
            .call(
                "GET",
                &format!("/v1/facts?scope=user:u&state={state}"),
                Some(&root),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["facts"], json!([]), "{state}");
    }

    let narrow = server.key("acme", &["agent:a"], Trust::Agent, false).await;
    let (status, body) = server
        .call("GET", "/v1/facts?scope=user:u", Some(&narrow), None)
        .await;
    assert_error(status, &body, 403, "forbidden");

    let (status, body) = server
        .call(
            "GET",
            "/v1/facts?scope=agent:a&limit=501",
            Some(&narrow),
            None,
        )
        .await;
    assert_error(status, &body, 400, "invalid_input");
    let (status, body) = server
        .call("GET", "/v1/facts?scope=nonsense", Some(&narrow), None)
        .await;
    assert_error(status, &body, 400, "invalid_input");

    // The same scope name in another tenant holds nothing.
    let other = server.root("other").await;
    let (_, body) = server
        .call(
            "GET",
            "/v1/facts?scope=user:u&state=all",
            Some(&other),
            None,
        )
        .await;
    assert_eq!(body["facts"], json!([]));
    server.stop().await.expect("stop");
}

struct Page {
    url: String,
    http: reqwest::Client,
}

impl Page {
    async fn start(server: &Server, key: &Key) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("address").port();
        let state =
            UiState::new(server.url.clone(), key.token.clone(), TOKEN, port).expect("state");
        tokio::spawn(async move { axum::serve(listener, router(state)).await });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            http: reqwest::Client::new(),
        }
    }

    fn api(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.http
            .request(method, format!("{}/api/{path}", self.url))
            .header("x-memoryos-ui", TOKEN)
    }
}

#[tokio::test]
async fn the_page_serves_itself_locked_down() {
    let server = Server::start().await;
    let root = server.root("acme").await;
    let page = Page::start(&server, &root).await;

    let response = page.http.get(&page.url).send().await.expect("index");
    assert_eq!(response.status(), StatusCode::OK);
    let policy = response.headers()[header::CONTENT_SECURITY_POLICY]
        .to_str()
        .expect("policy");
    assert!(policy.contains("script-src 'self'"), "{policy}");
    assert!(!policy.contains("unsafe-inline"), "{policy}");
    assert!(response.text().await.expect("body").contains("/app.js"));
    for asset in ["app.js", "style.css"] {
        let response = page
            .http
            .get(format!("{}/{asset}", page.url))
            .send()
            .await
            .expect("asset");
        assert_eq!(response.status(), StatusCode::OK, "{asset}");
    }

    // DNS rebinding reaches this listener with a hostile Host header.
    let response = page
        .http
        .get(&page.url)
        .header(header::HOST, "evil.example")
        .send()
        .await
        .expect("rebound");
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let response = page
        .api(reqwest::Method::GET, "scopes")
        .header(header::HOST, "evil.example:80")
        .send()
        .await
        .expect("rebound api");
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    server.stop().await.expect("stop");
}

#[tokio::test]
async fn the_api_needs_the_token_and_the_key_stays_here() {
    let server = Server::start().await;
    let root = server.root("acme").await;
    commit(
        &server,
        &root,
        "agent:a",
        json!([fact("Ada likes tea", "Ada", "likes", None)]),
    )
    .await;
    let page = Page::start(&server, &root).await;

    let scopes = format!("{}/api/scopes", page.url);
    let missing = page.http.get(&scopes).send().await.expect("no token");
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    let wrong = page
        .http
        .get(&scopes)
        .header("x-memoryos-ui", "guess")
        .send()
        .await
        .expect("wrong");
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);
    // The real key is not a way in either.
    let key = page
        .http
        .get(&scopes)
        .bearer_auth(&root.token)
        .send()
        .await
        .expect("key");
    assert_eq!(key.status(), StatusCode::UNAUTHORIZED);

    let response = page
        .api(reqwest::Method::GET, "scopes")
        .send()
        .await
        .expect("scopes");
    assert_eq!(response.status(), StatusCode::OK);
    let text = response.text().await.expect("body");
    assert!(text.contains("agent:a"), "{text}");
    assert!(!text.contains(&root.token), "the key reached the page");

    let response = page
        .api(reqwest::Method::GET, "facts?scope=agent:a")
        .send()
        .await
        .expect("facts");
    let body: Value = response.json().await.expect("json");
    assert_eq!(statements(&body), ["Ada likes tea"]);

    let response = page
        .api(reqwest::Method::POST, "recall")
        .json(&json!({"scopes": ["agent:a"], "query": "tea"}))
        .send()
        .await
        .expect("recall");
    assert_eq!(response.status(), StatusCode::OK);
    server.stop().await.expect("stop");
}

#[tokio::test]
async fn review_works_and_destructive_operations_do_not_pass() {
    let server = Server::start().await;
    let root = server.root("acme").await;
    let held = commit(
        &server,
        &root,
        "user:u",
        json!([fact("Held back", "Ada", "note", None)]),
    )
    .await;
    let id = held["facts"][0]["id"].as_str().expect("id").to_owned();
    let episode = held["episode_id"].as_str().expect("episode").to_owned();
    let page = Page::start(&server, &root).await;

    let response = page
        .api(reqwest::Method::GET, "quarantine?scopes=user:u")
        .send()
        .await
        .expect("list");
    let body: Value = response.json().await.expect("json");
    assert_eq!(statements(&body), ["Held back"]);

    for (method, path) in [
        (reqwest::Method::POST, format!("episodes/{episode}/purge")),
        (
            reqwest::Method::POST,
            format!("episodes%2F{episode}%2Fpurge"),
        ),
        (reqwest::Method::POST, format!("facts/{id}/invalidate")),
        (reqwest::Method::DELETE, "users/u/memory".to_owned()),
        (reqwest::Method::POST, "commits".to_owned()),
        (
            reqwest::Method::GET,
            format!("facts/%2e%2e%2fepisodes%2f{episode}"),
        ),
    ] {
        let response = page.api(method, &path).send().await.expect("refused");
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
    let (status, _) = server
        .call("GET", &format!("/v1/episodes/{episode}"), Some(&root), None)
        .await;
    assert_eq!(status, StatusCode::OK, "the episode must still exist");

    let response = page
        .api(reqwest::Method::POST, &format!("quarantine/{id}/approve"))
        .send()
        .await
        .expect("approve");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        server.recall_statements(&root, "user:u", "Held back").await,
        ["Held back"]
    );
    server.stop().await.expect("stop");
}

#[tokio::test]
async fn a_key_that_cannot_review_gets_the_servers_answer() {
    let server = Server::start().await;
    let agent = server.key("acme", &["user:*"], Trust::Agent, false).await;
    let page = Page::start(&server, &agent).await;
    let response = page
        .api(reqwest::Method::GET, "quarantine?scopes=user:u")
        .send()
        .await
        .expect("list");
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    server.stop().await.expect("stop");
}

#[tokio::test]
async fn a_stopped_server_is_reported_and_not_a_hang() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("address").port();
    // A port that was just free and now has no listener.
    let gone = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let gone_port = gone.local_addr().expect("address").port();
    drop(gone);
    let state = UiState::new(
        format!("http://127.0.0.1:{gone_port}"),
        "k".to_owned(),
        TOKEN,
        port,
    )
    .expect("state");
    tokio::spawn(async move { axum::serve(listener, router(state)).await });
    let response = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/api/scopes"))
        .header("x-memoryos-ui", TOKEN)
        .send()
        .await
        .expect("answer");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body: Value = response.json().await.expect("json");
    assert_eq!(body["error"]["code"], "server_unreachable");
}
