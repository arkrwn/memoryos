#![cfg(feature = "integration-sqlite")]

use std::process::Stdio;
use std::time::{Duration, Instant};

use memoryos_core::{MemoryStore, Trust};
use memoryos_store::SqliteStore;
use reqwest::StatusCode;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::process::{Child, Command};

const BINARY: &str = env!("CARGO_BIN_EXE_memoryos");

fn database_url(directory: &TempDir) -> String {
    format!("sqlite://{}/memoryos.db", directory.path().display())
}

fn env(database_url: &str) -> Vec<(String, String)> {
    vec![
        ("MEMORYOS_STORE".to_owned(), "sqlite".to_owned()),
        ("DATABASE_URL".to_owned(), database_url.to_owned()),
        ("MEMORYOS_TEST_EMBEDDER".to_owned(), "0".to_owned()),
        ("MEMORYOS_RERANKER".to_owned(), "off".to_owned()),
    ]
}

fn command(env: &[(String, String)], args: &[&str]) -> Command {
    let mut command = Command::new(BINARY);
    command
        .env_clear()
        .envs(
            env.iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command
}

struct Output {
    success: bool,
    stdout: String,
    stderr: String,
}

async fn run(env: &[(String, String)], args: &[&str]) -> Output {
    let output = tokio::time::timeout(Duration::from_secs(60), command(env, args).output())
        .await
        .expect("command timeout")
        .expect("run memoryos");
    Output {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

async fn create_key(env: &[(String, String)], tenant: &str) -> String {
    let output = run(
        env,
        &[
            "key",
            "create",
            "--tenant",
            tenant,
            "--name",
            "sqlite-test",
            "--scopes",
            "*",
            "--max-trust",
            "system",
            "--admin",
        ],
    )
    .await;
    assert!(output.success, "{}", output.stderr);
    output.stdout.trim().to_owned()
}

struct Running {
    child: Child,
    url: String,
    http: reqwest::Client,
}

impl Running {
    async fn start(env: &[(String, String)]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("address").port();
        drop(listener);
        let mut env = env.to_vec();
        env.push(("MEMORYOS_BIND".to_owned(), format!("127.0.0.1:{port}")));
        let child = command(&env, &["serve"]).spawn().expect("start memoryos");
        let running = Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
            http: reqwest::Client::new(),
        };
        running.wait_for("/readyz").await;
        running
    }

    async fn wait_for(&self, path: &str) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            let response = self.http.get(format!("{}{path}", self.url)).send().await;
            if response.is_ok_and(|response| response.status().is_success()) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("memoryos did not answer {path}");
    }

    async fn call(
        &self,
        method: &str,
        path: &str,
        token: &str,
        body: Value,
    ) -> (StatusCode, Value) {
        let response = self
            .http
            .request(
                method.parse().expect("method"),
                format!("{}{path}", self.url),
            )
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = response.status();
        let body = response.json().await.expect("JSON response");
        (status, body)
    }

    async fn stop(self) -> Output {
        let pid = self.child.id().expect("process id").to_string();
        let status = std::process::Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .expect("signal memoryos");
        assert!(status.success());
        let output = tokio::time::timeout(Duration::from_secs(40), self.child.wait_with_output())
            .await
            .expect("shutdown timeout")
            .expect("wait for memoryos");
        Output {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    }
}

fn statement(text: &str) -> Value {
    json!({
        "statement": text,
        "subject": {"name": "Ada", "kind": "person"},
        "predicate": "fact_about"
    })
}

async fn recall(server: &Running, token: &str) -> Value {
    let (status, body) = server
        .call(
            "POST",
            "/v1/recall",
            token,
            json!({"scopes": ["agent:a1"], "query": "Ada lives"}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_migrate_and_key_administration_work_without_postgres() {
    let directory = TempDir::new().expect("temporary directory");
    let url = database_url(&directory);
    let env = env(&url);

    for _ in 0..2 {
        let output = run(&env, &["migrate"]).await;
        assert!(output.success, "{}", output.stderr);
    }

    let token = create_key(&env, "sqlite-tenant").await;
    assert!(token.starts_with("mos_"));
    let store = SqliteStore::connect(&url).await.expect("connect");
    let key = store
        .find_api_key(&memoryos_server::auth::hash_token(&token))
        .await
        .expect("find key")
        .expect("created key");
    assert_eq!(key.tenant, "sqlite-tenant");
    assert_eq!(key.max_trust, Trust::System);
    assert!(key.is_admin);

    let output = run(&env, &["key", "revoke", &key.id.to_string()]).await;
    assert!(output.success, "{}", output.stderr);
    assert!(store
        .find_api_key(&memoryos_server::auth::hash_token(&token))
        .await
        .expect("find revoked key")
        .is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_rest_mcp_and_restart_work_without_postgres() {
    let directory = TempDir::new().expect("temporary directory");
    let url = database_url(&directory);
    let env = env(&url);
    let output = run(&env, &["migrate"]).await;
    assert!(output.success, "{}", output.stderr);
    let token = create_key(&env, "sqlite-tenant").await;

    let server = Running::start(&env).await;
    let (status, committed) = server
        .call(
            "POST",
            "/v1/commits",
            &token,
            json!({
                "scope": "agent:a1",
                "source": "chat",
                "content": "Ada lives in Bandung",
                "idempotency_key": "sqlite-restart",
                "facts": [statement("Ada lives in Bandung")]
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{committed}");
    assert_eq!(committed["facts"][0]["status"], "active");

    let recalled = recall(&server, &token).await;
    assert_eq!(
        recalled["facts"][0]["fact"]["statement"], "Ada lives in Bandung",
        "{recalled}"
    );

    let (status, tools) = server
        .call(
            "POST",
            "/mcp",
            &token,
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{tools}");
    assert!(tools["result"]["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .any(|tool| tool["name"] == "commit_facts"));

    let (status, mcp_recall) = server
        .call(
            "POST",
            "/mcp",
            &token,
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "recall",
                    "arguments": {"scopes": ["agent:a1"], "query": "Ada lives"}
                }
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{mcp_recall}");
    assert_eq!(mcp_recall["result"]["isError"], false, "{mcp_recall}");
    assert!(mcp_recall["result"]["content"][0]["text"]
        .as_str()
        .expect("MCP text")
        .contains("Ada lives in Bandung"));

    let output = server.stop().await;
    assert!(output.success, "{}", output.stderr);

    let server = Running::start(&env).await;
    let recalled = recall(&server, &token).await;
    assert_eq!(
        recalled["facts"][0]["fact"]["statement"], "Ada lives in Bandung",
        "{recalled}"
    );

    let (status, replay) = server
        .call(
            "POST",
            "/v1/commits",
            &token,
            json!({
                "scope": "agent:a1",
                "source": "chat",
                "content": "Ada lives in Bandung",
                "idempotency_key": "sqlite-restart",
                "facts": [statement("Ada lives in Bandung")]
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    assert_eq!(replay["facts"][0]["id"], committed["facts"][0]["id"]);
    let output = server.stop().await;
    assert!(output.success, "{}", output.stderr);
}
