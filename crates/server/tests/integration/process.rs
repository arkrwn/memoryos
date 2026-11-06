//! The real `memoryos` binary as a child process: startup checks, key commands, signals and logs.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use memoryos_core::{MemoryStore, Scope, Trust};
use memoryos_store::PgStore;
use reqwest::StatusCode;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};

use crate::common::{
    empty_database, env_for_host, env_for_worker, eventually, fresh_database, seed_fact, statement,
    unique, Mock,
};

const BINARY: &str = env!("CARGO_BIN_EXE_memoryos");
const SIGNAL_LIMIT: Duration = Duration::from_secs(40);

fn command(env: &[(String, String)], args: &[&str]) -> Command {
    let mut command = Command::new(BINARY);
    command
        .env_clear()
        .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
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
        .expect("the command should finish")
        .expect("run the binary");
    Output {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn db_env(url: &str) -> Vec<(String, String)> {
    vec![("DATABASE_URL".to_owned(), url.to_owned())]
}

async fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    listener.local_addr().expect("address").port()
}

async fn create_key(database_url: &str, tenant: &str, extra: &[&str]) -> String {
    let mut args = vec![
        "key",
        "create",
        "--tenant",
        tenant,
        "--name",
        "test",
        "--scopes",
        "*",
        "--max-trust",
        "system",
    ];
    args.extend_from_slice(extra);
    let output = run(&db_env(database_url), &args).await;
    assert!(output.success, "{}", output.stderr);
    output.stdout.trim().to_owned()
}

struct Running {
    child: Child,
    url: String,
    http: reqwest::Client,
}

impl Running {
    async fn start(mut env: Vec<(String, String)>, args: &[&str]) -> Self {
        let port = free_port().await;
        env.push(("MEMORYOS_BIND".to_owned(), format!("127.0.0.1:{port}")));
        let child = command(&env, args).spawn().expect("start the server");
        let running = Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
            http: reqwest::Client::new(),
        };
        running.wait_healthy().await;
        running
    }

    async fn wait_healthy(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            let health = self.http.get(format!("{}/healthz", self.url)).send().await;
            if health.is_ok_and(|r| r.status().is_success()) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("the server did not answer /healthz");
    }

    fn signal(&self, name: &str) {
        let pid = self.child.id().expect("running").to_string();
        let status = std::process::Command::new("kill")
            .args([&format!("-{name}"), &pid])
            .status()
            .expect("run kill");
        assert!(status.success());
    }

    /// Exits on the signal and returns its stdout.
    async fn stop(self, signal: &str) -> Output {
        self.signal(signal);
        let output = tokio::time::timeout(SIGNAL_LIMIT, self.child.wait_with_output())
            .await
            .expect("the process should exit after the signal")
            .expect("wait");
        Output {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_mode_fails_fast_without_the_llm_key() {
    let url = fresh_database().await;
    let mock = Mock::new().await;
    let without_anthropic: Vec<(String, String)> = env_for_worker(&mock, &url)
        .into_iter()
        .filter(|(name, _)| name != "ANTHROPIC_API_KEY")
        .collect();
    for args in [&["serve"][..], &["serve", "--no-worker"], &["worker"]] {
        let started = Instant::now();
        let output = run(&without_anthropic, args).await;
        assert!(!output.success, "{args:?} should refuse to start");
        assert!(
            output.stderr.contains("ANTHROPIC_API_KEY"),
            "{args:?}: {}",
            output.stderr
        );
        assert!(started.elapsed() < Duration::from_secs(5), "{args:?}");
    }

    // The openai provider needs its own key, and so does the default embedder.
    let without_openai: Vec<(String, String)> = env_for_worker(&mock, &url)
        .into_iter()
        .filter(|(name, _)| name != "OPENAI_API_KEY")
        .collect();
    let output = run(&without_openai, &["serve"]).await;
    assert!(!output.success);
    assert!(
        output.stderr.contains("OPENAI_API_KEY"),
        "{}",
        output.stderr
    );

    // The worker command makes no sense in host mode.
    let output = run(&env_for_host(&mock, &url), &["worker"]).await;
    assert!(!output.success);
    assert!(
        output.stderr.contains("MEMORYOS_MODE=worker"),
        "{}",
        output.stderr
    );

    // Nothing runs with no database, and the message says what is missing.
    let output = run(&[], &["serve"]).await;
    assert!(!output.success);
    assert!(output.stderr.contains("DATABASE_URL"), "{}", output.stderr);
}

#[tokio::test(flavor = "multi_thread")]
async fn migrate_and_key_commands_work_on_a_new_database() {
    let url = empty_database().await;
    let env = db_env(&url);

    let output = run(&env, &["migrate"]).await;
    assert!(output.success, "{}", output.stderr);
    let output = run(&env, &["migrate"]).await;
    assert!(
        output.success,
        "migrate must be repeatable: {}",
        output.stderr
    );

    let output = run(
        &env,
        &[
            "key",
            "create",
            "--tenant",
            "t1",
            "--name",
            "ci",
            "--scopes",
            "user:*,agent:a1",
            "--max-trust",
            "agent",
            "--admin",
        ],
    )
    .await;
    assert!(output.success, "{}", output.stderr);
    let token = output.stdout.trim().to_owned();
    assert!(token.starts_with("mos_"), "stdout holds only the token");
    assert_eq!(token.len(), 47);
    assert!(!output.stderr.contains(&token), "the token is printed once");

    let store = PgStore::connect(&url, 1).await.expect("connect");
    let key = store
        .find_api_key(&memoryos_server::auth::hash_token(&token))
        .await
        .expect("lookup")
        .expect("the key exists");
    assert_eq!(key.tenant, "t1");
    assert_eq!(key.scopes, ["user:*", "agent:a1"]);
    assert_eq!(key.max_trust, Trust::Agent);
    assert!(key.is_admin);

    let id = key.id.to_string();
    let output = run(&env, &["key", "revoke", &id]).await;
    assert!(output.success, "{}", output.stderr);
    assert!(store
        .find_api_key(&memoryos_server::auth::hash_token(&token))
        .await
        .expect("lookup")
        .is_none());

    // Bad input is refused before anything is written.
    for args in [
        vec![
            "key",
            "create",
            "--tenant",
            "t",
            "--name",
            "n",
            "--scopes",
            "team:*",
            "--max-trust",
            "agent",
        ],
        vec![
            "key",
            "create",
            "--tenant",
            "t",
            "--name",
            "n",
            "--scopes",
            "*",
            "--max-trust",
            "root",
        ],
        vec!["key", "revoke", "not-a-uuid"],
        vec!["key", "revoke", "00000000-0000-0000-0000-000000000000"],
    ] {
        let output = run(&env, &args).await;
        assert!(!output.success, "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}: {}", output.stdout);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn sigterm_completes_an_in_flight_request_and_logs_stay_clean() {
    let url = fresh_database().await;
    let mock = Mock::new().await;
    let token = create_key(&url, "signals", &[]).await;
    let secret = format!("very-private-memory-{}", unique("s"));
    let server = Running::start(env_for_host(&mock, &url), &["serve"]).await;

    // A normal request, so the log has a line for one with content in it.
    let response = server
        .http
        .post(format!("{}/v1/commits", server.url))
        .bearer_auth(&token)
        .header("x-request-id", "req-under-test")
        .json(
            &json!({"scope": "agent:a", "source": "s", "content": secret,
                      "facts": [statement(&secret, "Ada")]}),
        )
        .send()
        .await
        .expect("commit");
    assert_eq!(response.status(), StatusCode::OK);

    mock.delay_embeddings(1_500).await;
    let in_flight = tokio::spawn({
        let http = server.http.clone();
        let url = format!("{}/v1/commits", server.url);
        let token = token.clone();
        async move {
            http.post(url)
                .bearer_auth(token)
                .json(
                    &json!({"scope": "agent:a", "source": "s", "content": "slow",
                              "facts": [statement("Ada is slow", "Ada")]}),
                )
                .send()
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;

    let output = server.stop("TERM").await;
    assert!(output.success, "exit status: {}", output.stderr);
    let response = in_flight.await.expect("task").expect("response");
    assert_eq!(response.status(), StatusCode::OK);

    // Every stdout line is a JSON log, and the request line carries the id.
    let mut request_line = None;
    for line in output.stdout.lines() {
        let parsed: Value =
            serde_json::from_str(line).unwrap_or_else(|_| panic!("not a JSON log line: {line}"));
        if parsed["span"]["request_id"] == "req-under-test" {
            request_line = Some(parsed);
        }
    }
    let request_line = request_line.expect("a log line with the request id");
    assert_eq!(request_line["span"]["route"], "/v1/commits");
    assert_eq!(request_line["fields"]["status"], 200);

    // Neither the bearer token, the provider key nor memory content is logged.
    for leak in [token.as_str(), "test-openai-key", secret.as_str()] {
        assert!(!output.stdout.contains(leak), "the log holds {leak}");
        assert!(!output.stderr.contains(leak), "stderr holds {leak}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn sigint_also_stops_the_server_and_the_embedded_worker() {
    let url = fresh_database().await;
    let mock = Mock::new().await;
    let server = Running::start(env_for_worker(&mock, &url), &["serve"]).await;
    let output = server.stop("INT").await;
    assert!(output.success, "exit status: {}", output.stderr);
    assert!(
        output.stdout.contains("worker started"),
        "{}",
        output.stdout
    );
    assert!(
        output.stdout.contains("worker stopped"),
        "{}",
        output.stdout
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn no_worker_leaves_jobs_for_the_worker_command() {
    let url = fresh_database().await;
    let mock = Mock::new().await;
    mock.stub_extraction("the job text", json!([statement("Ada dances", "Ada")]))
        .await;
    let token = create_key(&url, "split", &[]).await;
    let server = Running::start(env_for_worker(&mock, &url), &["serve", "--no-worker"]).await;

    let response = server
        .http
        .post(format!("{}/v1/episodes", server.url))
        .bearer_auth(&token)
        .json(&json!({"scope": "agent:a", "source": "s", "trust": "user",
                      "content": "the job text"}))
        .send()
        .await
        .expect("episode");
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let episode: Value = response.json().await.expect("json");
    let path = format!(
        "{}/v1/episodes/{}",
        server.url,
        episode["id"].as_str().expect("id")
    );

    let read = || async {
        let response = server
            .http
            .get(&path)
            .bearer_auth(&token)
            .send()
            .await
            .expect("read");
        let body: Value = response.json().await.expect("json");
        body["processed_at"].is_string()
    };
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(!read().await, "nothing should process it under --no-worker");
    assert_eq!(mock.llm_calls().await, 0);

    let worker = command(&env_for_worker(&mock, &url), &["worker"])
        .spawn()
        .expect("start the worker");
    eventually("the worker command to process the job", read).await;

    let pid = worker.id().expect("running").to_string();
    let status = std::process::Command::new("kill")
        .args(["-TERM", &pid])
        .status()
        .expect("kill");
    assert!(status.success());
    let output = tokio::time::timeout(SIGNAL_LIMIT, worker.wait_with_output())
        .await
        .expect("the worker should exit")
        .expect("wait");
    assert!(output.status.success());
    server.stop("TERM").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn otlp_traces_are_exported_only_when_configured() {
    let url = fresh_database().await;
    let mock = Mock::new().await;
    mock.stub(json!({
        "request": {"method": "POST", "url": format!("/{}/v1/traces", mock.prefix)},
        "response": {"status": 200},
    }))
    .await;
    let token = create_key(&url, "otlp", &[]).await;

    // Without the variable, nothing is sent.
    let server = Running::start(env_for_host(&mock, &url), &["serve"]).await;
    let response = server
        .http
        .get(format!("{}/v1/agents/a/blocks", server.url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(server.stop("TERM").await.success);
    assert_eq!(mock.count("/v1/traces").await, 0);

    let mut env = env_for_host(&mock, &url);
    env.push(("OTEL_EXPORTER_OTLP_ENDPOINT".to_owned(), mock.base()));
    let server = Running::start(env, &["serve"]).await;
    let response = server
        .http
        .get(format!("{}/v1/agents/a/blocks", server.url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    let output = server.stop("TERM").await;
    assert!(output.success, "exit status: {}", output.stderr);
    assert!(
        mock.count("/v1/traces").await >= 1,
        "shutdown should flush spans: {}",
        output.stderr
    );
}

/// A real `LocalEmbedder` whose model host never answers. No model is downloaded.
#[tokio::test(flavor = "multi_thread")]
async fn a_stalled_model_download_blocks_neither_health_nor_the_api_nor_shutdown() {
    let url = fresh_database().await;
    let mock = Mock::new().await;
    mock.stub(json!({
        "request": {"method": "ANY", "urlPattern": format!("/{}/models/.*", mock.prefix)},
        "response": {"status": 200, "fixedDelayMilliseconds": 120_000},
    }))
    .await;
    let token = create_key(&url, "local", &[]).await;

    // Text search can answer without the model, so seed a fact directly.
    let store = PgStore::connect(&url, 2).await.expect("connect");
    seed_fact(
        &store,
        "local",
        &Scope::agent("a"),
        Trust::User,
        "Ada sails boats",
        None,
    )
    .await;

    let cache = model_cache_dir();
    let env: Vec<(String, String)> = [
        ("DATABASE_URL", url.as_str()),
        ("MEMORYOS_MODE", "host"),
        ("HF_ENDPOINT", &format!("{}/models", mock.base())),
        ("MEMORYOS_MODEL_CACHE", cache.to_str().expect("path")),
        ("MEMORYOS_REQUEST_TIMEOUT_SECS", "4"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
    .collect();

    let started = Instant::now();
    let server = Running::start(env, &["serve"]).await;
    assert!(
        started.elapsed() < Duration::from_secs(15),
        "the API must not wait for the model"
    );

    let ready = server
        .http
        .get(format!("{}/readyz", server.url))
        .send()
        .await
        .expect("readyz");
    assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: Value = ready.json().await.expect("json");
    assert_ne!(body["embedder"], "ready", "{body}");
    assert_eq!(body["database"], "ok", "{body}");

    // Recall answers through text search once the embed timeout passes.
    let started = Instant::now();
    let response = server
        .http
        .post(format!("{}/v1/recall", server.url))
        .bearer_auth(&token)
        .json(&json!({"scopes": ["agent:a"], "query": "sails"}))
        .send()
        .await
        .expect("recall");
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json");
    assert_eq!(body["facts"][0]["fact"]["statement"], "Ada sails boats");
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "{:?}",
        started.elapsed()
    );

    // A commit needs the model, so it hits the time limit rather than hanging.
    let response = server
        .http
        .post(format!("{}/v1/commits", server.url))
        .bearer_auth(&token)
        .json(&json!({"scope": "agent:a", "source": "s", "content": "c",
                      "facts": [statement("Ada waits", "Ada")]}))
        .send()
        .await
        .expect("commit");
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);

    // Proves the attempt went to the stub, so this test cannot download the real model.
    assert!(
        mock.count("/models/.*").await >= 1,
        "the model download should have reached the stalled host"
    );

    // The blocking download thread cannot be cancelled, and shutdown does not wait for it.
    let started = Instant::now();
    let output = server.stop("TERM").await;
    assert!(output.success, "{}", output.stderr);
    assert!(
        started.elapsed() < Duration::from_secs(15),
        "shutdown took {:?}",
        started.elapsed()
    );
    assert!(
        output.stdout.contains("loading the local embedding model"),
        "{}",
        output.stdout
    );
    let _ = std::fs::remove_dir_all(cache);
}

fn model_cache_dir() -> PathBuf {
    std::env::temp_dir().join(unique("memoryos-model-cache"))
}
