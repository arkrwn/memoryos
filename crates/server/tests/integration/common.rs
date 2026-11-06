//! One Postgres and one WireMock container per test binary. Each test gets its own database, cloned
//! from a migrated template, and its own URL prefix on WireMock, so tests run in parallel without
//! sharing jobs, rows or stubs. The router is real and listens on a real socket.

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use memoryos_core::{
    FactOp, FactStatus, MemoryStore, NewEntity, NewEpisode, NewFact, RequestContext, Scope, Trust,
    EMBEDDING_DIMS,
};
use memoryos_ingest::CancellationToken;
use memoryos_sdk::{MemoryOs, Options};
use memoryos_server::app;
use memoryos_server::auth::{generate_token, hash_token};
use memoryos_server::config::Config;
use memoryos_store::PgStore;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{GenericImage, ImageExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use uuid::Uuid;

const TEMPLATE_DB: &str = "memoryos_template";

static NEXT: AtomicUsize = AtomicUsize::new(0);

pub fn unique(prefix: &str) -> String {
    format!(
        "{prefix}_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    )
}

// ---- Containers ------------------------------------------------------------------------------

struct Containers {
    postgres_url: String,
    wiremock_url: String,
}

fn containers() -> &'static Containers {
    static SHARED: OnceLock<Containers> = OnceLock::new();
    SHARED.get_or_init(|| {
        std::thread::spawn(|| {
            tokio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(async {
                    let (postgres_url, wiremock_url) =
                        tokio::join!(start_postgres(), start_wiremock());
                    Containers {
                        postgres_url,
                        wiremock_url,
                    }
                })
        })
        .join()
        .expect("container thread")
    })
}

async fn start_postgres() -> String {
    let container = GenericImage::new("pgvector/pgvector", "pg17")
        .with_exposed_port(5432.tcp())
        .with_wait_for(WaitFor::message_on_stderr(
            "database system is ready to accept connections",
        ))
        .with_env_var("POSTGRES_PASSWORD", "postgres")
        // The image declares a volume here. Without this mount each run leaks an anonymous volume,
        // because the reaper's `docker rm -f` does not remove volumes.
        .with_mount(Mount::tmpfs_mount("/var/lib/postgresql/data"))
        .with_cmd(["postgres", "-c", "max_connections=500", "-c", "fsync=off"])
        .start()
        .await
        .expect("start pgvector container");
    let port = container.get_host_port_ipv4(5432).await.expect("port");
    reap_on_exit(container.id());
    // A static would never drop the handle anyway. The reaper removes the container.
    std::mem::forget(container);

    let url = format!("postgres://postgres:postgres@127.0.0.1:{port}");
    let admin = connect_admin(&url).await;
    sqlx::query(&format!("CREATE DATABASE {TEMPLATE_DB}"))
        .execute(&admin)
        .await
        .expect("create template");
    let store = PgStore::connect(&format!("{url}/{TEMPLATE_DB}"), 2)
        .await
        .expect("connect template");
    store.migrate().await.expect("migrate template");
    drop(store);
    // The template must have no connections when it is cloned. Pool shutdown is asynchronous.
    for _ in 0..100 {
        let open: i64 =
            sqlx::query_scalar("SELECT count(*) FROM pg_stat_activity WHERE datname = $1")
                .bind(TEMPLATE_DB)
                .fetch_one(&admin)
                .await
                .expect("count connections");
        if open == 0 {
            break;
        }
        sqlx::query("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1")
            .bind(TEMPLATE_DB)
            .execute(&admin)
            .await
            .expect("terminate");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    admin.close().await;
    url
}

async fn connect_admin(url: &str) -> PgPool {
    for _ in 0..120 {
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .connect(&format!("{url}/postgres"))
            .await;
        if let Ok(pool) = pool {
            return pool;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("postgres did not accept connections");
}

async fn start_wiremock() -> String {
    let container = GenericImage::new("wiremock/wiremock", "3.3.1")
        .with_exposed_port(8080.tcp())
        .with_wait_for(WaitFor::message_on_stdout("port:"))
        .with_cmd(["--global-response-templating"])
        .start()
        .await
        .expect("start wiremock container");
    let port = container.get_host_port_ipv4(8080).await.expect("port");
    reap_on_exit(container.id());
    std::mem::forget(container);

    let url = format!("http://127.0.0.1:{port}");
    let http = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if let Ok(response) = http.get(format!("{url}/__admin/mappings")).send().await {
            if response.status().is_success() {
                return url;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("wiremock admin API did not become ready");
}

/// Removes the container when this test binary exits, however it exits.
#[allow(clippy::zombie_processes)] // The child must outlive this process, so it is never waited on.
fn reap_on_exit(container_id: &str) {
    let script = format!(
        "while kill -0 {pid} 2>/dev/null; do sleep 1; done; docker rm -f {container_id}",
        pid = std::process::id()
    );
    Command::new("sh")
        .args(["-c", &script])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn reaper");
}

/// A fresh, migrated database. Returns its URL.
pub async fn fresh_database() -> String {
    let base = &containers().postgres_url;
    let name = unique("db");
    let admin = connect_admin(base).await;
    sqlx::query(&format!("CREATE DATABASE {name} TEMPLATE {TEMPLATE_DB}"))
        .execute(&admin)
        .await
        .expect("clone template");
    admin.close().await;
    format!("{base}/{name}")
}

/// A database with no tables, for `memoryos migrate`.
pub async fn empty_database() -> String {
    let base = &containers().postgres_url;
    let name = unique("empty");
    let admin = connect_admin(base).await;
    sqlx::query(&format!("CREATE DATABASE {name}"))
        .execute(&admin)
        .await
        .expect("create database");
    admin.close().await;
    format!("{base}/{name}")
}

pub async fn drop_database(url: &str) {
    let base = &containers().postgres_url;
    let name = url.rsplit('/').next().expect("database name");
    let admin = connect_admin(base).await;
    sqlx::query(&format!("DROP DATABASE {name} WITH (FORCE)"))
        .execute(&admin)
        .await
        .expect("drop database");
    admin.close().await;
}

// ---- WireMock --------------------------------------------------------------------------------

/// This test's corner of WireMock. Every path it stubs starts with `prefix`.
pub struct Mock {
    pub prefix: String,
    http: reqwest::Client,
}

impl Mock {
    pub async fn new() -> Self {
        let mock = Self {
            prefix: unique("t"),
            http: reqwest::Client::new(),
        };
        mock.stub_embeddings().await;
        mock
    }

    fn admin(&self, path: &str) -> String {
        format!("{}/__admin/{path}", containers().wiremock_url)
    }

    /// With `/v1`, the form `OPENAI_BASE_URL` takes.
    pub fn openai_base(&self) -> String {
        format!("{}/{}/v1", containers().wiremock_url, self.prefix)
    }

    /// Without `/v1`, the form `ANTHROPIC_BASE_URL` takes.
    pub fn base(&self) -> String {
        format!("{}/{}", containers().wiremock_url, self.prefix)
    }

    pub fn messages_path(&self) -> String {
        format!("/{}/v1/messages", self.prefix)
    }

    pub fn embeddings_path(&self) -> String {
        format!("/{}/v1/embeddings", self.prefix)
    }

    pub async fn stub(&self, mapping: Value) {
        let response = self
            .http
            .post(self.admin("mappings"))
            .json(&mapping)
            .send()
            .await
            .expect("register stub");
        assert_eq!(response.status().as_u16(), 201, "stub rejected");
    }

    /// One vector per input, whatever the batch size. Random, so they are all far apart and no two
    /// entity names merge by similarity. Tests that need a result to be found rely on text search.
    async fn stub_embeddings(&self) {
        let parts: Vec<&str> = (0..EMBEDDING_DIMS)
            .map(|_| "{{randomDecimal lower=-1.0 upper=1.0}}")
            .collect();
        let vector = parts.join(",");
        let body = format!(
            "{{\"data\":[{{{{#each (jsonPath request.body '$.input') as |item|}}}}\
             {{{{#unless @first}}}},{{{{/unless}}}}\
             {{\"index\":{{{{@index}}}},\"embedding\":[{vector}]}}\
             {{{{/each}}}}]}}"
        );
        self.stub(json!({
            "priority": 9,
            "request": {"method": "POST", "url": self.embeddings_path()},
            "response": {
                "status": 200,
                "headers": {"Content-Type": "application/json"},
                "body": body,
            },
        }))
        .await;
    }

    /// Answers every embedding call after `millis`.
    pub async fn delay_embeddings(&self, millis: u64) {
        let parts = vec!["0.5"; EMBEDDING_DIMS];
        let body = format!(
            "{{\"data\":[{{{{#each (jsonPath request.body '$.input') as |item|}}}}\
             {{{{#unless @first}}}},{{{{/unless}}}}\
             {{\"index\":{{{{@index}}}},\"embedding\":[{}]}}\
             {{{{/each}}}}]}}",
            parts.join(",")
        );
        self.stub(json!({
            "priority": 1,
            "request": {"method": "POST", "url": self.embeddings_path()},
            "response": {
                "status": 200,
                "fixedDelayMilliseconds": millis,
                "headers": {"Content-Type": "application/json"},
                "body": body,
            },
        }))
        .await;
    }

    /// Requests received under this test's prefix whose path matches `pattern`.
    pub async fn count(&self, pattern: &str) -> u64 {
        let found: Value = self
            .http
            .post(self.admin("requests/count"))
            .json(&json!({"method": "ANY", "urlPathPattern": format!("/{}{pattern}", self.prefix)}))
            .send()
            .await
            .expect("count requests")
            .json()
            .await
            .expect("count body");
        found["count"].as_u64().expect("count")
    }

    /// Any model call, to either provider.
    pub async fn llm_calls(&self) -> u64 {
        self.count("/v1/(messages|chat/completions)").await
    }

    /// A forced call to `tool` whose request contains `text`.
    pub async fn stub_forced(&self, tool: &str, text: &str, input: Value) {
        self.stub(json!({
            "request": {
                "method": "POST",
                "url": self.messages_path(),
                "bodyPatterns": [
                    {"equalToJson": {"tool_choice": {"type": "tool", "name": tool}},
                     "ignoreExtraElements": true},
                    {"contains": text},
                ],
            },
            "response": tool_use_reply(tool, input),
        }))
        .await;
    }

    pub async fn stub_extraction(&self, text: &str, statements: Value) {
        self.stub_forced("record_extraction", text, json!({"statements": statements}))
            .await;
    }

    /// Replies to model calls in order, for requests whose body contains `marker`.
    pub async fn script(&self, marker: &str, steps: Vec<Value>) {
        let scenario = unique("scenario");
        for (index, step) in steps.into_iter().enumerate() {
            let state = if index == 0 {
                "Started".to_owned()
            } else {
                format!("step-{index}")
            };
            self.stub(json!({
                "scenarioName": scenario,
                "requiredScenarioState": state,
                "newScenarioState": format!("step-{}", index + 1),
                "request": {
                    "method": "POST",
                    "url": self.messages_path(),
                    "bodyPatterns": [{"contains": marker}],
                },
                "response": step,
            }))
            .await;
        }
    }
}

pub fn statement(text: &str, subject: &str) -> Value {
    json!({
        "statement": text,
        "subject": {"name": subject, "kind": "person"},
        "predicate": "fact_about",
    })
}

pub fn tool_use_reply(tool: &str, input: Value) -> Value {
    json!({
        "status": 200,
        "transformers": [],
        "jsonBody": {
            "content": [{"type": "tool_use", "id": "toolu_1", "name": tool, "input": input}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    })
}

pub fn text_reply(text: &str) -> Value {
    json!({
        "status": 200,
        "jsonBody": {
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    })
}

// ---- Server ----------------------------------------------------------------------------------

pub struct Server {
    pub url: String,
    pub database_url: String,
    pub mock: Mock,
    pub store: Arc<PgStore>,
    http: reqwest::Client,
    shutdown: CancellationToken,
    task: JoinHandle<Result<(), String>>,
}

#[derive(Clone)]
pub struct Key {
    pub id: Uuid,
    pub token: String,
}

pub fn env_for_host(mock: &Mock, database_url: &str) -> Vec<(String, String)> {
    pairs(&[
        ("DATABASE_URL", database_url),
        ("MEMORYOS_MODE", "host"),
        ("MEMORYOS_EMBEDDER", "openai"),
        ("OPENAI_API_KEY", "test-openai-key"),
        ("OPENAI_BASE_URL", &mock.openai_base()),
        ("MEMORYOS_DB_MAX_CONNECTIONS", "8"),
    ])
}

pub fn env_for_worker(mock: &Mock, database_url: &str) -> Vec<(String, String)> {
    pairs(&[
        ("DATABASE_URL", database_url),
        ("MEMORYOS_MODE", "worker"),
        ("MEMORYOS_EMBEDDER", "openai"),
        ("OPENAI_API_KEY", "test-openai-key"),
        ("OPENAI_BASE_URL", &mock.openai_base()),
        ("ANTHROPIC_API_KEY", "test-anthropic-key"),
        ("ANTHROPIC_BASE_URL", &mock.base()),
        ("MEMORYOS_DB_MAX_CONNECTIONS", "8"),
    ])
}

fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

impl Server {
    pub async fn host() -> Self {
        Self::start(true, &[], false).await
    }

    pub async fn worker() -> Self {
        Self::start(false, &[], true).await
    }

    pub async fn host_with(extra: &[(&str, &str)]) -> Self {
        Self::start(true, extra, false).await
    }

    pub async fn worker_without_loop() -> Self {
        Self::start(false, &[], false).await
    }

    async fn start(host: bool, extra: &[(&str, &str)], run_worker: bool) -> Self {
        let mock = Mock::new().await;
        let database_url = fresh_database().await;
        let mut env = if host {
            env_for_host(&mock, &database_url)
        } else {
            env_for_worker(&mock, &database_url)
        };
        env.extend(pairs(extra));
        let config = Config::from_lookup(|name| {
            env.iter()
                .rev()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        })
        .expect("test configuration");

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let shutdown = CancellationToken::new();
        let task = tokio::spawn(app::serve(config, listener, run_worker, shutdown.clone()));
        let store = PgStore::connect(&database_url, 3)
            .await
            .expect("test store");
        let server = Self {
            url: format!("http://{address}"),
            database_url,
            mock,
            store: Arc::new(store),
            http: reqwest::Client::new(),
            shutdown,
            task,
        };
        server.wait_until_up().await;
        server
    }

    async fn wait_until_up(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            let health = self.http.get(format!("{}/healthz", self.url)).send().await;
            if health.is_ok_and(|r| r.status().is_success()) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the server did not come up");
    }

    pub async fn stop(self) -> Result<(), String> {
        self.shutdown.cancel();
        self.task.await.expect("server task")
    }

    pub async fn key(&self, tenant: &str, scopes: &[&str], max_trust: Trust, admin: bool) -> Key {
        let token = generate_token().expect("token");
        let scopes: Vec<String> = scopes.iter().map(|s| (*s).to_owned()).collect();
        let key = self
            .store
            .create_api_key(
                tenant,
                &unique("key"),
                &hash_token(&token),
                &scopes,
                max_trust,
                admin,
            )
            .await
            .expect("create key");
        Key { id: key.id, token }
    }

    /// `*` on every scope, admin, system trust. For setup.
    pub async fn root(&self, tenant: &str) -> Key {
        self.key(tenant, &["*"], Trust::System, true).await
    }

    pub fn sdk(&self, key: &Key) -> MemoryOs {
        MemoryOs::new(&self.url, &key.token, Options::default()).expect("sdk client")
    }

    /// Sends one request and returns status, headers and the body parsed as JSON, or null.
    pub async fn call(
        &self,
        method: &str,
        path: &str,
        key: Option<&Key>,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let (status, _, value) = self.call_full(method, path, key, body).await;
        (status, value)
    }

    pub async fn call_full(
        &self,
        method: &str,
        path: &str,
        key: Option<&Key>,
        body: Option<Value>,
    ) -> (StatusCode, reqwest::header::HeaderMap, Value) {
        let method = Method::from_bytes(method.as_bytes()).expect("method");
        let mut request = self.http.request(method, format!("{}{path}", self.url));
        if let Some(key) = key {
            request = request.bearer_auth(&key.token);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.expect("send request");
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response.bytes().await.expect("read body");
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, headers, value)
    }

    pub fn raw_client(&self) -> &reqwest::Client {
        &self.http
    }

    /// Commits one fact and returns its id, status and episode id.
    pub async fn commit_one(
        &self,
        key: &Key,
        scope: &str,
        statement_text: &str,
    ) -> (Uuid, String, Uuid) {
        let (status, body) = self
            .call(
                "POST",
                "/v1/commits",
                Some(key),
                Some(json!({
                    "scope": scope,
                    "source": "test",
                    "content": statement_text,
                    "facts": [statement(statement_text, "Ada")],
                })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        (
            uuid_at(&body, "/facts/0/id"),
            body["facts"][0]["status"]
                .as_str()
                .expect("status")
                .to_owned(),
            uuid_at(&body, "/episode_id"),
        )
    }

    pub async fn fact_json(&self, key: &Key, id: Uuid) -> Value {
        let (status, body) = self
            .call("GET", &format!("/v1/facts/{id}"), Some(key), None)
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    pub async fn recall_statements(&self, key: &Key, scope: &str, query: &str) -> Vec<String> {
        let (status, body) = self
            .call(
                "POST",
                "/v1/recall",
                Some(key),
                Some(json!({"scopes": [scope], "query": query})),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        statements_in(&body)
    }

    /// Writes a fact directly, for trust levels the API cannot reach in the mode under test.
    pub async fn seed_fact(
        &self,
        tenant: &str,
        scope: &Scope,
        trust: Trust,
        text: &str,
        supersedes: Option<Uuid>,
    ) -> Uuid {
        seed_fact(self.store.as_ref(), tenant, scope, trust, text, supersedes).await
    }
}

pub async fn seed_fact(
    store: &dyn MemoryStore,
    tenant: &str,
    scope: &Scope,
    trust: Trust,
    text: &str,
    supersedes: Option<Uuid>,
) -> Uuid {
    let ctx = RequestContext {
        tenant: tenant.to_owned(),
        actor: "test".to_owned(),
    };
    let episode = store
        .insert_unqueued_episode(
            &ctx,
            NewEpisode {
                scope: scope.clone(),
                source: "seed".to_owned(),
                trust,
                content: text.to_owned(),
                occurred_at: None,
                metadata: json!({}),
                idempotency_key: None,
            },
        )
        .await
        .expect("seed episode");
    let mut vector = vec![0.1f32; EMBEDDING_DIMS];
    vector[0] = 1.0;
    let subject = store
        .upsert_entity(
            &ctx,
            NewEntity {
                scope: scope.clone(),
                kind: "person".to_owned(),
                name: "Ada".to_owned(),
                aliases: vec![],
                embedding: vector.clone(),
            },
        )
        .await
        .expect("seed entity");
    let new = NewFact {
        scope: scope.clone(),
        subject_id: subject.id,
        predicate: "fact_about".to_owned(),
        object_id: None,
        statement: text.to_owned(),
        embedding: vector,
        trust,
        status: FactStatus::Active,
        valid_from: episode.occurred_at,
        valid_to: None,
    };
    let op = match supersedes {
        Some(old_fact_id) => FactOp::Supersede { old_fact_id, new },
        None => FactOp::Add(new),
    };
    let ids = store
        .apply_extraction(&ctx, episode.id, vec![op])
        .await
        .expect("seed fact");
    ids[0]
}

pub fn uuid_at(body: &Value, pointer: &str) -> Uuid {
    body.pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("no string at {pointer} in {body}"))
        .parse()
        .expect("uuid")
}

pub fn statements_in(body: &Value) -> Vec<String> {
    body["facts"]
        .as_array()
        .expect("facts array")
        .iter()
        .map(|item| {
            item["fact"]["statement"]
                .as_str()
                .expect("statement")
                .to_owned()
        })
        .collect()
}

/// Every error must have exactly the SPEC section 9 shape.
#[track_caller]
pub fn assert_error(status: StatusCode, body: &Value, expected: u16, code: &str) {
    assert_eq!(status.as_u16(), expected, "{body}");
    assert_eq!(body["error"]["code"], code, "{body}");
    assert!(body["error"]["message"].is_string(), "{body}");
    assert_eq!(
        body.as_object().map(serde_json::Map::len),
        Some(1),
        "{body}"
    );
    assert_eq!(
        body["error"].as_object().map(serde_json::Map::len),
        Some(2),
        "{body}"
    );
}

/// Waits for a condition the worker makes true.
pub async fn eventually<F, Fut>(what: &str, mut check: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if check().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("timed out waiting for {what}");
}

pub fn random_uuid() -> Uuid {
    Uuid::now_v7()
}
