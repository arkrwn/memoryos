//! One Postgres container per test binary. Each test gets its own database, cloned from a migrated
//! template, so tests run in parallel without sharing rows. The router is real and listens on a real
//! socket. Embeddings come from the test embedder, selected by `MEMORYOS_TEST_EMBEDDER`.

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use memoryos_core::{
    FactOp, FactStatus, MemoryStore, NewEntity, NewEpisode, NewFact, RequestContext, Scope, Trust,
    EMBEDDING_DIMS,
};
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
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
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
}

fn containers() -> &'static Containers {
    static SHARED: OnceLock<Containers> = OnceLock::new();
    SHARED.get_or_init(|| {
        std::thread::spawn(|| {
            tokio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(async {
                    Containers {
                        postgres_url: start_postgres().await,
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

// ---- A plain HTTP endpoint ---------------------------------------------------------------------

/// Records the path of every request it reads. It answers each with 200, or with `stalled` never
/// answers at all, which is what a model host that hangs looks like.
pub struct HttpEndpoint {
    pub url: String,
    paths: Arc<Mutex<Vec<String>>>,
}

impl HttpEndpoint {
    pub async fn answering() -> Self {
        Self::start(true).await
    }

    pub async fn stalled() -> Self {
        Self::start(false).await
    }

    async fn start(answer: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let paths = Arc::new(Mutex::new(Vec::new()));
        let recorded = paths.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(serve_requests(stream, recorded.clone(), answer));
            }
        });
        Self {
            url: format!("http://{address}"),
            paths,
        }
    }

    pub fn requests(&self) -> Vec<String> {
        self.paths.lock().expect("paths").clone()
    }
}

async fn serve_requests(mut stream: TcpStream, paths: Arc<Mutex<Vec<String>>>, answer: bool) {
    let mut buffer = Vec::new();
    let mut chunk = [0; 4096];
    loop {
        // One complete request is in the buffer once the head has ended and the body has arrived.
        let head_end = buffer.windows(4).position(|window| window == b"\r\n\r\n");
        if let Some(head_end) = head_end {
            let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
            let body_length = head
                .lines()
                .filter_map(|line| line.split_once(':'))
                .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, value)| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buffer.len() >= head_end + 4 + body_length {
                let path = head
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_owned();
                paths.lock().expect("paths").push(path);
                if !answer {
                    std::future::pending::<()>().await;
                }
                let reply = "HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n";
                if stream.write_all(reply.as_bytes()).await.is_err() {
                    return;
                }
                buffer.drain(..head_end + 4 + body_length);
                continue;
            }
        }
        match stream.read(&mut chunk).await {
            Ok(read) if read > 0 => buffer.extend_from_slice(&chunk[..read]),
            _ => return,
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

pub struct Key {
    pub id: Uuid,
    pub token: String,
}

/// What the binary needs to run here, with the test embedder and no model.
pub fn test_env(database_url: &str) -> Vec<(String, String)> {
    pairs(&[
        ("DATABASE_URL", database_url),
        ("MEMORYOS_TEST_EMBEDDER", "0"),
        ("MEMORYOS_DB_MAX_CONNECTIONS", "8"),
    ])
}

/// The same, with every embedding call taking `millis`.
pub fn slow_env(database_url: &str, millis: &str) -> Vec<(String, String)> {
    let mut env = test_env(database_url);
    env.push(("MEMORYOS_TEST_EMBEDDER".to_owned(), millis.to_owned()));
    env
}

pub fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

// ---- Server ----------------------------------------------------------------------------------

pub struct Server {
    pub url: String,
    pub database_url: String,
    pub store: Arc<PgStore>,
    http: reqwest::Client,
    shutdown: CancellationToken,
    task: JoinHandle<Result<(), String>>,
}

impl Server {
    pub async fn start() -> Self {
        Self::start_with(&[]).await
    }

    pub async fn start_with(extra: &[(&str, &str)]) -> Self {
        let database_url = fresh_database().await;
        let mut env = test_env(&database_url);
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
        let task = tokio::spawn(app::serve(config, listener, shutdown.clone()));
        let store = PgStore::connect(&database_url, 3)
            .await
            .expect("test store");
        let server = Self {
            url: format!("http://{address}"),
            database_url,
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

    /// Writes a fact directly, for trust levels commits cannot reach.
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
    let (episode, _) = store
        .claim_episode(
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
        .apply_commit(&ctx, episode.id, vec![op])
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

pub fn random_uuid() -> Uuid {
    Uuid::now_v7()
}
