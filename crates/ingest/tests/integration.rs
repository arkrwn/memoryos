//! The write path against real PostgreSQL, with the deterministic test embedder. One container per
//! test binary. Each test clones a migrated template into its own database, so tests run in parallel.

#![cfg(feature = "integration")]

mod common;

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use memoryos_core::{
    Error, Fact, FactFilter, FactOp, FactStatus, MemoryStore, NewEntity, NewEpisode, NewFact,
    RequestContext, Scope, Trust, EMBEDDING_DIMS,
};
use memoryos_embed::{vector_for, TestEmbedder};
use memoryos_ingest::{
    CommitFact, CommitRequest, CommitResponse, CommittedFact, EntityRef, HostCommitter,
    SingleValued,
};
use memoryos_store::PgStore;
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use sqlx::{AssertSqlSafe, PgPool};
use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{GenericImage, ImageExt};
use uuid::Uuid;

const TEMPLATE_DB: &str = "memoryos_template";

/// `claim_episode` for a test that does not care which call created the row.
trait InsertEpisode {
    async fn insert_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> memoryos_core::Result<memoryos_core::Episode>;
}

impl InsertEpisode for PgStore {
    async fn insert_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> memoryos_core::Result<memoryos_core::Episode> {
        self.claim_episode(ctx, episode).await.map(|(row, _)| row)
    }
}

// ---- Containers ------------------------------------------------------------------------------

fn postgres_url() -> &'static str {
    static URL: OnceLock<String> = OnceLock::new();
    // A `OnceLock` initializer cannot await, and the test's runtime cannot block on itself.
    URL.get_or_init(|| {
        std::thread::spawn(|| {
            tokio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(start_postgres())
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
    let port = container
        .get_host_port_ipv4(5432)
        .await
        .expect("mapped port");
    reap_on_exit(container.id());
    // A static would never drop the handle anyway. The reaper removes the container.
    std::mem::forget(container);

    let url = format!("postgres://postgres:postgres@127.0.0.1:{port}");
    let admin = connect_with_retry(&format!("{url}/postgres")).await;
    sqlx::query(AssertSqlSafe(format!("CREATE DATABASE {TEMPLATE_DB}")))
        .execute(&admin)
        .await
        .expect("create template");
    admin.close().await;
    let store = PgStore::connect(&format!("{url}/{TEMPLATE_DB}"), 2)
        .await
        .expect("connect template");
    store.migrate().await.expect("migrate template");
    drop(store);
    // The template must have no connections when it is cloned. Pool shutdown is asynchronous.
    let admin = connect_with_retry(&format!("{url}/postgres")).await;
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

async fn connect_with_retry(url: &str) -> PgPool {
    for _ in 0..120 {
        if let Ok(pool) = PgPoolOptions::new().max_connections(2).connect(url).await {
            return pool;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("postgres did not accept connections");
}

// ---- Test world ------------------------------------------------------------------------------

/// One test's database and its embedder.
struct World {
    store: Arc<PgStore>,
    admin: PgPool,
    embedder: Arc<TestEmbedder>,
    host: HostCommitter,
    name: String,
}

/// Drops the test's database when the test ends. The container keeps its data in memory, in a
/// Docker VM that other containers share, and each database is about 8 MB: kept to the end, a
/// run's databases fill the VM's memory and new connections stall. On a thread with its own
/// runtime, because a drop cannot await.
impl Drop for World {
    fn drop(&mut self) {
        let server = postgres_url().to_owned();
        let sql = format!("DROP DATABASE IF EXISTS {} WITH (FORCE)", self.name);
        let dropped = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(async {
                    let pool = connect_with_retry(&format!("{server}/postgres")).await;
                    let result = sqlx::query(AssertSqlSafe(sql)).execute(&pool).await;
                    pool.close().await;
                    result
                })
        })
        .join();
        // A test that already failed must keep its own message, so this never panics while
        // unwinding. A database left behind costs memory, not correctness.
        if !std::thread::panicking() {
            dropped
                .expect("drop thread")
                .expect("drop the test database");
        }
    }
}

async fn world() -> World {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let server = postgres_url();
    let name = format!("t{}", NEXT.fetch_add(1, Ordering::SeqCst));

    let bootstrap = connect_with_retry(&format!("{server}/postgres")).await;
    sqlx::query(AssertSqlSafe(format!(
        "CREATE DATABASE {name} TEMPLATE {TEMPLATE_DB}"
    )))
    .execute(&bootstrap)
    .await
    .expect("clone template");
    bootstrap.close().await;
    let url = format!("{server}/{name}");
    let store = Arc::new(PgStore::connect(&url, 8).await.expect("connect"));
    let embedder = Arc::new(TestEmbedder::new());
    World {
        host: HostCommitter::new(store.clone(), embedder.clone()),
        store,
        admin: connect_with_retry(&url).await,
        embedder,
        name,
    }
}

fn ctx() -> RequestContext {
    tenant_ctx("acme")
}

fn tenant_ctx(tenant: &str) -> RequestContext {
    RequestContext {
        tenant: tenant.into(),
        actor: "tester".into(),
    }
}

fn date(year: i32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0)
        .single()
        .expect("valid date")
}

/// A vector whose leading components are `values`, and zero after that.
fn sparse(values: &[f32]) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIMS];
    vector[..values.len()].copy_from_slice(values);
    vector
}

impl World {
    async fn count(&self, table: &str) -> i64 {
        sqlx::query_scalar(AssertSqlSafe(format!("SELECT count(*) FROM {table}")))
            .fetch_one(&self.admin)
            .await
            .expect("count")
    }

    async fn recall_ids(
        &self,
        scope: &Scope,
        query: &str,
        as_of: Option<DateTime<Utc>>,
    ) -> Vec<Uuid> {
        let filter = FactFilter {
            as_of,
            known_at: None,
        };
        self.store
            .search_facts_vector(
                &ctx(),
                std::slice::from_ref(scope),
                &vector_for(query),
                &filter,
                20,
            )
            .await
            .expect("search")
            .into_iter()
            .map(|scored| scored.fact.id)
            .collect()
    }

    /// An active fact stored directly, because commits cannot go above `agent` trust.
    async fn seed_fact(
        &self,
        scope: &Scope,
        trust: Trust,
        statement: &str,
        subject: &str,
        subject_embedding: Vec<f32>,
    ) -> Fact {
        let episode = self
            .store
            .insert_episode(
                &ctx(),
                NewEpisode {
                    scope: scope.clone(),
                    source: "seed".into(),
                    trust,
                    content: statement.into(),
                    occurred_at: Some(date(2020)),
                    metadata: json!({}),
                    idempotency_key: None,
                },
            )
            .await
            .expect("insert episode");
        let subject_id = self.seed_entity(scope, subject, subject_embedding).await;
        let ids = self
            .store
            .apply_commit(
                &ctx(),
                episode.id,
                vec![FactOp::Add(NewFact {
                    scope: scope.clone(),
                    subject_id,
                    predicate: "fact_about".into(),
                    object_id: None,
                    statement: statement.into(),
                    embedding: vector_for(statement),
                    trust,
                    status: FactStatus::Active,
                    valid_from: date(2020),
                    valid_to: None,
                    expires_at: None,
                })],
            )
            .await
            .expect("apply");
        self.store.get_fact(&ctx(), ids[0]).await.expect("fact")
    }

    /// A current fact on Robert. The vector is what names merge against.
    async fn seed_on_robert(&self, scope: &Scope, trust: Trust, text: &str) -> Fact {
        self.seed_fact(scope, trust, text, "Robert", sparse(&[1.0]))
            .await
    }
}

// ---- Commits ---------------------------------------------------------------------------------

fn add_fact(text: &str, subject: &str) -> CommitFact {
    CommitFact {
        statement: text.into(),
        subject: EntityRef {
            name: subject.into(),
            kind: "person".into(),
        },
        predicate: "fact_about".into(),
        object: None,
        valid_from: None,
        expires_at: None,
        supersedes: None,
        corroborates: None,
    }
}

fn commit_request(scope: &Scope, content: &str, facts: Vec<CommitFact>) -> CommitRequest {
    CommitRequest {
        scope: scope.clone(),
        source: "host".into(),
        content: content.into(),
        idempotency_key: None,
        facts,
    }
}

fn is_invalid<T>(result: memoryos_core::Result<T>) -> String {
    match result {
        Err(Error::InvalidInput(message)) => message,
        Err(other) => panic!("expected InvalidInput, got {other:?}"),
        Ok(_) => panic!("expected InvalidInput, got Ok"),
    }
}

/// Commits one fact and returns its id.
async fn commit_one(w: &World, scope: &Scope, text: &str, subject: &str) -> Uuid {
    let request = commit_request(
        scope,
        &format!("source of {text}"),
        vec![add_fact(text, subject)],
    );
    let response = w
        .host
        .commit(&ctx(), request, Trust::Agent)
        .await
        .expect("commit");
    response.facts[0].id
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_commit_adds_facts_in_one_embedding_batch() {
    let w = world().await;
    common::host_commit_adds_facts_in_one_embedding_batch(w.store.clone(), &w.host, &w.embedder)
        .await;
    assert_eq!(w.count("entities").await, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_trust_is_clamped_and_gated() {
    let w = world().await;
    let (user, agent) = (Scope::user("u1"), Scope::agent("a1"));
    for (scope, max_trust, content, trust, status) in [
        (
            &agent,
            Trust::System,
            "s1",
            Trust::Agent,
            FactStatus::Active,
        ),
        (&agent, Trust::User, "s2", Trust::Agent, FactStatus::Active),
        (&agent, Trust::Agent, "s3", Trust::Agent, FactStatus::Active),
        (
            &agent,
            Trust::External,
            "s4",
            Trust::External,
            FactStatus::Quarantined,
        ),
        (
            &user,
            Trust::System,
            "s5",
            Trust::Agent,
            FactStatus::Quarantined,
        ),
        (
            &user,
            Trust::External,
            "s6",
            Trust::External,
            FactStatus::Quarantined,
        ),
    ] {
        // A distinct statement each time, since a repeat would corroborate the earlier fact.
        let statement = format!("fact {content}");
        let request = commit_request(scope, content, vec![add_fact(&statement, "Ada")]);
        let response = w
            .host
            .commit(&ctx(), request, max_trust)
            .await
            .expect("commit");
        assert_eq!(response.facts[0].status, status, "{content}");
        let fact = w
            .store
            .get_fact(&ctx(), response.facts[0].id)
            .await
            .expect("fact");
        assert_eq!(fact.trust, trust, "{content}");
        assert_eq!(fact.status, status, "{content}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_supersede_closes_the_old_fact() {
    let w = world().await;
    common::host_supersede_closes_the_old_fact(w.store.clone(), &w.host).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_supersede_of_higher_trust_is_quarantined_with_pending() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let old = w
        .seed_fact(
            &scope,
            Trust::User,
            "Ada is vegan",
            "Ada",
            vector_for("ada"),
        )
        .await;

    let mut fact = add_fact("Ada eats fish", "Ada");
    fact.supersedes = Some(old.id);
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "host words", vec![fact]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    assert_eq!(response.facts[0].status, FactStatus::Quarantined);

    let new = w
        .store
        .get_fact(&ctx(), response.facts[0].id)
        .await
        .expect("new");
    assert_eq!(new.pending_supersede_of, Some(old.id));
    let old = w.store.get_fact(&ctx(), old.id).await.expect("old");
    assert_eq!(old.valid_to, None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_corroborate_links_without_changing_trust_or_creating_entities() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let id = commit_one(&w, &scope, "Ada plays go", "Ada").await;
    let entities = w.count("entities").await;

    let mut fact = add_fact("Ada still plays go", "Someone New");
    fact.corroborates = Some(id);
    let calls = w.embedder.batches().len();
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "again", vec![fact]),
            Trust::External,
        )
        .await
        .expect("commit");

    assert_eq!(response.facts.len(), 1);
    assert_eq!(response.facts[0].id, id);
    assert_eq!(response.facts[0].status, FactStatus::Active);
    let after = w.store.get_fact(&ctx(), id).await.expect("fact");
    assert_eq!(after.trust, Trust::Agent);
    assert_eq!(after.status, FactStatus::Active);
    assert_eq!(after.source_episode_ids.len(), 2);
    assert_eq!(w.count("facts").await, 1);
    assert_eq!(w.count("entities").await, entities);
    assert_eq!(
        w.embedder.batches().len(),
        calls,
        "a corroboration embeds nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_rejects_another_tenants_fact_and_leaves_nothing_behind() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let foreign = commit_one(&w, &scope, "Tenant secret", "Ada").await;
    let episodes = w.count("episodes").await;

    for (index, reference) in ["supersedes", "corroborates"].into_iter().enumerate() {
        let mut fact = add_fact("Attack", "Ada");
        match reference {
            "supersedes" => fact.supersedes = Some(foreign),
            _ => fact.corroborates = Some(foreign),
        }
        let request = commit_request(&scope, &format!("attack {index}"), vec![fact]);
        let message = is_invalid(
            w.host
                .commit(&tenant_ctx("other"), request, Trust::Agent)
                .await,
        );
        assert!(!message.contains("Tenant secret"), "{message}");
    }
    assert_eq!(
        w.count("episodes").await,
        episodes,
        "refused commits are discarded"
    );
    assert_eq!(
        w.store
            .get_fact(&ctx(), foreign)
            .await
            .expect("fact")
            .valid_to,
        None
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_rejects_a_fact_in_another_scope() {
    let w = world().await;
    let other_scope = Scope::agent("a2");
    let id = commit_one(&w, &other_scope, "Other scope fact", "Ada").await;

    let mut fact = add_fact("Replacement", "Ada");
    fact.supersedes = Some(id);
    let scope = Scope::agent("a1");
    let result = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "cross scope", vec![fact]),
            Trust::Agent,
        )
        .await;
    is_invalid(result);
    assert_eq!(
        w.store.get_fact(&ctx(), id).await.expect("fact").valid_to,
        None
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_rejects_facts_that_are_not_current_and_active() {
    let w = world().await;
    let scope = Scope::agent("a1");

    let unknown = Uuid::now_v7();
    let closed = commit_one(&w, &scope, "Closed fact", "Ada").await;
    let mut closer = add_fact("Closer", "Ada");
    closer.supersedes = Some(closed);
    w.host
        .commit(
            &ctx(),
            commit_request(&scope, "closing", vec![closer]),
            Trust::Agent,
        )
        .await
        .expect("close");

    let retracted = commit_one(&w, &scope, "Retracted fact", "Ada").await;
    w.store
        .invalidate_fact(&ctx(), retracted, "wrong")
        .await
        .expect("invalidate");

    let held = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "held", vec![add_fact("Held fact", "Ada")]),
            Trust::External,
        )
        .await
        .expect("held")
        .facts[0]
        .id;

    for (name, id) in [
        ("unknown", unknown),
        ("closed", closed),
        ("retracted", retracted),
        ("quarantined", held),
    ] {
        let mut fact = add_fact("New", "Ada");
        fact.supersedes = Some(id);
        let request = commit_request(&scope, &format!("try {name}"), vec![fact]);
        let message = is_invalid(w.host.commit(&ctx(), request, Trust::Agent).await);
        assert_eq!(
            message,
            format!("fact {id} is not current in this scope"),
            "{name}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_replay_returns_the_episode_without_applying_twice() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let old = commit_one(&w, &scope, "Old fact", "Ada").await;

    let mut fact = add_fact("New fact", "Ada");
    fact.supersedes = Some(old);
    let mut request = commit_request(&scope, "keyed", vec![fact]);
    request.idempotency_key = Some("key-1".into());

    let first = w
        .host
        .commit(&ctx(), request.clone(), Trust::Agent)
        .await
        .expect("first");
    assert_eq!(first.facts.len(), 1);
    let (facts, episodes) = (w.count("facts").await, w.count("episodes").await);

    // The old fact is closed by now, so re-checking references would wrongly refuse this.
    let replay = w
        .host
        .commit(&ctx(), request.clone(), Trust::Agent)
        .await
        .expect("replay");
    assert_eq!(replay.episode_id, first.episode_id);
    assert_eq!(replay.facts.len(), 1);
    assert_eq!(replay.facts[0].id, first.facts[0].id);
    assert_eq!(replay.facts[0].status, first.facts[0].status);
    assert_eq!(w.count("facts").await, facts);
    assert_eq!(w.count("episodes").await, episodes);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_replay_reports_quarantined_status() {
    let w = world().await;
    let scope = Scope::user("u1");
    let mut request = commit_request(&scope, "keyed", vec![add_fact("Held", "Ada")]);
    request.idempotency_key = Some("key-held".into());
    let first = w
        .host
        .commit(&ctx(), request.clone(), Trust::Agent)
        .await
        .expect("first");
    let replay = w
        .host
        .commit(&ctx(), request, Trust::Agent)
        .await
        .expect("replay");
    assert_eq!(replay.facts[0].id, first.facts[0].id);
    assert_eq!(replay.facts[0].status, FactStatus::Quarantined);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_replay_keeps_request_order() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let linked = commit_one(&w, &scope, "Linked", "Ada").await;

    let mut corroborating = add_fact("Restated", "Ada");
    corroborating.corroborates = Some(linked);
    let mut request = commit_request(
        &scope,
        "keyed",
        vec![
            add_fact("One", "Ada"),
            corroborating,
            add_fact("Two", "Ada"),
        ],
    );
    request.idempotency_key = Some("key-order".into());

    let first = w
        .host
        .commit(&ctx(), request.clone(), Trust::Agent)
        .await
        .expect("first");
    assert_eq!(first.facts.len(), 3);
    assert_eq!(first.facts[1].id, linked);

    let replay = w
        .host
        .commit(&ctx(), request, Trust::Agent)
        .await
        .expect("replay");
    let ids = |r: &memoryos_ingest::CommitResponse| -> Vec<Uuid> {
        r.facts.iter().map(|f| f.id).collect()
    };
    assert_eq!(ids(&replay), ids(&first));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_retry_after_a_failed_attempt_applies_once() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let mut request = commit_request(&scope, "keyed", vec![add_fact("Retried fact", "Ada")]);
    request.idempotency_key = Some("key-retry".into());

    w.embedder.set_failure("Retried fact");
    let failed = w.host.commit(&ctx(), request.clone(), Trust::Agent).await;
    assert!(matches!(failed, Err(Error::Provider(_))), "{failed:?}");
    assert_eq!(
        w.count("episodes").await,
        0,
        "the failed attempt is discarded"
    );

    w.embedder.clear("Retried fact");
    let response = w
        .host
        .commit(&ctx(), request, Trust::Agent)
        .await
        .expect("retry");
    assert_eq!(response.facts.len(), 1);
    assert_eq!(w.count("facts").await, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_refuses_a_key_reused_for_a_different_request() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let mut request = commit_request(&scope, "keyed", vec![add_fact("Fact", "Ada")]);
    request.idempotency_key = Some("key-2".into());
    w.host
        .commit(&ctx(), request.clone(), Trust::Agent)
        .await
        .expect("first");

    let mut other_scope = request.clone();
    other_scope.scope = Scope::agent("a2");
    let message = is_invalid(w.host.commit(&ctx(), other_scope, Trust::Agent).await);
    assert!(!message.contains("a1"), "{message}");

    let mut other_content = request.clone();
    other_content.content = "different".into();
    is_invalid(w.host.commit(&ctx(), other_content, Trust::Agent).await);

    is_invalid(w.host.commit(&ctx(), request, Trust::External).await);
    assert_eq!(w.count("episodes").await, 1);
    assert_eq!(w.count("facts").await, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_key_of_a_higher_trust_episode_cannot_borrow_its_trust() {
    let w = world().await;
    let scope = Scope::user("u1");
    let content = "user words";
    w.store
        .insert_episode(
            &ctx(),
            NewEpisode {
                scope: scope.clone(),
                source: "host".into(),
                trust: Trust::User,
                content: content.into(),
                occurred_at: None,
                metadata: json!({}),
                idempotency_key: Some("key-3".into()),
            },
        )
        .await
        .expect("insert");
    let mut request = commit_request(&scope, content, vec![add_fact("Planted", "Ada")]);
    request.idempotency_key = Some("key-3".into());

    is_invalid(w.host.commit(&ctx(), request, Trust::Agent).await);
    assert_eq!(w.count("facts").await, 0);
}

/// One subject with more current facts than the subject query returns. The oldest is still current.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_current_target_is_accepted_on_a_subject_with_many_facts() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let subject = w
        .store
        .upsert_entity(
            &ctx(),
            NewEntity {
                scope: scope.clone(),
                kind: "person".into(),
                name: "Ada".into(),
                aliases: Vec::new(),
                embedding: vector_for("ada"),
            },
        )
        .await
        .expect("entity");
    let episode = w
        .store
        .insert_episode(
            &ctx(),
            NewEpisode {
                scope: scope.clone(),
                source: "seed".into(),
                trust: Trust::Agent,
                content: "seed".into(),
                occurred_at: None,
                metadata: json!({}),
                idempotency_key: None,
            },
        )
        .await
        .expect("episode");
    let ops = (0..201)
        .map(|i| {
            memoryos_core::FactOp::Add(memoryos_core::NewFact {
                scope: scope.clone(),
                subject_id: subject.id,
                predicate: "fact_about".into(),
                object_id: None,
                statement: format!("seed {i}"),
                embedding: vector_for("seed"),
                trust: Trust::Agent,
                status: FactStatus::Active,
                valid_from: date(2020),
                valid_to: None,
                expires_at: None,
            })
        })
        .collect();
    let ids = w
        .store
        .apply_commit(&ctx(), episode.id, ops)
        .await
        .expect("seed");

    let mut fact = add_fact("Moved on", "Ada");
    fact.supersedes = Some(ids[0]);
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "moved", vec![fact]),
            Trust::Agent,
        )
        .await
        .expect("the oldest fact is a valid target");
    let closed = w.store.get_fact(&ctx(), ids[0]).await.expect("old");
    assert_eq!(closed.superseded_by, Some(response.facts[0].id));
}

// ---- Replays that fail ------------------------------------------------------------------------

impl World {
    async fn wait_for_episode(&self) {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while self.count("episodes").await != 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the first request did not insert its episode"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

fn keyed_request(scope: &Scope, key: &str, fact: CommitFact) -> CommitRequest {
    let mut request = commit_request(scope, "same source", vec![fact]);
    request.idempotency_key = Some(key.into());
    request
}

/// The loser reads the unprocessed episode, the winner is acknowledged, and only then does the
/// loser's own work fail.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replay_failing_late_keeps_the_acknowledged_facts_of_the_winner() {
    let w = world().await;
    let scope = Scope::agent("a1");
    w.embedder
        .set_slow("Winner", Duration::from_secs(1), vector_for("Winner"));
    w.embedder.set_slow_failure("Loser", Duration::from_secs(3));

    let context = ctx();
    let winner = w.host.commit(
        &context,
        keyed_request(&scope, "race", add_fact("Winner", "Ada")),
        Trust::Agent,
    );
    let loser = async {
        w.wait_for_episode().await;
        w.host
            .commit(
                &ctx(),
                keyed_request(&scope, "race", add_fact("Loser", "Ada")),
                Trust::Agent,
            )
            .await
    };
    let (winner, loser) = tokio::join!(winner, loser);

    let winner = winner.expect("winner");
    assert!(loser.is_err(), "the loser's embedding call was refused");
    let kept = w
        .store
        .get_fact(&ctx(), winner.facts[0].id)
        .await
        .expect("the acknowledged fact is still there");
    assert_eq!(kept.statement, "Winner");
    assert_eq!(w.count("episodes").await, 1);
}

/// The loser fails while the winner is still working, and the winner must still be able to finish.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replay_failing_early_does_not_remove_the_episode_of_a_running_commit() {
    let w = world().await;
    let scope = Scope::agent("a1");
    w.embedder
        .set_slow("Winner", Duration::from_millis(1500), vector_for("Winner"));

    let context = ctx();
    let winner = w.host.commit(
        &context,
        keyed_request(&scope, "race", add_fact("Winner", "Ada")),
        Trust::Agent,
    );
    let loser = async {
        w.wait_for_episode().await;
        let mut bad_target = add_fact("Loser", "Ada");
        bad_target.supersedes = Some(Uuid::now_v7());
        // Not asserted: on a busy machine the winner can finish first, and this is then a plain
        // replay. What matters is the winner's result.
        let _ = w
            .host
            .commit(
                &ctx(),
                keyed_request(&scope, "race", bad_target),
                Trust::Agent,
            )
            .await;
    };
    let (winner, ()) = tokio::join!(winner, loser);

    let winner = winner.expect("the winner still commits");
    assert_eq!(winner.facts.len(), 1);
    assert_eq!(w.count("facts").await, 1);
}

// ---- Entity resolution -----------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_entity_name_resolves_to_one_entity() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let first = commit_one(&w, &scope, "First", "Ada Lovelace").await;

    // Different case and spacing, in two forms within one request: one embedding input, one entity.
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(
                &scope,
                "both",
                vec![
                    add_fact("Second", "  ADA   lovelace "),
                    add_fact("Third", "Ada Lovelace"),
                ],
            ),
            Trust::Agent,
        )
        .await
        .expect("commit");

    assert_eq!(w.count("entities").await, 1);
    let subject = w
        .store
        .get_fact(&ctx(), first)
        .await
        .expect("first")
        .subject_id;
    for fact in &response.facts {
        let got = w.store.get_fact(&ctx(), fact.id).await.expect("fact");
        assert_eq!(got.subject_id, subject);
    }
    let entity = w.store.get_entity(&ctx(), subject).await.expect("entity");
    assert_eq!(entity.name, "Ada Lovelace");
    assert!(entity.aliases.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn similar_name_merges_at_0_90_with_the_same_kind_only() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let commit = |statement: &'static str, name: &'static str, kind: &'static str| {
        let scope = scope.clone();
        let w = &w;
        async move {
            let mut fact = add_fact(statement, name);
            fact.subject.kind = kind.into();
            let request = commit_request(&scope, &format!("source {statement}"), vec![fact]);
            let response = w
                .host
                .commit(&ctx(), request, Trust::Agent)
                .await
                .expect("commit");
            w.store
                .get_fact(&ctx(), response.facts[0].id)
                .await
                .expect("fact")
                .subject_id
        }
    };

    w.embedder.set_vector("robert", sparse(&[1.0]));
    let robert = commit("s1", "Robert", "person").await;

    // Cosine with [1, 0, ...] is 9 / sqrt(81 + 9 + 9 + 1) = 9 / 10: exactly the threshold.
    w.embedder.set_vector("bob", sparse(&[9.0, 3.0, 3.0, 1.0]));
    assert_eq!(commit("s2", "Bob", "person").await, robert);
    let entity = w.store.get_entity(&ctx(), robert).await.expect("entity");
    assert_eq!(entity.name, "Robert");
    assert_eq!(entity.aliases, vec!["Bob".to_owned()]);

    // Cosine is 89 / sqrt(7921 + 2025 + 49 + 4 + 1) = 89 / 100: below the threshold.
    w.embedder
        .set_vector("rob", sparse(&[89.0, 45.0, 7.0, 2.0, 1.0]));
    assert_ne!(commit("s3", "Rob", "person").await, robert);

    // Same vector, other kind.
    w.embedder.set_vector("robert", sparse(&[1.0]));
    assert_ne!(commit("s4", "Robert", "company").await, robert);

    // The alias now resolves without adding itself again, whatever its vector.
    w.embedder.set_vector("bob", sparse(&[9.0, 3.0, 3.0, 1.0]));
    assert_eq!(commit("s5", "BOB", "person").await, robert);
    let entity = w.store.get_entity(&ctx(), robert).await.expect("entity");
    assert_eq!(entity.aliases, vec!["Bob".to_owned()]);

    assert_eq!(w.count("entities").await, 3);
}

impl World {
    async fn seed_entity(&self, scope: &Scope, name: &str, embedding: Vec<f32>) -> Uuid {
        self.store
            .upsert_entity(
                &ctx(),
                NewEntity {
                    scope: scope.clone(),
                    kind: "person".into(),
                    name: name.into(),
                    aliases: Vec::new(),
                    embedding,
                },
            )
            .await
            .expect("seed entity")
            .id
    }

    /// Fills `similar_entities` with closer matches, so a resolution that relies on ranking misses.
    async fn seed_closer_entities(&self, scope: &Scope) {
        for i in 0..25 {
            let vector = sparse(&[1.0, 0.001 * i as f32]);
            self.seed_entity(scope, &format!("filler {i}"), vector)
                .await;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_name_resolves_even_when_ranked_out_of_the_similar_list() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let bob = w.seed_entity(&scope, "Bob", sparse(&[0.0, 0.0, 1.0])).await;
    w.seed_closer_entities(&scope).await;
    let before = w.count("entities").await;

    w.embedder.set_vector("bob", sparse(&[1.0]));
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "exact", vec![add_fact("s1", "BOB")]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let fact = w
        .store
        .get_fact(&ctx(), response.facts[0].id)
        .await
        .expect("fact");
    assert_eq!(fact.subject_id, bob);
    assert_eq!(w.count("entities").await, before);
    let aliased: i64 = sqlx::query_scalar("SELECT count(*) FROM entities WHERE aliases <> '{}'")
        .fetch_one(&w.admin)
        .await
        .expect("count aliased");
    assert_eq!(aliased, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn known_alias_resolves_even_when_ranked_out_of_the_similar_list() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let robert = w
        .seed_entity(&scope, "Robert", sparse(&[0.0, 0.0, 1.0]))
        .await;
    w.store
        .add_entity_alias(&ctx(), robert, "Bobby  Tables", None)
        .await
        .expect("alias");
    w.seed_closer_entities(&scope).await;
    let before = w.count("entities").await;

    w.embedder.set_vector("bobby tables", sparse(&[1.0]));
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "alias", vec![add_fact("s1", "bobby tables")]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let fact = w
        .store
        .get_fact(&ctx(), response.facts[0].id)
        .await
        .expect("fact");
    assert_eq!(fact.subject_id, robert);
    assert_eq!(w.count("entities").await, before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quarantined_host_commit_adds_no_alias_to_an_existing_entity() {
    let w = world().await;
    let scope = Scope::shared("team");
    let robert = w.seed_entity(&scope, "Robert", sparse(&[1.0])).await;

    w.embedder.set_vector("bob", sparse(&[1.0]));
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "low trust", vec![add_fact("planted", "Bob")]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    assert_eq!(response.facts[0].status, FactStatus::Quarantined);

    let fact = w
        .store
        .get_fact(&ctx(), response.facts[0].id)
        .await
        .expect("fact");
    assert_eq!(fact.subject_id, robert);
    let entity = w.store.get_entity(&ctx(), robert).await.expect("entity");
    assert!(entity.aliases.is_empty(), "{:?}", entity.aliases);
}

/// The commit's new fact is held, and its only active link on Robert is a corroboration.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_corroboration_link_does_not_earn_an_alias() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let linked = w.seed_on_robert(&scope, Trust::User, "linked").await;
    let outranking = w.seed_on_robert(&scope, Trust::User, "outranking").await;

    w.embedder.set_vector("bob", sparse(&[1.0]));
    let mut corroborating = add_fact("restated", "Robert");
    corroborating.corroborates = Some(linked.id);
    let mut planted = add_fact("planted", "Bob");
    planted.supersedes = Some(outranking.id);
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "both", vec![corroborating, planted]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    assert_eq!(response.facts[1].status, FactStatus::Quarantined);

    let entity = w
        .store
        .get_entity(&ctx(), linked.subject_id)
        .await
        .expect("entity");
    assert!(entity.aliases.is_empty(), "{:?}", entity.aliases);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn purging_the_source_of_an_alias_removes_the_alias() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let seed = w.seed_on_robert(&scope, Trust::Agent, "seed").await;

    w.embedder.set_vector("bob", sparse(&[1.0]));
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "merge", vec![add_fact("merged", "Bob")]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let aliases = |entity: memoryos_core::Entity| entity.aliases;
    let got = w.store.get_entity(&ctx(), seed.subject_id).await;
    assert_eq!(aliases(got.expect("entity")), ["Bob"]);

    w.store
        .purge_episode(&ctx(), response.episode_id)
        .await
        .expect("purge");
    let got = w.store.get_entity(&ctx(), seed.subject_id).await;
    assert!(aliases(got.expect("entity")).is_empty());
    let found = w
        .store
        .find_entity(&ctx(), &scope, "person", "Bob")
        .await
        .expect("lookup");
    assert!(found.is_none());
}

/// Not barred by its source, but outranked: every fact it would add is still quarantined.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn outranked_commit_adds_no_alias_to_an_existing_entity() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let seed = w.seed_on_robert(&scope, Trust::User, "Seed").await;

    w.embedder.set_vector("bob", sparse(&[1.0]));
    let mut fact = add_fact("planted", "Bob");
    fact.supersedes = Some(seed.id);
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "outranked", vec![fact]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    assert_eq!(response.facts[0].status, FactStatus::Quarantined);

    let entity = w
        .store
        .get_entity(&ctx(), seed.subject_id)
        .await
        .expect("entity");
    assert!(entity.aliases.is_empty(), "{:?}", entity.aliases);
}

/// The corroborated fact is active, but that activity is not this source's to lend.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_corroboration_adds_no_alias_to_an_existing_entity() {
    let w = world().await;
    let scope = Scope::user("u1");
    let seed = w.seed_on_robert(&scope, Trust::User, "Seed").await;

    w.embedder.set_vector("bob", sparse(&[1.0]));
    let mut fact = add_fact("Restated", "Bob");
    fact.corroborates = Some(seed.id);
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "web page", vec![fact]),
            Trust::External,
        )
        .await
        .expect("commit");

    let after = w.store.get_fact(&ctx(), seed.id).await.expect("fact");
    assert!(after.source_episode_ids.contains(&response.episode_id));
    let entity = w
        .store
        .get_entity(&ctx(), seed.subject_id)
        .await
        .expect("entity");
    assert!(entity.aliases.is_empty(), "{:?}", entity.aliases);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn low_trust_new_entity_is_created_and_purged_with_its_fact() {
    let w = world().await;
    let scope = Scope::shared("team");
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "low trust", vec![add_fact("planted", "Mallory")]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    assert_eq!(w.count("entities").await, 1);

    let report = w
        .store
        .purge_episode(&ctx(), response.episode_id)
        .await
        .expect("purge");
    assert_eq!(report.entities_deleted, 1);
    assert_eq!(w.count("entities").await, 0);
}

// ---- Matching on commit ----------------------------------------------------------------------

/// Commits one fact under `ctx` and returns its response entry.
async fn commit_as(
    w: &World,
    ctx: &RequestContext,
    scope: &Scope,
    text: &str,
    subject: &str,
) -> CommittedFact {
    let request = commit_request(
        scope,
        &format!("source of {text}"),
        vec![add_fact(text, subject)],
    );
    let mut response = w
        .host
        .commit(ctx, request, Trust::Agent)
        .await
        .expect("commit");
    assert_eq!(response.facts.len(), 1);
    response.facts.remove(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_exact_repeat_corroborates_the_existing_fact() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let first = commit_as(&w, &ctx(), &scope, "Ada plays go", "Ada").await;
    assert!(!first.corroborated);
    let entities = w.count("entities").await;

    // A new subject name would create an entity if the repeat were treated as a new fact.
    let repeat = commit_as(&w, &ctx(), &scope, "Ada plays go", "Someone New").await;
    assert_eq!(repeat.id, first.id);
    assert!(repeat.corroborated);
    assert_eq!(repeat.status, FactStatus::Active);
    assert!(repeat.related.is_empty());
    assert_eq!(w.count("facts").await, 1);
    assert_eq!(w.count("entities").await, entities);
    let fact = w.store.get_fact(&ctx(), first.id).await.expect("fact");
    assert_eq!(fact.source_episode_ids.len(), 2);
    assert_eq!(fact.trust, Trust::Agent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_case_and_spacing_variant_also_corroborates() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let first = commit_as(&w, &ctx(), &scope, "Ada plays go", "Ada").await;
    let variant = commit_as(&w, &ctx(), &scope, "  ada PLAYS \t go. ", "Ada").await;
    assert_eq!(variant.id, first.id);
    assert!(variant.corroborated);
    assert_eq!(w.count("facts").await, 1);
    let fact = w.store.get_fact(&ctx(), first.id).await.expect("fact");
    assert_eq!(fact.statement, "Ada plays go");
    assert_eq!(fact.source_episode_ids.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_different_statement_is_created_and_lists_the_similar_fact() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let old = commit_as(&w, &ctx(), &scope, "Ada lives in Paris", "Ada").await;
    let new = commit_as(&w, &ctx(), &scope, "Ada lives in Berlin", "Ada").await;
    assert_ne!(new.id, old.id);
    assert!(!new.corroborated);
    assert_eq!(w.count("facts").await, 2);
    let related: Vec<Uuid> = new.related.iter().map(|r| r.id).collect();
    assert_eq!(related, [old.id]);
    assert_eq!(new.related[0].statement, "Ada lives in Paris");
    assert!(new.related[0].score > 0.0);
    // Listed, not acted on: the old fact stays current.
    let old = w.store.get_fact(&ctx(), old.id).await.expect("old");
    assert_eq!((old.valid_to, old.superseded_by), (None, None));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn matching_never_crosses_scopes_or_tenants() {
    let w = world().await;
    common::matching_never_crosses_scopes_or_tenants(w.store.clone(), &w.host).await;
    assert_eq!(w.count("facts").await, 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_quarantined_fact_is_never_matched_or_listed() {
    let w = world().await;
    // Agent trust in a user scope is held for review.
    let scope = Scope::user("u1");
    let held = commit_as(&w, &ctx(), &scope, "Ada plays go", "Ada").await;
    assert_eq!(held.status, FactStatus::Quarantined);

    let again = commit_as(&w, &ctx(), &scope, "Ada plays go", "Ada").await;
    assert_ne!(again.id, held.id);
    assert!(!again.corroborated);
    assert!(again.related.is_empty(), "{:?}", again.related);
    assert_eq!(w.count("facts").await, 2);
    let held = w.store.get_fact(&ctx(), held.id).await.expect("held");
    assert_eq!(held.source_episode_ids.len(), 1);
}

/// A replay has no matching results to go by, so this checks the real recorded times.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replay_reports_a_matched_repeat_in_request_order() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let existing = commit_as(&w, &ctx(), &scope, "Ada plays go", "Ada").await;

    let mut request = commit_request(
        &scope,
        "keyed",
        vec![
            add_fact("Ada reads novels", "Ada"),
            add_fact("ada plays go.", "Ada"),
            add_fact("Ada swims", "Ada"),
        ],
    );
    request.idempotency_key = Some("key-matched".into());
    let first = w
        .host
        .commit(&ctx(), request.clone(), Trust::Agent)
        .await
        .expect("first");
    let replay = w
        .host
        .commit(&ctx(), request, Trust::Agent)
        .await
        .expect("replay");

    let shape = |r: &CommitResponse| -> Vec<(Uuid, bool)> {
        r.facts.iter().map(|f| (f.id, f.corroborated)).collect()
    };
    let flags: Vec<bool> = shape(&first).into_iter().map(|(_, c)| c).collect();
    assert_eq!(flags, [false, true, false]);
    assert_eq!(first.facts[1].id, existing.id);
    assert_eq!(shape(&replay), shape(&first));
    assert_eq!(w.count("facts").await, 3);
}

// ---- Supersede onto an existing fact ---------------------------------------------------------

/// A one-fact request that supersedes `old` with `text`.
fn supersede_request(scope: &Scope, content: &str, text: &str, old: Uuid) -> CommitRequest {
    let mut fact = add_fact(text, "Ada");
    fact.supersedes = Some(old);
    commit_request(scope, content, vec![fact])
}

/// The flow the tool description asks for: a first call creates a fact and lists an older value as
/// `related`, then a second call sends the same statement superseding it.
async fn second_call_supersedes(w: &World, statement: &str) -> (Uuid, Uuid, CommittedFact) {
    let scope = Scope::agent("a1");
    let old = commit_as(w, &ctx(), &scope, "Ada lives in Jakarta", "Ada").await;
    let first = commit_as(w, &ctx(), &scope, "Ada lives in Bandung", "Ada").await;
    assert!(!first.corroborated);
    assert!(first.related.iter().any(|r| r.id == old.id));

    let mut second = w
        .host
        .commit(
            &ctx(),
            supersede_request(&scope, "moved", statement, old.id),
            Trust::Agent,
        )
        .await
        .expect("second call");
    assert_eq!(second.facts.len(), 1);
    (old.id, first.id, second.facts.remove(0))
}

/// Exactly one current fact remains, and it is the one the first call created.
async fn assert_closed_into(w: &World, old: Uuid, kept: Uuid) {
    let scope = Scope::agent("a1");
    assert_eq!(w.count("facts").await, 2);
    let kept = w.store.get_fact(&ctx(), kept).await.expect("kept");
    let old = w.store.get_fact(&ctx(), old).await.expect("old");
    assert_eq!(old.superseded_by, Some(kept.id));
    assert_eq!(old.valid_to, Some(kept.valid_from));
    assert_eq!((kept.superseded_by, kept.valid_to), (None, None));
    assert_eq!(kept.status, FactStatus::Active);
    assert_eq!(kept.statement, "Ada lives in Bandung");
    assert_eq!(kept.source_episode_ids.len(), 2);
    let history: Vec<Uuid> = w
        .store
        .fact_history(&ctx(), kept.id)
        .await
        .expect("history")
        .into_iter()
        .map(|fact| fact.id)
        .collect();
    assert_eq!(history, [old.id, kept.id]);
    assert_eq!(
        w.recall_ids(&scope, "Ada lives in Jakarta", None).await,
        [kept.id]
    );
    // History is intact: before the move, the old value was the one in effect.
    let before = kept.valid_from - chrono::Duration::milliseconds(1);
    assert!(w
        .recall_ids(&scope, "Ada lives in Jakarta", Some(before))
        .await
        .contains(&old.id));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_call_superseding_a_related_fact_creates_no_duplicate() {
    let w = world().await;
    let (old, created, second) = second_call_supersedes(&w, "Ada lives in Bandung").await;
    assert_eq!(second.id, created);
    assert!(second.corroborated);
    assert_eq!(second.superseded, Some(old));
    assert_eq!(second.status, FactStatus::Active);
    assert!(second.related.is_empty());
    assert_closed_into(&w, old, created).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_case_and_period_variant_of_the_second_call_also_reuses_the_fact() {
    let w = world().await;
    let (old, created, second) = second_call_supersedes(&w, "ada lives in  BANDUNG.").await;
    assert_eq!(second.id, created);
    assert!(second.corroborated);
    assert_eq!(second.superseded, Some(old));
    assert_closed_into(&w, old, created).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_supersede_with_no_current_repeat_creates_its_fact() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let old = commit_one(&w, &scope, "Ada lives in Rome", "Ada").await;
    let other = commit_one(&w, &scope, "Ada lives near Rome", "Ada").await;

    let response = w
        .host
        .commit(
            &ctx(),
            supersede_request(&scope, "moved", "Ada lives in Oslo", old),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let new = &response.facts[0];
    assert!(!new.corroborated);
    assert_eq!(new.superseded, None);
    assert!(new.related.is_empty());
    assert_ne!(new.id, other);
    assert_eq!(w.count("facts").await, 3);
    let old = w.store.get_fact(&ctx(), old).await.expect("old");
    assert_eq!(old.superseded_by, Some(new.id));
    let other = w.store.get_fact(&ctx(), other).await.expect("other");
    assert_eq!(
        (other.superseded_by, other.source_episode_ids.len()),
        (None, 1)
    );
}

/// The new value would be held for review, so it is created as before: only a quarantined new fact
/// can carry the close until an admin approves it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_supersede_of_higher_trust_still_creates_a_held_fact() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let old = w
        .seed_fact(
            &scope,
            Trust::User,
            "Ada is vegan",
            "Ada",
            vector_for("ada"),
        )
        .await;
    let existing = commit_one(&w, &scope, "Ada eats fish", "Ada").await;

    let response = w
        .host
        .commit(
            &ctx(),
            supersede_request(&scope, "host words", "Ada eats fish", old.id),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let held = &response.facts[0];
    assert_ne!(held.id, existing);
    assert!(!held.corroborated);
    assert_eq!(held.superseded, None);
    assert_eq!(held.status, FactStatus::Quarantined);
    let held = w.store.get_fact(&ctx(), held.id).await.expect("held");
    assert_eq!(held.pending_supersede_of, Some(old.id));
    let old = w.store.get_fact(&ctx(), old.id).await.expect("old");
    assert_eq!((old.valid_to, old.superseded_by), (None, None));
    let existing = w.store.get_fact(&ctx(), existing).await.expect("existing");
    assert_eq!(existing.source_episode_ids.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replayed_second_call_changes_nothing() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let old = commit_one(&w, &scope, "Ada lives in Jakarta", "Ada").await;
    let created = commit_one(&w, &scope, "Ada lives in Bandung", "Ada").await;
    let mut request = supersede_request(&scope, "moved", "Ada lives in Bandung", old);
    request.idempotency_key = Some("key-move".into());

    let first = w
        .host
        .commit(&ctx(), request.clone(), Trust::Agent)
        .await
        .expect("first");
    let counts = (
        w.count("facts").await,
        w.count("episodes").await,
        w.count("fact_sources").await,
    );
    let replay = w
        .host
        .commit(&ctx(), request, Trust::Agent)
        .await
        .expect("replay");
    assert_eq!(replay.episode_id, first.episode_id);
    let shape = |r: &CommitResponse| -> Vec<(Uuid, bool, Option<Uuid>)> {
        r.facts
            .iter()
            .map(|f| (f.id, f.corroborated, f.superseded))
            .collect()
    };
    assert_eq!(shape(&first), [(created, true, Some(old))]);
    assert_eq!(shape(&replay), shape(&first));
    assert_eq!(
        (
            w.count("facts").await,
            w.count("episodes").await,
            w.count("fact_sources").await,
        ),
        counts
    );
    assert_closed_into(&w, old, created).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_supersede_never_reuses_a_fact_of_another_scope_or_tenant() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let globex = tenant_ctx("globex");
    let other_scope = commit_as(
        &w,
        &ctx(),
        &Scope::agent("a2"),
        "Ada lives in Bandung",
        "Ada",
    )
    .await;
    let other_tenant = commit_as(&w, &globex, &scope, "Ada lives in Bandung", "Ada").await;
    let old = commit_one(&w, &scope, "Ada lives in Jakarta", "Ada").await;

    let response = w
        .host
        .commit(
            &ctx(),
            supersede_request(&scope, "moved", "Ada lives in Bandung", old),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let new = &response.facts[0];
    assert!(!new.corroborated);
    assert_ne!(new.id, other_scope.id);
    assert_ne!(new.id, other_tenant.id);
    assert_eq!(w.count("facts").await, 4);
    let old = w.store.get_fact(&ctx(), old).await.expect("old");
    assert_eq!(old.superseded_by, Some(new.id));
    for (ctx, foreign) in [(ctx(), other_scope.id), (globex, other_tenant.id)] {
        let fact = w.store.get_fact(&ctx, foreign).await.expect("fact");
        assert_eq!(fact.source_episode_ids.len(), 1);
        assert_eq!(fact.superseded_by, None);
    }
}

/// The two-call flow, returning the old fact, the reused fact, its original episode and call 2's.
async fn two_calls(w: &World) -> (Uuid, Uuid, Uuid, Uuid) {
    let scope = Scope::agent("a1");
    let old = commit_one(w, &scope, "Ada lives in Jakarta", "Ada").await;
    let kept = commit_one(w, &scope, "Ada lives in Bandung", "Ada").await;
    let original = w
        .store
        .get_fact(&ctx(), kept)
        .await
        .expect("kept")
        .source_episode_ids[0];
    let second = w
        .host
        .commit(
            &ctx(),
            supersede_request(&scope, "moved", "Ada lives in Bandung", old),
            Trust::Agent,
        )
        .await
        .expect("second call");
    assert_eq!(second.facts[0].id, kept);
    assert_closed_into(w, old, kept).await;
    (old, kept, original, second.episode_id)
}

/// The fact is current and alone in its history.
async fn assert_current_alone(w: &World, id: Uuid) -> Fact {
    let fact = w.store.get_fact(&ctx(), id).await.expect("fact");
    assert_eq!((fact.superseded_by, fact.valid_to), (None, None));
    assert_eq!(fact.status, FactStatus::Active);
    let history: Vec<Uuid> = w
        .store
        .fact_history(&ctx(), id)
        .await
        .expect("history")
        .into_iter()
        .map(|fact| fact.id)
        .collect();
    assert_eq!(history, [id]);
    fact
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn purging_the_second_call_reopens_the_old_fact() {
    let w = world().await;
    let (old, kept, original, second) = two_calls(&w).await;

    w.store.purge_episode(&ctx(), second).await.expect("purge");
    assert_current_alone(&w, old).await;
    let kept_fact = assert_current_alone(&w, kept).await;
    assert_eq!(kept_fact.source_episode_ids, [original]);
    assert_eq!(w.count("facts").await, 2);

    // Purging the reused fact's original source afterwards removes it and leaves the old one.
    w.store
        .purge_episode(&ctx(), original)
        .await
        .expect("purge");
    assert_current_alone(&w, old).await;
    assert_eq!(w.count("facts").await, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn purging_the_reused_facts_source_first_keeps_the_close_until_call_two_goes() {
    let w = world().await;
    let (old, kept, original, second) = two_calls(&w).await;

    // Call 2 still supports the reused fact, so it and the close stay.
    w.store
        .purge_episode(&ctx(), original)
        .await
        .expect("purge");
    let kept_fact = w.store.get_fact(&ctx(), kept).await.expect("kept");
    assert_eq!(kept_fact.source_episode_ids, [second]);
    let old_fact = w.store.get_fact(&ctx(), old).await.expect("old");
    assert_eq!(old_fact.superseded_by, Some(kept));

    w.store.purge_episode(&ctx(), second).await.expect("purge");
    assert_current_alone(&w, old).await;
    assert_eq!(w.count("facts").await, 1);
}

// ---- Expiry ------------------------------------------------------------------------------------

/// A fact on Ada that started ten days ago and ends at `expires_at`.
fn expiring_fact(text: &str, expires_at: DateTime<Utc>) -> CommitFact {
    let mut fact = add_fact(text, "Ada");
    fact.valid_from = Some(Utc::now() - chrono::Duration::days(10));
    fact.expires_at = Some(expires_at);
    fact
}

async fn recalled(w: &World, scope: &Scope, query: &str) -> Vec<Uuid> {
    let retriever = memoryos_retrieve::Retriever::new(w.store.clone(), w.embedder.clone());
    let request = memoryos_retrieve::RecallRequest {
        scopes: vec![scope.clone()],
        query: query.into(),
        limit: 20,
        as_of: None,
        known_at: None,
    };
    retriever
        .recall(&ctx(), request)
        .await
        .expect("recall")
        .facts
        .into_iter()
        .map(|scored| scored.fact.id)
        .collect()
}

async fn text_hits(w: &World, scope: &Scope, query: &str) -> Vec<Uuid> {
    w.store
        .search_facts_text(
            &ctx(),
            std::slice::from_ref(scope),
            query,
            &FactFilter::default(),
            20,
        )
        .await
        .expect("text search")
        .into_iter()
        .map(|scored| scored.fact.id)
        .collect()
}

async fn listed(w: &World, scope: &Scope, state: memoryos_core::FactListState) -> Vec<Uuid> {
    w.store
        .list_facts(&ctx(), scope, state, 50, None)
        .await
        .expect("list")
        .into_iter()
        .map(|view| view.fact.id)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_expired_fact_leaves_recall_but_stays_in_history() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(
                &scope,
                "Ada was on leave, and is reviewing until next week",
                vec![
                    expiring_fact("Ada is on leave", Utc::now() - chrono::Duration::days(1)),
                    expiring_fact(
                        "Ada is reviewing the parser",
                        Utc::now() + chrono::Duration::days(7),
                    ),
                ],
            ),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let (expired, future) = (response.facts[0].id, response.facts[1].id);
    let stored = w.store.get_fact(&ctx(), expired).await.expect("expired");
    assert!(stored.expires_at.is_some());
    assert_eq!((stored.valid_to, stored.superseded_by), (None, None));

    // Neither vector nor text search, nor recall, returns the expired fact. The other one is current.
    for ids in [
        w.recall_ids(&scope, "Ada is on leave", None).await,
        text_hits(&w, &scope, "Ada").await,
        recalled(&w, &scope, "Ada is on leave").await,
    ] {
        assert!(!ids.contains(&expired), "{ids:?}");
        assert!(ids.contains(&future), "{ids:?}");
    }
    // Asked about a time before it expired, it is there.
    let then = Utc::now() - chrono::Duration::days(5);
    assert!(w
        .recall_ids(&scope, "Ada is on leave", Some(then))
        .await
        .contains(&expired));

    // History and the past listing keep it.
    let history = w
        .store
        .fact_history(&ctx(), expired)
        .await
        .expect("history");
    assert_eq!(
        history.iter().map(|fact| fact.id).collect::<Vec<_>>(),
        [expired]
    );
    use memoryos_core::FactListState::{Current, Past};
    assert_eq!(listed(&w, &scope, Past).await, [expired]);
    assert_eq!(listed(&w, &scope, Current).await, [future]);

    // Not current: no repeat matches it, and no supersede or corroborate may name it.
    let again = commit_as(&w, &ctx(), &scope, "Ada is on leave", "Ada").await;
    assert!(!again.corroborated);
    assert_ne!(again.id, expired);
    assert!(again.related.iter().all(|related| related.id != expired));
    let mut corroborating = add_fact("Ada is away", "Ada");
    corroborating.corroborates = Some(expired);
    for named in [
        supersede_request(&scope, "back", "Ada is back", expired),
        commit_request(&scope, "away", vec![corroborating]),
    ] {
        let message = is_invalid(w.host.commit(&ctx(), named, Trust::Agent).await);
        assert!(message.contains("not current"), "{message}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fact_drops_out_of_recall_when_its_expiry_passes() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let mut fact = add_fact("Ada is in a meeting", "Ada");
    fact.expires_at = Some(Utc::now() + chrono::Duration::seconds(2));
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "meeting", vec![fact]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    let id = response.facts[0].id;
    assert_eq!(response.facts[0].status, FactStatus::Active);
    assert!(recalled(&w, &scope, "Ada is in a meeting")
        .await
        .contains(&id));

    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(!recalled(&w, &scope, "Ada is in a meeting")
        .await
        .contains(&id));
    let fact = w.store.get_fact(&ctx(), id).await.expect("fact");
    assert_eq!((fact.valid_to, fact.superseded_by), (None, None));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_expiry_not_after_the_start_is_refused_and_stores_nothing() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let mut fact = add_fact("Ada was away", "Ada");
    fact.valid_from = Some(date(2024));
    fact.expires_at = Some(date(2023));
    let message = is_invalid(
        w.host
            .commit(
                &ctx(),
                commit_request(&scope, "away", vec![fact]),
                Trust::Agent,
            )
            .await,
    );
    assert!(message.contains("expires_at"), "{message}");
    assert_eq!(w.count("episodes").await, 0);
    assert_eq!(w.count("facts").await, 0);
}

// ---- Single-valued predicates ------------------------------------------------------------------

/// Where Ada lives, with the city as the object, as a host would send it.
fn lives_in(city: &str) -> CommitFact {
    let mut fact = add_fact(&format!("Ada lives in {city}"), "Ada");
    fact.predicate = "lives_in".into();
    fact.object = Some(EntityRef {
        name: city.into(),
        kind: "place".into(),
    });
    fact
}

/// Commits one fact under `ctx` as an agent key and returns the response entry and episode.
async fn commit_fact(
    host: &HostCommitter,
    ctx: &RequestContext,
    scope: &Scope,
    fact: CommitFact,
) -> (CommittedFact, Uuid) {
    let request = commit_request(scope, &format!("source of {}", fact.statement), vec![fact]);
    let mut response = host
        .commit(ctx, request, Trust::Agent)
        .await
        .expect("commit");
    (response.facts.remove(0), response.episode_id)
}

impl World {
    /// A current `lives_in` fact on Ada at `trust`, stored directly because commits stop at agent.
    async fn seed_lives_in(&self, scope: &Scope, trust: Trust, city: &str) -> Fact {
        let statement = format!("Ada lives in {city}");
        let episode = self
            .store
            .insert_episode(
                &ctx(),
                NewEpisode {
                    scope: scope.clone(),
                    source: "seed".into(),
                    trust,
                    content: statement.clone(),
                    occurred_at: Some(date(2020)),
                    metadata: json!({}),
                    idempotency_key: None,
                },
            )
            .await
            .expect("insert episode");
        let subject_id = self.seed_entity(scope, "Ada", vector_for("Ada")).await;
        let ids = self
            .store
            .apply_commit(
                &ctx(),
                episode.id,
                vec![FactOp::Add(NewFact {
                    scope: scope.clone(),
                    subject_id,
                    predicate: "lives_in".into(),
                    object_id: None,
                    embedding: vector_for(&statement),
                    statement,
                    trust,
                    status: FactStatus::Active,
                    valid_from: date(2020),
                    valid_to: None,
                    expires_at: None,
                })],
            )
            .await
            .expect("apply");
        self.store.get_fact(&ctx(), ids[0]).await.expect("fact")
    }

    async fn is_current(&self, id: Uuid) -> bool {
        let fact = self.store.get_fact(&ctx(), id).await.expect("fact");
        fact.status == FactStatus::Active && fact.superseded_by.is_none() && fact.valid_to.is_none()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_new_value_of_a_single_valued_predicate_closes_the_old_one() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let (old, _) = commit_fact(&w.host, &ctx(), &scope, lives_in("Jakarta")).await;
    // A multi-valued fact on the same subject is never touched.
    let (likes, _) = commit_fact(&w.host, &ctx(), &scope, add_fact("Ada likes tea", "Ada")).await;

    let (new, episode) = commit_fact(&w.host, &ctx(), &scope, lives_in("Bandung")).await;
    assert_eq!(new.status, FactStatus::Active);
    assert!(!new.corroborated);
    assert_eq!(new.superseded, Some(old.id));
    assert!(
        new.related.iter().all(|r| r.id != old.id),
        "{:?}",
        new.related
    );
    let closed = w.store.get_fact(&ctx(), old.id).await.expect("old");
    assert_eq!(closed.superseded_by, Some(new.id));
    assert!(closed.valid_to.is_some());
    assert!(w.is_current(new.id).await);
    assert!(w.is_current(likes.id).await);
    let now = w.recall_ids(&scope, "Ada lives in", None).await;
    assert!(now.contains(&new.id) && !now.contains(&old.id), "{now:?}");
    let history: Vec<Uuid> = w
        .store
        .fact_history(&ctx(), new.id)
        .await
        .expect("history")
        .into_iter()
        .map(|fact| fact.id)
        .collect();
    assert_eq!(history, [old.id, new.id]);

    // Purging the commit that closed the old value reopens it, as any supersede purge does.
    w.store.purge_episode(&ctx(), episode).await.expect("purge");
    assert!(w.is_current(old.id).await);
    let reopened = w.store.get_fact(&ctx(), old.id).await.expect("old");
    assert_eq!(reopened.superseded_by, None);

    // Saying the current value again is a repeat, and closes nothing.
    let (repeat, _) = commit_fact(&w.host, &ctx(), &scope, lives_in("Jakarta")).await;
    assert_eq!((repeat.id, repeat.corroborated), (old.id, true));
    assert_eq!(repeat.superseded, None);
    assert!(w.is_current(old.id).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_predicate_matches_after_normalizing_to_snake_case() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let mut first = lives_in("Jakarta");
    first.predicate = "Lives In".into();
    let (old, _) = commit_fact(&w.host, &ctx(), &scope, first).await;
    let mut second = lives_in("Bandung");
    second.predicate = "livesIn".into();
    let (new, _) = commit_fact(&w.host, &ctx(), &scope, second).await;
    assert_eq!(new.superseded, Some(old.id));
    assert!(!w.is_current(old.id).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_quarantined_new_value_waits_for_approval_to_close_the_old_one() {
    let w = world().await;
    // An agent's commit to a user scope is quarantined whatever it says. The first value is
    // approved, so the old fact is current at the same trust as the new one.
    let scope = Scope::user("u1");
    let (old, _) = commit_fact(&w.host, &ctx(), &scope, lives_in("Jakarta")).await;
    assert_eq!(
        (old.status, old.superseded),
        (FactStatus::Quarantined, None)
    );
    w.store
        .approve_fact(&ctx(), old.id)
        .await
        .expect("approve old");
    let (new, _) = commit_fact(&w.host, &ctx(), &scope, lives_in("Bandung")).await;
    assert_eq!(new.status, FactStatus::Quarantined);
    assert_eq!(new.superseded, Some(old.id));
    let held = w.store.get_fact(&ctx(), new.id).await.expect("new");
    assert_eq!(held.pending_supersede_of, Some(old.id));
    assert!(w.is_current(old.id).await);

    w.store.approve_fact(&ctx(), new.id).await.expect("approve");
    let closed = w.store.get_fact(&ctx(), old.id).await.expect("old");
    assert_eq!(closed.superseded_by, Some(new.id));
    assert!(w.is_current(new.id).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_key_below_the_old_facts_trust_does_not_close_it_and_lists_it() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let old = w.seed_lives_in(&scope, Trust::User, "Jakarta").await;
    let (new, _) = commit_fact(&w.host, &ctx(), &scope, lives_in("Bandung")).await;
    assert_eq!(new.superseded, None);
    assert_eq!(new.related.first().map(|r| r.id), Some(old.id));
    assert_eq!(new.related[0].statement, "Ada lives in Jakarta");
    assert!(w.is_current(old.id).await);
    // The store's subject and predicate rule still holds the lower-trust fact for review.
    let added = w.store.get_fact(&ctx(), new.id).await.expect("new");
    assert_eq!(added.pending_supersede_of, None);
    assert_eq!(new.status, FactStatus::Quarantined);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn another_scope_or_tenant_is_never_touched() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let globex = tenant_ctx("globex");
    let (other_scope, _) =
        commit_fact(&w.host, &ctx(), &Scope::agent("a2"), lives_in("Jakarta")).await;
    let (other_tenant, _) = commit_fact(&w.host, &globex, &scope, lives_in("Jakarta")).await;

    let (mine, _) = commit_fact(&w.host, &ctx(), &scope, lives_in("Bandung")).await;
    assert_eq!(mine.superseded, None);
    assert!(w.is_current(other_scope.id).await);
    let foreign = w
        .store
        .get_fact(&globex, other_tenant.id)
        .await
        .expect("other tenant");
    assert_eq!((foreign.superseded_by, foreign.valid_to), (None, None));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_expired_value_or_a_turned_off_list_closes_nothing() {
    let w = world().await;
    let scope = Scope::agent("a1");
    let mut leave = lives_in("Jakarta");
    leave.valid_from = Some(Utc::now() - chrono::Duration::days(10));
    leave.expires_at = Some(Utc::now() - chrono::Duration::days(1));
    let (expired, _) = commit_fact(&w.host, &ctx(), &scope, leave).await;
    let (new, _) = commit_fact(&w.host, &ctx(), &scope, lives_in("Bandung")).await;
    assert_eq!(new.superseded, None);
    let untouched = w.store.get_fact(&ctx(), expired.id).await.expect("expired");
    assert_eq!(untouched.superseded_by, None);

    let off = HostCommitter::new(w.store.clone(), w.embedder.clone())
        .with_single_valued(SingleValued::parse(","));
    let (other, _) = commit_fact(&off, &ctx(), &scope, lives_in("Surabaya")).await;
    assert_eq!(other.superseded, None);
    assert!(w.is_current(new.id).await);
    assert!(w.is_current(other.id).await);

    // A configured list replaces the default.
    let custom = HostCommitter::new(w.store.clone(), w.embedder.clone())
        .with_single_valued(SingleValued::parse("likes"));
    let likes = |text: &str| {
        let mut fact = add_fact(text, "Ada");
        fact.predicate = "likes".into();
        fact
    };
    let (tea, _) = commit_fact(&custom, &ctx(), &scope, likes("Ada likes tea")).await;
    let (coffee, _) = commit_fact(&custom, &ctx(), &scope, likes("Ada likes coffee")).await;
    assert_eq!(coffee.superseded, Some(tea.id));
    let (medan, _) = commit_fact(&custom, &ctx(), &scope, lives_in("Medan")).await;
    assert_eq!(medan.superseded, None);
}
