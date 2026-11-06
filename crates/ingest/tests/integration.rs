//! The write path against real PostgreSQL, with the deterministic test embedder. One container per
//! test binary. Each test clones a migrated template into its own database, so tests run in parallel.

#![cfg(feature = "integration")]

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
use memoryos_ingest::{CommitFact, CommitRequest, EntityRef, HostCommitter};
use memoryos_store::PgStore;
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
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
    sqlx::query(&format!("CREATE DATABASE {TEMPLATE_DB}"))
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
}

async fn world() -> World {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let server = postgres_url();
    let name = format!("t{}", NEXT.fetch_add(1, Ordering::SeqCst));

    let bootstrap = connect_with_retry(&format!("{server}/postgres")).await;
    sqlx::query(&format!("CREATE DATABASE {name} TEMPLATE {TEMPLATE_DB}"))
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
        sqlx::query_scalar(&format!("SELECT count(*) FROM {table}"))
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
    let scope = Scope::agent("a1");
    let mut knows = add_fact("Ada knows Bob", "Ada");
    knows.object = Some(EntityRef {
        name: "Bob".into(),
        kind: "person".into(),
    });
    let mut tea = add_fact("Ada likes tea", "ada");
    tea.valid_from = Some(date(2020));
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "chat text", vec![knows, tea]),
            Trust::Agent,
        )
        .await
        .expect("commit");

    assert_eq!(response.facts.len(), 2);
    assert!(response
        .facts
        .iter()
        .all(|f| f.status == FactStatus::Active));
    let episode = w
        .store
        .get_episode(&ctx(), response.episode_id)
        .await
        .expect("episode");
    assert_eq!(
        (episode.trust, episode.content.as_str()),
        (Trust::Agent, "chat text")
    );
    assert!(episode.processed_at.is_some());
    assert_eq!(w.count("entities").await, 2);

    let knows = w
        .store
        .get_fact(&ctx(), response.facts[0].id)
        .await
        .expect("fact");
    assert_eq!(knows.valid_from, episode.occurred_at);
    assert!(knows.object_id.is_some());
    let tea = w
        .store
        .get_fact(&ctx(), response.facts[1].id)
        .await
        .expect("fact");
    assert_eq!(tea.valid_from, date(2020));
    assert_eq!(tea.subject_id, knows.subject_id);

    // One provider call, statements first, then each distinct entity.
    assert_eq!(
        w.embedder.batches(),
        [["Ada knows Bob", "Ada likes tea", "ada", "bob"]]
    );
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
        let request = commit_request(scope, content, vec![add_fact("fact", "Ada")]);
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
    let scope = Scope::agent("a1");
    let old_id = commit_one(&w, &scope, "Ada lives in Rome", "Ada").await;

    let mut fact = add_fact("Ada lives in Oslo", "Ada");
    fact.supersedes = Some(old_id);
    let response = w
        .host
        .commit(
            &ctx(),
            commit_request(&scope, "moved", vec![fact]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    assert_eq!(response.facts[0].status, FactStatus::Active);

    let old = w.store.get_fact(&ctx(), old_id).await.expect("old");
    assert_eq!(old.superseded_by, Some(response.facts[0].id));
    assert!(old.valid_to.is_some());
    assert_eq!(
        w.recall_ids(&scope, "Ada lives in Rome", None).await,
        vec![response.facts[0].id]
    );
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
