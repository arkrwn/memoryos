//! `Retriever` against real Postgres and a real `OpenAiEmbedder` talking to WireMock. One pair of
//! containers per test binary; each test clones a migrated template into its own database.
//!
//! Vector search returns the nearest 50 facts however far they are, so "found only by" tests seed
//! 50 closer fillers and confirm the precondition by calling the store directly.

#![cfg(feature = "integration")]

use std::collections::BTreeSet;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, TimeZone, Utc};
use memoryos_core::{
    Embedder, Error, FactFilter, FactOp, FactStatus, MemoryStore, NewEntity, NewEpisode, NewFact,
    RequestContext, Scope, ScoredFact, Trust, EMBEDDING_DIMS,
};
use memoryos_llm::OpenAiEmbedder;
use memoryos_retrieve::{RecallRequest, RecallResponse, Retriever};
use memoryos_store::PgStore;
use serde_json::{json, Value};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{GenericImage, ImageExt};
use uuid::Uuid;

const TEMPLATE_DB: &str = "memoryos_template";
const EMBEDDINGS_URL: &str = "/v1/embeddings";

/// Embeds to the query direction, angle 0.
const QUERY: &str = "zebra";
/// WireMock answers 500 for this input.
const FAILING_QUERY: &str = "failing";
/// WireMock answers with the zero vector, which the store's vector search refuses.
const ZERO_QUERY: &str = "zeroed";
/// WireMock answers correctly, after `SLOW_REPLY_MS`.
const SLOW_QUERY: &str = "slowed";
const SLOW_REPLY_MS: u64 = 3000;
/// Far from the query direction, behind every filler.
const FAR: f32 = 170.0;

// ---- Containers ------------------------------------------------------------------------------

struct Servers {
    postgres_url: String,
    wiremock_url: String,
}

fn servers() -> &'static Servers {
    static SERVERS: OnceLock<Servers> = OnceLock::new();
    SERVERS.get_or_init(|| {
        std::thread::spawn(|| {
            tokio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(async {
                    let (postgres_url, wiremock_url) =
                        tokio::join!(start_postgres(), start_wiremock());
                    Servers {
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
        .with_cmd(["postgres", "-c", "max_connections=400", "-c", "fsync=off"])
        .start()
        .await
        .expect("start pgvector container");
    let port = container.get_host_port_ipv4(5432).await.expect("port");
    reap_on_exit(container.id());
    // A static would never drop the handle anyway. The reaper removes the container.
    std::mem::forget(container);

    let url = format!("postgres://postgres:postgres@127.0.0.1:{port}");
    let admin = connect_with_retry(&format!("{url}/postgres")).await;
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

async fn start_wiremock() -> String {
    let container = GenericImage::new("wiremock/wiremock", "3.3.1")
        .with_exposed_port(8080.tcp())
        .with_wait_for(WaitFor::message_on_stdout("port:"))
        .start()
        .await
        .expect("start wiremock container");
    let port = container.get_host_port_ipv4(8080).await.expect("port");
    reap_on_exit(container.id());
    std::mem::forget(container);

    let url = format!("http://127.0.0.1:{port}");
    let http = reqwest::Client::new();
    wait_until_ready(&http, &url).await;
    stub_input(&http, &url, QUERY, 200, embeddings_reply(&at_angle(0.0))).await;
    stub_input(
        &http,
        &url,
        ZERO_QUERY,
        200,
        embeddings_reply(&vec![0.0; EMBEDDING_DIMS]),
    )
    .await;
    stub_input(
        &http,
        &url,
        FAILING_QUERY,
        500,
        json!({"error": {"message": "down"}}),
    )
    .await;
    let slow = json!({
        "request": {
            "method": "POST",
            "url": EMBEDDINGS_URL,
            "bodyPatterns": [{
                "equalToJson": {"input": [SLOW_QUERY]},
                "ignoreExtraElements": true,
            }],
        },
        "response": {
            "status": 200,
            "fixedDelayMilliseconds": SLOW_REPLY_MS,
            "jsonBody": embeddings_reply(&at_angle(0.0)),
        },
    });
    let response = http
        .post(format!("{url}/__admin/mappings"))
        .json(&slow)
        .send()
        .await
        .expect("register slow stub");
    assert_eq!(response.status().as_u16(), 201, "stub rejected");
    url
}

/// The startup banner can come before the admin API accepts connections.
async fn wait_until_ready(http: &reqwest::Client, url: &str) {
    for _ in 0..300 {
        if let Ok(response) = http.get(format!("{url}/__admin/mappings")).send().await {
            if response.status().is_success() {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("wiremock admin API did not become ready");
}

async fn stub_input(http: &reqwest::Client, url: &str, input: &str, status: u16, body: Value) {
    let mapping = json!({
        "request": {
            "method": "POST",
            "url": EMBEDDINGS_URL,
            "bodyPatterns": [{
                "equalToJson": {"input": [input]},
                "ignoreExtraElements": true,
            }],
        },
        "response": {"status": status, "jsonBody": body},
    });
    let response = http
        .post(format!("{url}/__admin/mappings"))
        .json(&mapping)
        .send()
        .await
        .expect("register stub");
    assert_eq!(response.status().as_u16(), 201, "stub rejected");
}

fn embeddings_reply(vector: &[f32]) -> Value {
    json!({
        "object": "list",
        "model": "test",
        "data": [{"object": "embedding", "index": 0, "embedding": vector}],
    })
}

async fn embedding_requests(input: &str) -> u64 {
    let pattern = json!({
        "method": "POST",
        "url": EMBEDDINGS_URL,
        "bodyPatterns": [{"equalToJson": {"input": [input]}, "ignoreExtraElements": true}],
    });
    let reply: Value = reqwest::Client::new()
        .post(format!("{}/__admin/requests/count", servers().wiremock_url))
        .json(&pattern)
        .send()
        .await
        .expect("count requests")
        .json()
        .await
        .expect("count body");
    reply["count"].as_u64().expect("count field")
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

// ---- Fixture ---------------------------------------------------------------------------------

struct Fixture {
    store: Arc<PgStore>,
    /// Superuser, for injecting faults the store API refuses to create.
    admin: PgPool,
    retriever: Retriever,
    ctx: RequestContext,
    scope: Scope,
}

async fn fixture() -> Fixture {
    static NEXT_DB: AtomicUsize = AtomicUsize::new(0);
    let servers = servers();
    let name = format!("recall_{}", NEXT_DB.fetch_add(1, Ordering::Relaxed));
    let admin = connect_with_retry(&format!("{}/postgres", servers.postgres_url)).await;
    sqlx::query(&format!("CREATE DATABASE {name} TEMPLATE {TEMPLATE_DB}"))
        .execute(&admin)
        .await
        .expect("clone template");
    admin.close().await;

    let url = format!("{}/{name}", servers.postgres_url);
    let store = Arc::new(PgStore::connect(&url, 16).await.expect("connect store"));
    let embedder: Arc<dyn Embedder> = Arc::new(OpenAiEmbedder::new(
        "test-key-not-a-real-secret".into(),
        "test".into(),
        Some(format!("{}/v1", servers.wiremock_url)),
    ));
    Fixture {
        retriever: Retriever::new(store.clone(), embedder),
        store,
        admin: connect_with_retry(&url).await,
        ctx: tenant("tenant-a"),
        scope: Scope::user("ada"),
    }
}

fn tenant(name: &str) -> RequestContext {
    RequestContext {
        tenant: name.into(),
        actor: "test".into(),
    }
}

/// Unit vector `degrees` away from the query direction, so cosine similarity is its cosine.
fn at_angle(degrees: f32) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIMS];
    vector[0] = degrees.to_radians().cos();
    vector[1] = degrees.to_radians().sin();
    vector
}

fn date(year: i32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0)
        .single()
        .expect("valid date")
}

fn new_fact(scope: &Scope, subject: Uuid, statement: &str, degrees: f32) -> NewFact {
    NewFact {
        scope: scope.clone(),
        subject_id: subject,
        predicate: "says".into(),
        object_id: None,
        statement: statement.into(),
        embedding: at_angle(degrees),
        trust: Trust::User,
        status: FactStatus::Active,
        valid_from: date(2020),
        valid_to: None,
    }
}

fn episode(scope: &Scope, trust: Trust, content: &str) -> NewEpisode {
    NewEpisode {
        scope: scope.clone(),
        source: "test".into(),
        trust,
        content: content.into(),
        occurred_at: None,
        metadata: json!({}),
        idempotency_key: None,
    }
}

impl Fixture {
    async fn entity_in(&self, ctx: &RequestContext, scope: &Scope, name: &str) -> Uuid {
        let entity = NewEntity {
            scope: scope.clone(),
            kind: "thing".into(),
            name: name.into(),
            aliases: vec![],
            embedding: at_angle(0.0),
        };
        self.store
            .upsert_entity(ctx, entity)
            .await
            .expect("upsert entity")
            .id
    }

    async fn entity(&self, name: &str) -> Uuid {
        self.entity_in(&self.ctx, &self.scope, name).await
    }

    /// Stores one processed episode carrying `ops`, with the trust the ops' facts must have.
    async fn apply_in(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        trust: Trust,
        ops: Vec<FactOp>,
    ) -> Vec<Uuid> {
        let source = self
            .store
            .insert_unqueued_episode(ctx, episode(scope, trust, "source"))
            .await
            .expect("insert episode");
        self.store
            .apply_extraction(ctx, source.id, ops)
            .await
            .expect("apply extraction")
    }

    async fn add(&self, subject: Uuid, statement: &str, degrees: f32) -> Uuid {
        let fact = new_fact(&self.scope, subject, statement, degrees);
        let ops = vec![FactOp::Add(fact)];
        self.apply_in(&self.ctx, &self.scope, Trust::User, ops)
            .await[0]
    }

    /// Adds `count` unrelated facts closer to the query than anything at `FAR`.
    async fn add_fillers(&self, count: usize) {
        let subject = self.entity("filler").await;
        let ops = (0..count)
            .map(|i| {
                let degrees = 1.0 + i as f32;
                let statement = format!("unrelated filler number {i}");
                FactOp::Add(new_fact(&self.scope, subject, &statement, degrees))
            })
            .collect();
        self.apply_in(&self.ctx, &self.scope, Trust::User, ops)
            .await;
    }

    async fn vector_hits(&self, filter: &FactFilter) -> BTreeSet<Uuid> {
        let found = self
            .store
            .search_facts_vector(
                &self.ctx,
                std::slice::from_ref(&self.scope),
                &at_angle(0.0),
                filter,
                50,
            )
            .await
            .expect("vector search");
        ids(&found)
    }

    async fn text_hits(&self, filter: &FactFilter) -> BTreeSet<Uuid> {
        let found = self
            .store
            .search_facts_text(
                &self.ctx,
                std::slice::from_ref(&self.scope),
                QUERY,
                filter,
                50,
            )
            .await
            .expect("text search");
        ids(&found)
    }

    fn request(&self, query: &str) -> RecallRequest {
        RecallRequest {
            scopes: vec![self.scope.clone()],
            query: query.into(),
            limit: 100,
            as_of: None,
            known_at: None,
        }
    }

    async fn recall(&self, query: &str) -> RecallResponse {
        self.retriever
            .recall(&self.ctx, self.request(query))
            .await
            .expect("recall")
    }
}

fn ids(facts: &[ScoredFact]) -> BTreeSet<Uuid> {
    facts.iter().map(|scored| scored.fact.id).collect()
}

fn score_of(facts: &[ScoredFact], id: Uuid) -> f64 {
    facts
        .iter()
        .find(|scored| scored.fact.id == id)
        .expect("fact in result")
        .score
}

fn is_invalid<T>(result: memoryos_core::Result<T>) -> bool {
    matches!(result, Err(Error::InvalidInput(_)))
}

// ---- Each list on its own --------------------------------------------------------------------

#[tokio::test]
async fn finds_a_fact_only_by_vector() {
    let f = fixture().await;
    let subject = f.entity("a").await;
    let target = f
        .add(subject, "nothing in common with the query", 1.0)
        .await;

    let filter = FactFilter::default();
    assert!(f.vector_hits(&filter).await.contains(&target));
    assert!(!f.text_hits(&filter).await.contains(&target));

    assert!(ids(&f.recall(QUERY).await.facts).contains(&target));
}

#[tokio::test]
async fn finds_a_fact_only_by_text() {
    let f = fixture().await;
    f.add_fillers(50).await;
    let subject = f.entity("a").await;
    let target = f.add(subject, "a zebra was seen", FAR).await;

    let filter = FactFilter::default();
    assert!(!f.vector_hits(&filter).await.contains(&target));
    assert!(f.text_hits(&filter).await.contains(&target));

    assert!(ids(&f.recall(QUERY).await.facts).contains(&target));
}

#[tokio::test]
async fn finds_a_fact_only_by_graph() {
    let f = fixture().await;
    f.add_fillers(50).await;
    let subject = f.entity("a").await;
    let seed = f.add(subject, "a zebra was seen", 0.5).await;
    let target = f.add(subject, "shares only a subject", FAR).await;

    let filter = FactFilter::default();
    let (vector, text) = (f.vector_hits(&filter).await, f.text_hits(&filter).await);
    assert!(!vector.contains(&target) && !text.contains(&target));
    assert!(text.contains(&seed));

    let facts = f.recall(QUERY).await.facts;
    assert!(ids(&facts).contains(&target));
    // The seed subject is first, and within it the newest edge is the target.
    assert_eq!(score_of(&facts, target), 1.0 / 61.0);
}

#[tokio::test]
async fn graph_expansion_follows_edges_that_point_at_the_subject() {
    let f = fixture().await;
    f.add_fillers(50).await;
    let (subject, other) = (f.entity("a").await, f.entity("b").await);
    f.add(subject, "a zebra was seen", 0.5).await;
    let mut edge = new_fact(&f.scope, other, "points at the subject", FAR);
    edge.object_id = Some(subject);
    let target = f
        .apply_in(&f.ctx, &f.scope, Trust::User, vec![FactOp::Add(edge)])
        .await[0];

    assert!(ids(&f.recall(QUERY).await.facts).contains(&target));
}

#[tokio::test]
async fn only_the_top_five_fused_facts_seed_the_graph() {
    let f = fixture().await;
    // Seven subjects, each with a seed fact near the query and a sibling far from it.
    let mut siblings = Vec::new();
    for i in 0..7 {
        let subject = f.entity(&format!("s{i}")).await;
        f.add(subject, "seed fact", 1.0 + i as f32).await;
        siblings.push(f.add(subject, "sibling fact", FAR).await);
    }
    // The seeds plus these 43 fill the vector list, which keeps every sibling out of it.
    let filler = f.entity("filler").await;
    let ops = (0..43)
        .map(|i| {
            let degrees = 20.0 + i as f32;
            FactOp::Add(new_fact(&f.scope, filler, "far filler", degrees))
        })
        .collect();
    f.apply_in(&f.ctx, &f.scope, Trust::User, ops).await;

    let vector = f.vector_hits(&FactFilter::default()).await;
    assert!(siblings.iter().all(|id| !vector.contains(id)));

    let found = ids(&f.recall(QUERY).await.facts);
    let reached: Vec<bool> = siblings.iter().map(|id| found.contains(id)).collect();
    assert_eq!(reached, [true, true, true, true, true, false, false]);
}

#[tokio::test]
async fn graph_candidates_are_capped_at_50() {
    let f = fixture().await;
    let hub = f.entity("hub").await;
    // Newest is nearest, so the vector list and the newest edges are the same 50 facts. Any edge
    // past the cap would be an older fact and make the result longer than 50.
    let ops = (0..120)
        .map(|i| {
            let degrees = 121.0 - i as f32;
            let statement = if i == 119 { "a zebra" } else { "hub edge" };
            FactOp::Add(new_fact(&f.scope, hub, statement, degrees))
        })
        .collect();
    let inserted = f.apply_in(&f.ctx, &f.scope, Trust::User, ops).await;

    let facts = f.recall(QUERY).await.facts;
    let newest: BTreeSet<Uuid> = inserted[70..].iter().copied().collect();
    assert_eq!(ids(&facts), newest);
}

// ---- Fusion ----------------------------------------------------------------------------------

#[tokio::test]
async fn a_fact_in_both_lists_ranks_higher() {
    let f = fixture().await;
    let (a, b) = (f.entity("a").await, f.entity("b").await);
    let vector_only = f.add(a, "words that do not match", 1.0).await;
    let both = f.add(b, "a zebra was seen", 2.0).await;

    let facts = f.recall(QUERY).await.facts;
    assert_eq!(facts[0].fact.id, both);
    assert_eq!(facts[1].fact.id, vector_only);
    // Vector rank 2, text rank 1, graph rank 1.
    assert_eq!(facts[0].score, 1.0 / 62.0 + 1.0 / 61.0 + 1.0 / 61.0);
    // Vector rank 1, graph rank 2.
    assert_eq!(facts[1].score, 1.0 / 61.0 + 1.0 / 62.0);
}

#[tokio::test]
async fn each_fact_appears_once_ordered_by_score() {
    let f = fixture().await;
    let subject = f.entity("a").await;
    for i in 0..8 {
        f.add(subject, "a zebra was seen", i as f32).await;
    }
    let facts = f.recall(QUERY).await.facts;
    assert_eq!(facts.len(), 8);
    assert_eq!(ids(&facts).len(), 8);
    assert!(facts.windows(2).all(|pair| pair[0].score >= pair[1].score));
}

// ---- Time ------------------------------------------------------------------------------------

#[tokio::test]
async fn superseded_fact_is_absent_now_and_present_with_as_of() {
    let f = fixture().await;
    let subject = f.entity("a").await;
    let old = f.add(subject, "the zebra lives in Paris", 1.0).await;
    let mut replacement = new_fact(&f.scope, subject, "the zebra lives in Rome", 1.0);
    replacement.valid_from = date(2022);
    let op = FactOp::Supersede {
        old_fact_id: old,
        new: replacement,
    };
    let new = f.apply_in(&f.ctx, &f.scope, Trust::User, vec![op]).await[0];

    let now = ids(&f.recall(QUERY).await.facts);
    assert!(now.contains(&new) && !now.contains(&old));

    let mut past = f.request(QUERY);
    past.as_of = Some(date(2021));
    let then = f.retriever.recall(&f.ctx, past).await.expect("recall");
    let then = ids(&then.facts);
    assert!(then.contains(&old) && !then.contains(&new));
}

#[tokio::test]
async fn graph_expansion_uses_the_time_filter() {
    let f = fixture().await;
    f.add_fillers(50).await;
    let subject = f.entity("a").await;
    f.add(subject, "a zebra was seen", 0.5).await;
    let old = f.add(subject, "first version", FAR).await;
    let mut replacement = new_fact(&f.scope, subject, "second version", FAR);
    replacement.valid_from = date(2022);
    let op = FactOp::Supersede {
        old_fact_id: old,
        new: replacement,
    };
    let new = f.apply_in(&f.ctx, &f.scope, Trust::User, vec![op]).await[0];

    let now = ids(&f.recall(QUERY).await.facts);
    assert!(now.contains(&new) && !now.contains(&old));

    let filter = FactFilter {
        as_of: Some(date(2021)),
        known_at: None,
    };
    assert!(!f.vector_hits(&filter).await.contains(&old));
    assert!(!f.text_hits(&filter).await.contains(&old));
    let mut past = f.request(QUERY);
    past.as_of = filter.as_of;
    let then = f.retriever.recall(&f.ctx, past).await.expect("recall");
    let then = ids(&then.facts);
    assert!(then.contains(&old) && !then.contains(&new));
}

// ---- Visibility ------------------------------------------------------------------------------

#[tokio::test]
async fn quarantined_and_rejected_facts_are_absent_from_every_list() {
    let f = fixture().await;
    let subject = f.entity("a").await;
    let visible = f.add(subject, "a zebra was seen", 0.5).await;

    // Each could match all three lists: same words, near vector, same subject as the seed.
    let mut poison = new_fact(&f.scope, subject, "a zebra is poison", 0.0);
    poison.trust = Trust::External;
    poison.status = FactStatus::Quarantined;
    let ops = vec![FactOp::Add(poison.clone()), FactOp::Add(poison)];
    let held = f.apply_in(&f.ctx, &f.scope, Trust::External, ops).await;
    let (quarantined, rejected) = (held[0], held[1]);
    f.store.reject_fact(&f.ctx, rejected).await.expect("reject");

    let filter = FactFilter::default();
    let scopes = [f.scope.clone()];
    let neighborhood = f
        .store
        .neighbors(&f.ctx, &scopes, subject, 1, &filter, 5000)
        .await
        .expect("neighbors");
    let edges: BTreeSet<Uuid> = neighborhood.edges.iter().map(|edge| edge.id).collect();

    let found = ids(&f.recall(QUERY).await.facts);
    assert!(found.contains(&visible));
    for hidden in [quarantined, rejected] {
        assert!(!found.contains(&hidden));
        assert!(!f.vector_hits(&filter).await.contains(&hidden));
        assert!(!f.text_hits(&filter).await.contains(&hidden));
        assert!(!edges.contains(&hidden));
    }
}

#[tokio::test]
async fn other_tenant_is_absent() {
    let f = fixture().await;
    let other = tenant("tenant-b");
    let mine = f.add(f.entity("a").await, "a zebra was seen", 1.0).await;
    let subject = f.entity_in(&other, &f.scope, "a").await;
    let fact = new_fact(&f.scope, subject, "a zebra was seen", 1.0);
    let theirs = f
        .apply_in(&other, &f.scope, Trust::User, vec![FactOp::Add(fact)])
        .await[0];
    f.store
        .insert_episode(&other, episode(&f.scope, Trust::User, "a zebra is pending"))
        .await
        .expect("insert episode");

    let response = f.recall(QUERY).await;
    assert_eq!(ids(&response.facts), BTreeSet::from([mine]));
    assert!(response.pending_episodes.is_empty());

    let response = f
        .retriever
        .recall(&other, f.request(QUERY))
        .await
        .expect("recall");
    assert_eq!(ids(&response.facts), BTreeSet::from([theirs]));
    assert_eq!(response.pending_episodes.len(), 1);
}

#[tokio::test]
async fn other_scope_is_absent_unless_requested() {
    let f = fixture().await;
    let elsewhere = Scope::shared("elsewhere");
    let mine = f.add(f.entity("a").await, "a zebra was seen", 1.0).await;
    let subject = f.entity_in(&f.ctx, &elsewhere, "a").await;
    let fact = new_fact(&elsewhere, subject, "a zebra was seen", 1.0);
    let theirs = f
        .apply_in(&f.ctx, &elsewhere, Trust::User, vec![FactOp::Add(fact)])
        .await[0];
    f.store
        .insert_episode(
            &f.ctx,
            episode(&elsewhere, Trust::User, "a zebra is pending"),
        )
        .await
        .expect("insert episode");

    let response = f.recall(QUERY).await;
    assert_eq!(ids(&response.facts), BTreeSet::from([mine]));
    assert!(response.pending_episodes.is_empty());

    let mut both = f.request(QUERY);
    both.scopes.push(elsewhere);
    let response = f.retriever.recall(&f.ctx, both).await.expect("recall");
    assert_eq!(ids(&response.facts), BTreeSet::from([mine, theirs]));
    assert_eq!(response.pending_episodes.len(), 1);
}

// ---- Failure handling ------------------------------------------------------------------------

#[tokio::test]
async fn slow_embedding_times_out_and_still_returns_text_results() {
    let f = fixture().await;
    let (a, b) = (f.entity("a").await, f.entity("b").await);
    let text_hit = f.add(a, "the slowed report", 170.0).await;
    // Near the stubbed vector, so it would appear if the embedding were waited for.
    f.add(b, "no shared words", 0.0).await;

    let embedder: Arc<dyn Embedder> = Arc::new(OpenAiEmbedder::new(
        "test-key-not-a-real-secret".into(),
        "test".into(),
        Some(format!("{}/v1", servers().wiremock_url)),
    ));
    let short =
        Retriever::new(f.store.clone(), embedder).with_embed_timeout(Duration::from_millis(200));
    let started = Instant::now();
    let response = short
        .recall(&f.ctx, f.request(SLOW_QUERY))
        .await
        .expect("recall");
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
    assert_eq!(ids(&response.facts), BTreeSet::from([text_hit]));

    // The default gives up at 2 seconds, before the 3 second reply.
    let started = Instant::now();
    let response = f
        .retriever
        .recall(&f.ctx, f.request(SLOW_QUERY))
        .await
        .expect("recall");
    let elapsed = started.elapsed();
    assert!(elapsed >= Duration::from_millis(1900), "{elapsed:?}");
    assert!(
        elapsed < Duration::from_millis(SLOW_REPLY_MS),
        "{elapsed:?}"
    );
    assert_eq!(ids(&response.facts), BTreeSet::from([text_hit]));
}

#[tokio::test]
async fn graph_expansion_reads_a_hubs_newest_edges() {
    let f = fixture().await;
    f.add_fillers(50).await;
    let hub = f.entity("hub").await;
    let oldest = f.add(hub, "hub edge oldest", FAR).await;
    let ops = (0..60)
        .map(|i| {
            let statement = format!("hub edge {i}");
            FactOp::Add(new_fact(&f.scope, hub, &statement, FAR))
        })
        .collect();
    f.apply_in(&f.ctx, &f.scope, Trust::User, ops).await;
    let newest = f.add(hub, "hub edge newest", FAR).await;
    f.add(hub, "a zebra at the hub", FAR).await;

    let mut request = f.request(QUERY);
    request.limit = 100;
    let found = ids(&f
        .retriever
        .recall(&f.ctx, request)
        .await
        .expect("recall")
        .facts);
    assert!(found.contains(&newest));
    assert!(!found.contains(&oldest));
}

#[tokio::test]
async fn embedding_failure_still_returns_text_results() {
    let f = fixture().await;
    let (a, b) = (f.entity("a").await, f.entity("b").await);
    let text_hit = f.add(a, "the failing report", 170.0).await;
    let vector_only = f.add(b, "no shared words", 0.0).await;

    let before = embedding_requests(FAILING_QUERY).await;
    let facts = f.recall(FAILING_QUERY).await.facts;
    assert!(embedding_requests(FAILING_QUERY).await > before);

    let found = ids(&facts);
    assert!(found.contains(&text_hit));
    assert!(!found.contains(&vector_only));
}

#[tokio::test]
async fn vector_store_error_degrades_to_text_results() {
    let f = fixture().await;
    let (a, b) = (f.entity("a").await, f.entity("b").await);
    let text_hit = f.add(a, "the zeroed report", 170.0).await;
    let vector_only = f.add(b, "no shared words", 0.0).await;

    // Precondition: the store refuses the vector the embedder returns for this query.
    let refused = f
        .store
        .search_facts_vector(
            &f.ctx,
            std::slice::from_ref(&f.scope),
            &vec![0.0; EMBEDDING_DIMS],
            &FactFilter::default(),
            50,
        )
        .await;
    assert!(is_invalid(refused));

    let found = ids(&f.recall(ZERO_QUERY).await.facts);
    assert!(found.contains(&text_hit));
    assert!(!found.contains(&vector_only));
}

#[tokio::test]
async fn text_search_failure_is_an_error() {
    let f = fixture().await;
    let subject = f.entity("a").await;
    f.add(subject, "a zebra was seen", 0.0).await;
    sqlx::query(
        "ALTER FUNCTION search_fact_ids_text(text[], text, timestamptz, timestamptz, bigint) \
         RENAME TO gone",
    )
    .execute(&f.admin)
    .await
    .expect("rename function");

    let result = f.retriever.recall(&f.ctx, f.request(QUERY)).await;
    assert!(matches!(result, Err(Error::Storage(_))));
    let scopes = [f.scope.clone()];
    let result = f.retriever.candidates(&f.ctx, &scopes, QUERY, 10).await;
    assert!(matches!(result, Err(Error::Storage(_))));
}

#[tokio::test]
async fn neighbors_failure_on_one_subject_keeps_the_rest() {
    let f = fixture().await;
    f.add_fillers(50).await;
    let (broken, healthy) = (f.entity("broken").await, f.entity("healthy").await);
    let broken_seed = f.add(broken, "a zebra was seen", 0.1).await;
    let broken_sibling = f.add(broken, "broken sibling", FAR).await;
    f.add(healthy, "another zebra was seen", 0.2).await;
    let healthy_sibling = f.add(healthy, "healthy sibling", FAR).await;

    // The store keeps a fact and its subject in one scope, so a subject outside the requested
    // scopes takes direct SQL. `neighbors` then answers NotFound for this subject.
    sqlx::query("UPDATE entities SET scope = 'shared:elsewhere' WHERE id = $1")
        .bind(broken)
        .execute(&f.admin)
        .await
        .expect("move entity");
    let scopes = [f.scope.clone()];
    let failed = f
        .store
        .neighbors(&f.ctx, &scopes, broken, 1, &FactFilter::default(), 5000)
        .await;
    assert!(matches!(failed, Err(Error::NotFound(_))));

    let found = ids(&f.recall(QUERY).await.facts);
    assert!(found.contains(&broken_seed));
    assert!(found.contains(&healthy_sibling));
    assert!(!found.contains(&broken_sibling));
}

// ---- Pending episodes ------------------------------------------------------------------------

#[tokio::test]
async fn pending_episodes_follow_trust_and_time() {
    let f = fixture().await;
    let insert = |trust, content: &'static str| {
        f.store
            .insert_episode(&f.ctx, episode(&f.scope, trust, content))
    };
    let user = insert(Trust::User, "user zebra").await.expect("insert");
    let system = insert(Trust::System, "system zebra").await.expect("insert");
    insert(Trust::Agent, "agent zebra").await.expect("insert");
    insert(Trust::External, "external zebra")
        .await
        .expect("insert");
    let unrelated = insert(Trust::User, "other words").await.expect("insert");
    let processed = insert(Trust::User, "done zebra").await.expect("insert");
    f.store
        .apply_extraction(&f.ctx, processed.id, vec![])
        .await
        .expect("mark processed");

    let response = f.recall(QUERY).await;
    let pending: BTreeSet<Uuid> = response
        .pending_episodes
        .iter()
        .map(|scored| scored.episode.id)
        .collect();
    assert_eq!(pending, BTreeSet::from([user.id, system.id]));
    assert!(!pending.contains(&unrelated.id));

    let mut with_as_of = f.request(QUERY);
    with_as_of.as_of = Some(Utc::now());
    let response = f
        .retriever
        .recall(&f.ctx, with_as_of)
        .await
        .expect("recall");
    assert!(response.pending_episodes.is_empty());

    let mut with_known_at = f.request(QUERY);
    with_known_at.known_at = Some(Utc::now());
    let response = f
        .retriever
        .recall(&f.ctx, with_known_at)
        .await
        .expect("recall");
    assert!(response.pending_episodes.is_empty());
}

// ---- Candidates ------------------------------------------------------------------------------

#[tokio::test]
async fn candidates_match_recall_facts_and_stay_current() {
    let f = fixture().await;
    let subject = f.entity("a").await;
    let old = f.add(subject, "the zebra lives in Paris", 1.0).await;
    let mut replacement = new_fact(&f.scope, subject, "the zebra lives in Rome", 1.0);
    replacement.valid_from = date(2022);
    let op = FactOp::Supersede {
        old_fact_id: old,
        new: replacement,
    };
    let new = f.apply_in(&f.ctx, &f.scope, Trust::User, vec![op]).await[0];
    f.store
        .insert_episode(&f.ctx, episode(&f.scope, Trust::User, "a zebra is pending"))
        .await
        .expect("insert episode");

    let scopes = [f.scope.clone()];
    let candidates = f
        .retriever
        .candidates(&f.ctx, &scopes, QUERY, 10)
        .await
        .expect("candidates");
    assert_eq!(ids(&candidates), BTreeSet::from([new]));

    let recalled = f.recall(QUERY).await.facts;
    assert_eq!(candidates.len(), recalled.len());
    assert_eq!(candidates[0].score, recalled[0].score);
}

// ---- Limits ----------------------------------------------------------------------------------

#[tokio::test]
async fn invalid_requests_are_rejected() {
    let f = fixture().await;
    let scopes = [f.scope.clone()];

    for limit in [-1, 0, 101] {
        let mut request = f.request(QUERY);
        request.limit = limit;
        assert!(is_invalid(f.retriever.recall(&f.ctx, request).await));
        let candidates = f.retriever.candidates(&f.ctx, &scopes, QUERY, limit).await;
        assert!(is_invalid(candidates));
    }
    for blank in ["", "  \n\t"] {
        assert!(is_invalid(
            f.retriever.recall(&f.ctx, f.request(blank)).await
        ));
        let candidates = f.retriever.candidates(&f.ctx, &scopes, blank, 10).await;
        assert!(is_invalid(candidates));
    }

    let mut no_scopes = f.request(QUERY);
    no_scopes.scopes.clear();
    assert!(is_invalid(f.retriever.recall(&f.ctx, no_scopes).await));
    assert!(is_invalid(
        f.retriever.candidates(&f.ctx, &[], QUERY, 10).await
    ));

    let many: Vec<Scope> = (0..51).map(|i| Scope::user(format!("u{i}"))).collect();
    let mut too_many = f.request(QUERY);
    too_many.scopes = many.clone();
    assert!(is_invalid(f.retriever.recall(&f.ctx, too_many).await));
    assert!(is_invalid(
        f.retriever.candidates(&f.ctx, &many, QUERY, 10).await
    ));

    let mut fifty = f.request(QUERY);
    fifty.scopes = many[..50].to_vec();
    assert!(f.retriever.recall(&f.ctx, fifty).await.is_ok());
}

#[tokio::test]
async fn limit_defaults_to_10_and_truncates() {
    let f = fixture().await;
    let subject = f.entity("a").await;
    let ops = (0..120)
        .map(|i| FactOp::Add(new_fact(&f.scope, subject, "a zebra", i as f32 / 10.0)))
        .collect();
    f.apply_in(&f.ctx, &f.scope, Trust::User, ops).await;

    let body = json!({"scopes": [f.scope.to_string()], "query": QUERY});
    let request: RecallRequest = serde_json::from_value(body).expect("request");
    let response = f.retriever.recall(&f.ctx, request).await.expect("recall");
    assert_eq!(response.facts.len(), 10);

    let scopes = [f.scope.clone()];
    // Vector and text both pick the oldest 50, graph expansion the newest 50.
    for (limit, expected) in [(1, 1), (3, 3), (100, 100)] {
        let found = f
            .retriever
            .candidates(&f.ctx, &scopes, QUERY, limit)
            .await
            .expect("candidates");
        assert_eq!(found.len(), expected, "limit {limit}");
    }
}

#[tokio::test]
async fn pending_episodes_respect_the_limit() {
    let f = fixture().await;
    for _ in 0..3 {
        f.store
            .insert_episode(&f.ctx, episode(&f.scope, Trust::User, "a zebra is pending"))
            .await
            .expect("insert episode");
    }
    let mut request = f.request(QUERY);
    request.limit = 2;
    let response = f.retriever.recall(&f.ctx, request).await.expect("recall");
    assert_eq!(response.pending_episodes.len(), 2);
}
