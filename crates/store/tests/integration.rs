//! Real Postgres tests for `PgStore`. One container per test binary; each test clones a migrated
//! template into its own database, so tests run in parallel without sharing jobs or rows.
//!
//! Tests log in as the superuser on purpose: tenant isolation must hold through the role switch
//! alone, not because the login role lacks privileges.

#![cfg(feature = "integration")]

use std::collections::BTreeSet;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use memoryos_core::{
    Error, FactFilter, FactOp, FactStatus, MemoryStore, NewEntity, NewEpisode, NewFact,
    RequestContext, Scope, Trust, EMBEDDING_DIMS,
};
use memoryos_store::PgStore;
use pgvector::Vector;
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{GenericImage, ImageExt};
use uuid::Uuid;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const TEMPLATE_DB: &str = "memoryos_template";

// ---- Harness ---------------------------------------------------------------------------------

/// Starts the shared container once and returns its server URL, without a database name.
fn server_url() -> &'static str {
    static URL: OnceLock<String> = OnceLock::new();
    URL.get_or_init(|| {
        std::thread::spawn(|| {
            tokio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(start_server())
        })
        .join()
        .expect("container thread")
    })
}

async fn start_server() -> String {
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

struct Db {
    store: Arc<PgStore>,
    admin: PgPool,
    url: String,
}

async fn db() -> Db {
    db_with_pool(16).await
}

async fn db_with_pool(max_connections: u32) -> Db {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let server = server_url();
    let name = format!("t{}", NEXT.fetch_add(1, Ordering::SeqCst));
    let bootstrap = connect_with_retry(&format!("{server}/postgres")).await;
    sqlx::query(&format!("CREATE DATABASE {name} TEMPLATE {TEMPLATE_DB}"))
        .execute(&bootstrap)
        .await
        .expect("clone template");
    bootstrap.close().await;
    let url = format!("{server}/{name}");
    Db {
        store: Arc::new(
            PgStore::connect(&url, max_connections)
                .await
                .expect("connect"),
        ),
        admin: connect_with_retry(&url).await,
        url,
    }
}

fn ctx(tenant: &str) -> RequestContext {
    RequestContext {
        tenant: tenant.into(),
        actor: "tester".into(),
    }
}

fn user(id: &str) -> Scope {
    Scope::user(id)
}

fn agent_scope() -> Scope {
    Scope::agent("a1")
}

fn date(year: i32, month: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0)
        .single()
        .expect("valid date")
}

/// Unit vector on axis `i`. Different axes have cosine distance 1.
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; EMBEDDING_DIMS];
    v[i] = 1.0;
    v
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

fn new_fact(scope: &Scope, subject_id: Uuid, statement: &str, from: DateTime<Utc>) -> NewFact {
    NewFact {
        scope: scope.clone(),
        subject_id,
        predicate: "says".into(),
        object_id: None,
        statement: statement.into(),
        embedding: axis(0),
        trust: Trust::User,
        status: FactStatus::Active,
        valid_from: from,
        valid_to: None,
    }
}

async fn entity(db: &Db, c: &RequestContext, scope: &Scope, name: &str) -> Res<Uuid> {
    let created = db
        .store
        .upsert_entity(
            c,
            NewEntity {
                scope: scope.clone(),
                kind: "thing".into(),
                name: name.into(),
                aliases: vec![],
                embedding: axis(1),
            },
        )
        .await?;
    Ok(created.id)
}

async fn new_episode(db: &Db, c: &RequestContext, scope: &Scope, trust: Trust) -> Res<Uuid> {
    Ok(db
        .store
        .insert_episode(c, episode(scope, trust, "source text"))
        .await?
        .id)
}

/// Inserts one fact through its own episode. Returns (episode id, fact id).
async fn add_fact(db: &Db, c: &RequestContext, fact: NewFact) -> Res<(Uuid, Uuid)> {
    let episode_id = new_episode(db, c, &fact.scope, fact.trust).await?;
    let ids = db
        .store
        .apply_extraction(c, episode_id, vec![FactOp::Add(fact)])
        .await?;
    Ok((episode_id, ids[0]))
}

async fn supersede(db: &Db, c: &RequestContext, old_fact_id: Uuid, new: NewFact) -> Res<Uuid> {
    let episode_id = new_episode(db, c, &new.scope, new.trust).await?;
    let ids = db
        .store
        .apply_extraction(c, episode_id, vec![FactOp::Supersede { old_fact_id, new }])
        .await?;
    Ok(ids[0])
}

async fn count(db: &Db, sql: &str) -> Res<i64> {
    Ok(sqlx::query_scalar(sql).fetch_one(&db.admin).await?)
}

fn at(as_of: DateTime<Utc>) -> FactFilter {
    FactFilter {
        as_of: Some(as_of),
        known_at: None,
    }
}

fn is_not_found<T>(r: Result<T, Error>) -> bool {
    matches!(r, Err(Error::NotFound(_)))
}

fn is_invalid<T>(r: Result<T, Error>) -> bool {
    matches!(r, Err(Error::InvalidInput(_)))
}

// ---- Connectivity and migrations -------------------------------------------------------------

#[tokio::test]
async fn ping_and_repeat_migrate_succeed() -> TestResult {
    let db = db().await;
    db.store.ping().await?;
    db.store.migrate().await?;
    Ok(())
}

#[tokio::test]
async fn role_and_tenant_do_not_leak_to_the_pooled_connection() -> TestResult {
    // One connection: every call below reuses it. memoryos_app has no privileges on jobs, so
    // claim_job fails if the role survives a commit or a rollback.
    let db = db_with_pool(1).await;
    let c = ctx("t1");
    db.store
        .put_block(&c, "a", "persona", "v1", 100, None)
        .await?;
    assert!(db.store.claim_job("w", 30).await?.is_none());

    let conflict = db
        .store
        .put_block(&c, "a", "persona", "x", 100, Some(9))
        .await;
    assert!(matches!(conflict, Err(Error::VersionConflict { .. })));
    assert!(db.store.claim_job("w", 30).await?.is_none());
    Ok(())
}

// ---- Episodes --------------------------------------------------------------------------------

#[tokio::test]
async fn insert_episode_is_idempotent_and_enqueues_one_job() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let mut new = episode(&scope, Trust::User, "hello");
    new.idempotency_key = Some("key-1".into());

    let first = db.store.insert_episode(&c, new.clone()).await?;
    let second = db.store.insert_episode(&c, new.clone()).await?;
    assert_eq!(first.id, second.id);
    assert_eq!(count(&db, "SELECT count(*) FROM episodes").await?, 1);
    assert_eq!(count(&db, "SELECT count(*) FROM jobs").await?, 1);

    let job = db.store.claim_job("w", 30).await?.expect("job");
    assert_eq!(job.kind, "extract");
    assert_eq!(job.tenant, "t1");
    assert_eq!(job.payload, json!({ "episode_id": first.id.to_string() }));

    // The key is per tenant.
    let other = db.store.insert_episode(&ctx("t2"), new).await?;
    assert_ne!(other.id, first.id);
    assert_eq!(count(&db, "SELECT count(*) FROM jobs").await?, 2);
    Ok(())
}

#[tokio::test]
async fn concurrent_inserts_with_one_key_create_one_episode() -> TestResult {
    let db = db().await;
    let mut new = episode(&user("u1"), Trust::User, "hello");
    new.idempotency_key = Some("key-1".into());
    let calls = (0..8).map(|_| {
        let store = db.store.clone();
        let new = new.clone();
        tokio::spawn(async move { store.insert_episode(&ctx("t1"), new).await })
    });
    let ids: BTreeSet<Uuid> = futures::future::join_all(calls)
        .await
        .into_iter()
        .map(|r| r.expect("task").expect("insert").id)
        .collect();
    assert_eq!(ids.len(), 1);
    assert_eq!(count(&db, "SELECT count(*) FROM jobs").await?, 1);
    Ok(())
}

#[tokio::test]
async fn null_metadata_and_missing_occurred_at_get_defaults() -> TestResult {
    let db = db().await;
    let mut new = episode(&user("u1"), Trust::User, "hello");
    new.metadata = serde_json::Value::Null;
    let stored = db.store.insert_episode(&ctx("t1"), new).await?;
    assert_eq!(stored.metadata, json!({}));
    assert_eq!(stored.occurred_at, stored.created_at);
    Ok(())
}

/// The API limit on entity names has to fit the unique index in the worst case. Distinct code points
/// keep PostgreSQL from compressing the key.
#[tokio::test]
async fn an_entity_name_at_the_api_limit_fits_the_unique_index() -> TestResult {
    let db = db().await;
    let name: String = (0..512_u32)
        .map(|i| char::from_u32(0x10000 + i).expect("code point"))
        .collect();
    let created = db
        .store
        .upsert_entity(
            &ctx("t1"),
            NewEntity {
                scope: user("u1"),
                kind: "k".repeat(64),
                name,
                aliases: vec![],
                embedding: axis(1),
            },
        )
        .await?;
    assert_eq!(created.name.chars().count(), 512);
    Ok(())
}

#[tokio::test]
async fn subject_query_returns_the_latest_200_oldest_first() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let ops = (0..201)
        .map(|i| FactOp::Add(new_fact(&scope, subject, &format!("f{i}"), date(2020, 1))))
        .collect();
    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;
    let ids = db.store.apply_extraction(&c, episode_id, ops).await?;

    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    let got: Vec<Uuid> = current.iter().map(|fact| fact.id).collect();
    assert_eq!(got, ids[1..]);
    Ok(())
}

#[tokio::test]
async fn current_fact_applies_the_scope_and_the_current_filter() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let (_, id) = add_fact(&db, &c, new_fact(&scope, subject, "f", date(2020, 1))).await?;
    assert!(db.store.current_fact(&c, &scope, id).await?.is_some());
    assert!(db.store.current_fact(&c, &user("u2"), id).await?.is_none());
    assert!(db
        .store
        .current_fact(&ctx("t2"), &scope, id)
        .await?
        .is_none());
    db.store.invalidate_fact(&c, id, "wrong").await?;
    assert!(db.store.current_fact(&c, &scope, id).await?.is_none());
    Ok(())
}

#[tokio::test]
async fn invalidating_a_fact_reopens_what_it_closed() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let (_, first) = add_fact(&db, &c, new_fact(&scope, subject, "first", date(2020, 1))).await?;
    let second = supersede(
        &db,
        &c,
        first,
        new_fact(&scope, subject, "second", date(2021, 1)),
    )
    .await?;

    db.store.invalidate_fact(&c, second, "wrong").await?;
    let reopened = db.store.get_fact(&c, first).await?;
    assert_eq!((reopened.valid_to, reopened.superseded_by), (None, None));
    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, first);
    // A repeat does nothing, even though the fact was reopened and may have moved on.
    let third = supersede(
        &db,
        &c,
        first,
        new_fact(&scope, subject, "third", date(2022, 1)),
    )
    .await?;
    db.store.invalidate_fact(&c, second, "again").await?;
    assert_eq!(
        db.store.get_fact(&c, first).await?.superseded_by,
        Some(third)
    );
    Ok(())
}

#[tokio::test]
async fn invalidating_the_middle_of_a_chain_links_around_it_and_moves_pending_facts() -> TestResult
{
    let db = db().await;
    let c = ctx("t1");
    let scope = agent_scope();
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let (_, first) = add_fact(&db, &c, new_fact(&scope, subject, "first", date(2020, 1))).await?;
    let middle = supersede(
        &db,
        &c,
        first,
        new_fact(&scope, subject, "middle", date(2021, 1)),
    )
    .await?;
    let last = supersede(
        &db,
        &c,
        middle,
        new_fact(&scope, subject, "last", date(2022, 1)),
    )
    .await?;

    db.store.invalidate_fact(&c, middle, "wrong").await?;
    let linked = db.store.get_fact(&c, first).await?;
    assert_eq!(linked.superseded_by, Some(last));
    assert_eq!(linked.valid_to, Some(date(2022, 1)));

    // Pending on the head, which is then retracted: approval closes the reopened predecessor.
    let mut held = new_fact(&scope, subject, "held", date(2023, 1));
    held.trust = Trust::External;
    held.status = FactStatus::Quarantined;
    let pending = supersede(&db, &c, last, held).await?;
    db.store.invalidate_fact(&c, last, "wrong").await?;
    assert_eq!(
        db.store.get_fact(&c, pending).await?.pending_supersede_of,
        Some(first)
    );
    db.store.approve_fact(&c, pending).await?;
    assert_eq!(
        db.store.get_fact(&c, first).await?.superseded_by,
        Some(pending)
    );
    Ok(())
}

/// A fact in an agent scope, so the source rule does not already hold it.
#[tokio::test]
async fn an_add_under_a_higher_trust_fact_of_the_same_subject_and_predicate_is_held() -> TestResult
{
    let db = db().await;
    let c = ctx("t1");
    let scope = agent_scope();
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let (_, user_fact) = add_fact(
        &db,
        &c,
        new_fact(&scope, subject, "lives in Paris", date(2020, 1)),
    )
    .await?;
    let agent_fact = |statement: &str, predicate: &str| {
        let mut fact = new_fact(&scope, subject, statement, date(2021, 1));
        fact.trust = Trust::Agent;
        fact.predicate = predicate.into();
        fact
    };
    let status = |id: Uuid| {
        let db = &db;
        let c = &c;
        async move { Res::Ok(db.store.get_fact(c, id).await?) }
    };

    let (_, held) = add_fact(&db, &c, agent_fact("lives in Rome", "says")).await?;
    let held = status(held).await?;
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!(held.pending_supersede_of, None);
    assert_eq!(status(user_fact).await?.valid_to, None);

    let (_, other) = add_fact(&db, &c, agent_fact("likes tea", "likes")).await?;
    assert_eq!(status(other).await?.status, FactStatus::Active);
    let (_, equal) = add_fact(&db, &c, agent_fact("likes coffee", "likes")).await?;
    assert_eq!(status(equal).await?.status, FactStatus::Active);

    // Approval activates it and closes nothing, since it never declared a contradiction.
    assert_eq!(
        db.store.approve_fact(&c, held.id).await?.status,
        FactStatus::Active
    );
    assert_eq!(status(user_fact).await?.valid_to, None);

    // A retracted fact outranks nothing.
    db.store.invalidate_fact(&c, user_fact, "wrong").await?;
    let (_, after) = add_fact(&db, &c, agent_fact("lives in Oslo", "says")).await?;
    assert_eq!(status(after).await?.status, FactStatus::Active);
    Ok(())
}

#[tokio::test]
async fn a_block_keeps_and_returns_its_latest_100_versions() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let mut block = db
        .store
        .put_block(&c, "a1", "notes", "v1", 100, None)
        .await?;
    for i in 2..=105 {
        block = db
            .store
            .put_block(
                &c,
                "a1",
                "notes",
                &format!("v{i}"),
                100,
                Some(block.version),
            )
            .await?;
    }
    assert_eq!(
        count(&db, "SELECT count(*) FROM block_versions").await?,
        100
    );
    let versions = db.store.block_versions(&c, "a1", "notes").await?;
    assert_eq!(versions.len(), 100);
    assert_eq!(versions[0].version, 6);
    assert_eq!(versions[99].version, 105);

    // Rows from before pruning existed must not widen a read.
    sqlx::query(
        "INSERT INTO block_versions (block_id, tenant, version, content, edited_by) \
         SELECT $1, 't1', v, 'old', 'test' FROM generate_series(1, 5) AS v",
    )
    .bind(block.id)
    .execute(&db.admin)
    .await?;
    let versions = db.store.block_versions(&c, "a1", "notes").await?;
    assert_eq!(versions.len(), 100);
    assert_eq!(versions[0].version, 6);
    Ok(())
}

/// A[10,20), B[20,), and a fact backdated to 5 becomes [5,10), whichever of the two it names.
#[tokio::test]
async fn a_backdated_fact_ends_at_the_earliest_later_fact_of_the_chain() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    for name_b_as_target in [false, true] {
        let scope = user(if name_b_as_target { "u1" } else { "u2" });
        let subject = entity(&db, &c, &scope, "Ada").await?;
        let (_, a) = add_fact(&db, &c, new_fact(&scope, subject, "a", date(2010, 1))).await?;
        let b = supersede(&db, &c, a, new_fact(&scope, subject, "b", date(2020, 1))).await?;
        let target = if name_b_as_target { b } else { a };
        let old = supersede(
            &db,
            &c,
            target,
            new_fact(&scope, subject, "old", date(2005, 1)),
        )
        .await?;

        let got = db.store.get_fact(&c, old).await?;
        assert_eq!(got.valid_to, Some(date(2010, 1)));
        assert_eq!(got.superseded_by, Some(a));
        assert_eq!(
            db.store.get_fact(&c, a).await?.valid_to,
            Some(date(2020, 1))
        );
        assert_eq!(db.store.get_fact(&c, b).await?.valid_to, None);

        for (when, expected) in [(2007, old), (2015, a), (2025, b)] {
            let found = db
                .store
                .search_facts_vector(
                    &c,
                    std::slice::from_ref(&scope),
                    &axis(0),
                    &at(date(when, 1)),
                    10,
                )
                .await?;
            let ids: Vec<Uuid> = found.iter().map(|hit| hit.fact.id).collect();
            assert_eq!(ids, [expected], "as of {when}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn an_alias_goes_when_the_last_episode_that_earned_it_is_purged() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Robert").await?;
    let (first, _) = add_fact(&db, &c, new_fact(&scope, subject, "kept", date(2020, 1))).await?;
    let second = new_episode(&db, &c, &scope, Trust::User).await?;
    let third = new_episode(&db, &c, &scope, Trust::User).await?;
    db.store
        .add_entity_alias(&c, subject, "Bob", Some(second))
        .await?;
    db.store
        .add_entity_alias(&c, subject, "Bob", Some(third))
        .await?;
    db.store.add_entity_alias(&c, subject, "Rob", None).await?;

    db.store.purge_episode(&c, second).await?;
    let names = db.store.get_entity(&c, subject).await?.aliases;
    assert_eq!(names, ["Bob", "Rob"], "another episode also earned it");

    db.store.purge_episode(&c, third).await?;
    let names = db.store.get_entity(&c, subject).await?.aliases;
    assert_eq!(names, ["Rob"], "an alias with no recorded source stays");
    assert!(db
        .store
        .find_entity(&c, &scope, "thing", "bob")
        .await?
        .is_none());
    db.store.get_episode(&c, first).await?;
    Ok(())
}

#[tokio::test]
async fn claim_says_whether_this_call_created_the_episode() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let mut new = episode(&user("u1"), Trust::Agent, "source");
    new.idempotency_key = Some("k".into());
    let (first, created) = db.store.claim_unqueued_episode(&c, new.clone()).await?;
    assert!(created);
    let (second, created) = db.store.claim_unqueued_episode(&c, new).await?;
    assert!(!created);
    assert_eq!(first.id, second.id);
    Ok(())
}

#[tokio::test]
async fn discard_leaves_a_processed_episode_and_its_facts() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let (episode_id, fact_id) =
        add_fact(&db, &c, new_fact(&scope, subject, "kept", date(2025, 1))).await?;
    assert!(!db.store.discard_unprocessed_episode(&c, episode_id).await?);
    db.store.get_episode(&c, episode_id).await?;
    db.store.get_fact(&c, fact_id).await?;
    Ok(())
}

/// Whichever gets the row first, the other sees that: no acknowledged fact can lose its episode.
#[tokio::test(flavor = "multi_thread")]
async fn discard_and_apply_racing_leave_exactly_one_outcome() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    for round in 0..20 {
        let (claimed, _) = db
            .store
            .claim_unqueued_episode(&c, episode(&scope, Trust::User, "source"))
            .await?;
        let op = FactOp::Add(new_fact(&scope, subject, "raced", date(2025, 1)));
        let (discarded, applied) = tokio::join!(
            db.store.discard_unprocessed_episode(&c, claimed.id),
            db.store.apply_extraction(&c, claimed.id, vec![op]),
        );
        if discarded? {
            assert!(is_not_found(applied), "round {round}");
            assert!(is_not_found(db.store.get_episode(&c, claimed.id).await));
        } else {
            let ids = applied?;
            db.store.get_fact(&c, ids[0]).await?;
            db.store.get_episode(&c, claimed.id).await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn nul_byte_in_content_is_invalid_input_not_storage_error() -> TestResult {
    let db = db().await;
    let bad = episode(&user("u1"), Trust::User, "a\0b");
    assert!(is_invalid(db.store.insert_episode(&ctx("t1"), bad).await));
    Ok(())
}

// ---- apply_extraction ------------------------------------------------------------------------

#[tokio::test]
async fn apply_extraction_retry_is_a_no_op() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;
    let ops = || vec![FactOp::Add(new_fact(&scope, subject, "one", date(2020, 1)))];

    let first = db.store.apply_extraction(&c, episode_id, ops()).await?;
    assert_eq!(first.len(), 1);
    let second = db.store.apply_extraction(&c, episode_id, ops()).await?;
    assert!(second.is_empty());
    assert_eq!(count(&db, "SELECT count(*) FROM facts").await?, 1);
    assert!(db
        .store
        .get_episode(&c, episode_id)
        .await?
        .processed_at
        .is_some());
    Ok(())
}

#[tokio::test]
async fn concurrent_apply_extraction_applies_exactly_once() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;

    let calls = (0..8).map(|_| {
        let store = db.store.clone();
        let c = c.clone();
        let ops = vec![
            FactOp::Add(new_fact(&scope, subject, "added", date(2021, 1))),
            FactOp::Supersede {
                old_fact_id: old,
                new: new_fact(&scope, subject, "newer", date(2021, 2)),
            },
        ];
        tokio::spawn(async move { store.apply_extraction(&c, episode_id, ops).await })
    });
    let results = futures::future::join_all(calls).await;
    let mut applied = 0;
    for result in results {
        if !result??.is_empty() {
            applied += 1;
        }
    }
    assert_eq!(applied, 1);
    assert_eq!(count(&db, "SELECT count(*) FROM facts").await?, 3);
    Ok(())
}

#[tokio::test]
async fn failed_apply_extraction_rolls_back_and_can_retry() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;

    let bad = db
        .store
        .apply_extraction(
            &c,
            episode_id,
            vec![
                FactOp::Add(new_fact(&scope, subject, "good", date(2020, 1))),
                FactOp::Corroborate {
                    fact_id: Uuid::now_v7(),
                },
            ],
        )
        .await;
    assert!(is_not_found(bad));
    assert_eq!(count(&db, "SELECT count(*) FROM facts").await?, 0);
    assert!(db
        .store
        .get_episode(&c, episode_id)
        .await?
        .processed_at
        .is_none());

    let retry = db
        .store
        .apply_extraction(
            &c,
            episode_id,
            vec![FactOp::Add(new_fact(
                &scope,
                subject,
                "good",
                date(2020, 1),
            ))],
        )
        .await?;
    assert_eq!(retry.len(), 1);
    Ok(())
}

#[tokio::test]
async fn store_refuses_facts_that_break_trust_or_scope_rules() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let other_scope = user("u2");
    let other_subject = entity(&db, &c, &other_scope, "bob").await?;
    let good = |statement| new_fact(&scope, subject, statement, date(2020, 1));

    // An external episode can never produce an active fact, whatever the caller computed.
    let external = new_episode(&db, &c, &scope, Trust::External).await?;
    let mut poisoned = good("poison");
    poisoned.trust = Trust::External;
    assert!(is_invalid(
        db.store
            .apply_extraction(&c, external, vec![FactOp::Add(poisoned)])
            .await
    ));

    let e = new_episode(&db, &c, &scope, Trust::User).await?;
    let mut raised = good("raised");
    raised.trust = Trust::System;
    let mut wrong_scope = new_fact(&other_scope, other_subject, "x", date(2020, 1));
    wrong_scope.trust = Trust::User;
    let foreign_entity = new_fact(&scope, other_subject, "x", date(2020, 1));
    let mut short_embedding = good("short");
    short_embedding.embedding = vec![1.0; 3];
    let mut reversed = good("reversed");
    reversed.valid_to = Some(date(2019, 1));
    for bad in [
        raised,
        wrong_scope,
        foreign_entity,
        short_embedding,
        reversed,
    ] {
        assert!(is_invalid(
            db.store
                .apply_extraction(&c, e, vec![FactOp::Add(bad)])
                .await
        ));
    }
    assert_eq!(count(&db, "SELECT count(*) FROM facts").await?, 0);
    Ok(())
}

#[tokio::test]
async fn agent_facts_can_go_active_only_in_agent_scopes() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    for scope in [user("u1"), Scope::shared("team")] {
        let subject = entity(&db, &c, &scope, "ada").await?;
        let episode_id = new_episode(&db, &c, &scope, Trust::Agent).await?;
        let mut planted = new_fact(&scope, subject, "planted", date(2020, 1));
        planted.trust = Trust::Agent;
        assert!(is_invalid(
            db.store
                .apply_extraction(&c, episode_id, vec![FactOp::Add(planted.clone())])
                .await
        ));
        // The same statement through quarantine is accepted and stays out of recall.
        planted.status = FactStatus::Quarantined;
        let held = add_fact(&db, &c, planted).await?.1;
        assert_eq!(
            db.store.get_fact(&c, held).await?.status,
            FactStatus::Quarantined
        );
    }
    let scope = Scope::agent("bot");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let mut own = new_fact(&scope, subject, "own note", date(2020, 1));
    own.trust = Trust::Agent;
    let active = add_fact(&db, &c, own).await?.1;
    assert_eq!(
        db.store.get_fact(&c, active).await?.status,
        FactStatus::Active
    );
    Ok(())
}

#[tokio::test]
async fn corroborate_links_a_second_source_without_raising_trust() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = Scope::agent("bot");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let mut low = new_fact(&scope, subject, "fact", date(2020, 1));
    low.trust = Trust::Agent;
    let (first_source, fact_id) = add_fact(&db, &c, low).await?;
    let second_source = new_episode(&db, &c, &scope, Trust::System).await?;

    let inserted = db
        .store
        .apply_extraction(&c, second_source, vec![FactOp::Corroborate { fact_id }])
        .await?;
    assert!(inserted.is_empty());

    let fact = db.store.get_fact(&c, fact_id).await?;
    assert_eq!(fact.trust, Trust::Agent);
    let mut expected = vec![first_source, second_source];
    expected.sort();
    assert_eq!(fact.source_episode_ids, expected);
    Ok(())
}

// ---- Supersession and time -------------------------------------------------------------------

#[tokio::test]
async fn supersede_closes_the_old_fact_without_deleting_it() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "paris", date(2020, 1))).await?;
    let new = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "london", date(2022, 1)),
    )
    .await?;

    let closed = db.store.get_fact(&c, old).await?;
    assert_eq!(closed.valid_to, Some(date(2022, 1)));
    assert_eq!(closed.superseded_by, Some(new));

    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, new);

    let history = db.store.fact_history(&c, new).await?;
    assert_eq!(
        history.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![old, new]
    );
    assert_eq!(db.store.fact_history(&c, old).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn second_supersede_of_a_closed_fact_extends_the_chain() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, first) = add_fact(&db, &c, new_fact(&scope, subject, "a", date(2020, 1))).await?;
    let second = supersede(
        &db,
        &c,
        first,
        new_fact(&scope, subject, "b", date(2021, 1)),
    )
    .await?;
    // A job whose reconciliation read `first` before it closed.
    let third = supersede(
        &db,
        &c,
        first,
        new_fact(&scope, subject, "c", date(2022, 1)),
    )
    .await?;

    let first_after = db.store.get_fact(&c, first).await?;
    assert_eq!(first_after.valid_to, Some(date(2021, 1)));
    assert_eq!(first_after.superseded_by, Some(second));
    let second_after = db.store.get_fact(&c, second).await?;
    assert_eq!(second_after.valid_to, Some(date(2022, 1)));
    assert_eq!(second_after.superseded_by, Some(third));
    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert_eq!(
        current.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![third]
    );
    Ok(())
}

#[tokio::test]
async fn out_of_order_fact_is_stored_as_closed_history() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, current) =
        add_fact(&db, &c, new_fact(&scope, subject, "london", date(2022, 1))).await?;
    let late = supersede(
        &db,
        &c,
        current,
        new_fact(&scope, subject, "paris", date(2020, 1)),
    )
    .await?;

    let stored_late = db.store.get_fact(&c, late).await?;
    assert_eq!(stored_late.valid_to, Some(date(2022, 1)));
    assert_eq!(stored_late.status, FactStatus::Active);

    let still_current = db.store.get_fact(&c, current).await?;
    assert_eq!(still_current.valid_to, None);
    assert_eq!(still_current.superseded_by, None);

    let now = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert_eq!(now.iter().map(|f| f.id).collect::<Vec<_>>(), vec![current]);
    let then = db
        .store
        .search_facts_text(
            &c,
            std::slice::from_ref(&scope),
            "paris",
            &at(date(2021, 1)),
            10,
        )
        .await?;
    assert_eq!(then.len(), 1);
    assert_eq!(then[0].fact.id, late);
    assert_eq!(
        db.store
            .fact_history(&c, current)
            .await?
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![late, current]
    );
    Ok(())
}

#[tokio::test]
async fn as_of_and_known_at_follow_the_spec_filter() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "paris", date(2020, 1))).await?;
    let new = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "london", date(2022, 1)),
    )
    .await?;
    let ids = |facts: Vec<memoryos_core::ScoredFact>| {
        facts.into_iter().map(|s| s.fact.id).collect::<Vec<_>>()
    };

    // Valid time: each version in its own window, boundary excluded from the old one.
    for (when, expected) in [
        (date(2019, 1), vec![]),
        (date(2021, 1), vec![old]),
        (date(2022, 1), vec![new]),
        (date(2023, 1), vec![new]),
    ] {
        let found = db
            .store
            .search_facts_vector(&c, &scopes, &axis(0), &at(when), 10)
            .await?;
        assert_eq!(ids(found), expected, "as_of {when}");
    }

    // Knowledge time: before the new fact was recorded it is not visible at any valid time.
    let recorded = db.store.get_fact(&c, new).await?.recorded_at;
    let before = FactFilter {
        as_of: Some(date(2023, 1)),
        known_at: Some(recorded - chrono::Duration::seconds(1)),
    };
    assert!(db
        .store
        .search_facts_vector(&c, &scopes, &axis(0), &before, 10)
        .await?
        .is_empty());
    let at_record = FactFilter {
        as_of: Some(date(2023, 1)),
        known_at: Some(recorded),
    };
    assert_eq!(
        ids(db
            .store
            .search_facts_text(&c, &scopes, "london", &at_record, 10)
            .await?),
        vec![new]
    );

    // Retraction: hidden now, still visible to a read from before the retraction.
    let invalidated = db.store.invalidate_fact(&c, new, "wrong").await?;
    let retracted_at = invalidated.invalidated_at.expect("retraction time");
    assert!(!db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?
        .iter()
        .any(|fact| fact.id == new));
    let before_retraction = FactFilter {
        as_of: None,
        known_at: Some(retracted_at - chrono::Duration::microseconds(1)),
    };
    assert_eq!(
        ids(db
            .store
            .search_facts_text(&c, &scopes, "london", &before_retraction, 10)
            .await?),
        vec![new]
    );
    let audit = count(
        &db,
        "SELECT count(*) FROM audit_events WHERE action = 'fact.invalidate'",
    )
    .await?;
    assert_eq!(audit, 1);
    Ok(())
}

// ---- Quarantine ------------------------------------------------------------------------------

#[tokio::test]
async fn quarantined_supersede_leaves_old_open_until_approved() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "paris", date(2020, 1))).await?;

    let mut proposal = new_fact(&scope, subject, "london", date(2022, 1));
    proposal.trust = Trust::External;
    proposal.status = FactStatus::Quarantined;
    let new = supersede(&db, &c, old, proposal).await?;

    let open = db.store.get_fact(&c, old).await?;
    assert_eq!(open.valid_to, None);
    assert_eq!(open.superseded_by, None);
    let pending = db.store.get_fact(&c, new).await?.pending_supersede_of;
    assert_eq!(pending, Some(old));
    let listed = db.store.list_quarantined(&c, &scopes, 10).await?;
    assert_eq!(listed[0].pending_supersede_of, Some(old));

    // Quarantined facts are absent from every read path.
    let no_filter = FactFilter::default();
    assert!(db
        .store
        .search_facts_text(&c, &scopes, "london", &no_filter, 10)
        .await?
        .is_empty());
    assert_eq!(
        db.store
            .search_facts_vector(&c, &scopes, &axis(0), &no_filter, 10)
            .await?
            .len(),
        1
    );
    assert_eq!(
        db.store
            .neighbors(&c, &scopes, subject, 1, &no_filter, 5000)
            .await?
            .edges
            .len(),
        1
    );
    let listed = db.store.list_quarantined(&c, &scopes, 10).await?;
    assert_eq!(listed.iter().map(|f| f.id).collect::<Vec<_>>(), vec![new]);

    let approved = db.store.approve_fact(&c, new).await?;
    assert_eq!(approved.status, FactStatus::Active);
    assert_eq!(approved.pending_supersede_of, None);
    let closed = db.store.get_fact(&c, old).await?;
    assert_eq!(closed.valid_to, Some(date(2022, 1)));
    assert_eq!(closed.superseded_by, Some(new));
    assert!(db.store.list_quarantined(&c, &scopes, 10).await?.is_empty());
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM audit_events WHERE action = 'fact.approve'"
        )
        .await?,
        1
    );

    // A second approval is refused and writes nothing.
    assert!(is_invalid(db.store.approve_fact(&c, new).await));
    Ok(())
}

#[tokio::test]
async fn approving_an_out_of_order_quarantined_fact_keeps_the_old_one_current() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, current) =
        add_fact(&db, &c, new_fact(&scope, subject, "london", date(2022, 1))).await?;
    let mut proposal = new_fact(&scope, subject, "paris", date(2020, 1));
    proposal.trust = Trust::External;
    proposal.status = FactStatus::Quarantined;
    let late = supersede(&db, &c, current, proposal).await?;

    let approved = db.store.approve_fact(&c, late).await?;
    assert_eq!(approved.valid_to, Some(date(2022, 1)));
    assert_eq!(db.store.get_fact(&c, current).await?.valid_to, None);
    Ok(())
}

#[tokio::test]
async fn reject_hides_the_fact_and_leaves_the_old_one_open() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "paris", date(2020, 1))).await?;
    let mut proposal = new_fact(&scope, subject, "london", date(2022, 1));
    proposal.trust = Trust::External;
    proposal.status = FactStatus::Quarantined;
    let new = supersede(&db, &c, old, proposal).await?;

    let rejected = db.store.reject_fact(&c, new).await?;
    assert_eq!(rejected.status, FactStatus::Rejected);
    assert_eq!(db.store.get_fact(&c, old).await?.valid_to, None);
    assert!(db
        .store
        .list_quarantined(&c, std::slice::from_ref(&scope), 10)
        .await?
        .is_empty());
    assert!(is_invalid(db.store.approve_fact(&c, new).await));
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM audit_events WHERE action = 'fact.reject'"
        )
        .await?,
        1
    );
    Ok(())
}

// ---- Search ----------------------------------------------------------------------------------

/// `axis(0)` plus `noise` times a deterministic pseudo-random unit vector.
fn noisy(seed: u64, noise: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1);
    let mut random: Vec<f32> = (0..EMBEDDING_DIMS)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let norm = random.iter().map(|x| x * x).sum::<f32>().sqrt();
    random.iter_mut().for_each(|x| *x *= noise / norm);
    random[0] += 1.0;
    random
}

/// `user:big` gets 300 facts near `axis(0)` and `user:rare` gets 12 slightly farther ones. They are
/// interleaved and more than the requested limit, because HNSW can leave an individual outlier
/// unreachable no matter how far a scan iterates. That is index recall, not what this test checks.
async fn seed_selective(db: &Db, c: &RequestContext) -> Res<()> {
    let big = user("big");
    let rare = user("rare");
    let big_subject = entity(db, c, &big, "big").await?;
    let rare_subject = entity(db, c, &rare, "rare").await?;

    for batch in 0..12u64 {
        let episode_id = new_episode(db, c, &rare, Trust::User).await?;
        let mut fact = new_fact(&rare, rare_subject, "rare fact", date(2020, 1));
        fact.embedding = noisy(10_000 + batch, 0.7);
        db.store
            .apply_extraction(c, episode_id, vec![FactOp::Add(fact)])
            .await?;

        let episode_id = new_episode(db, c, &big, Trust::User).await?;
        let ops = (0..25u64)
            .map(|i| {
                let mut fact = new_fact(&big, big_subject, "big fact", date(2020, 1));
                fact.embedding = noisy(batch * 100 + i, 0.5);
                FactOp::Add(fact)
            })
            .collect();
        db.store.apply_extraction(c, episode_id, ops).await?;
    }
    sqlx::query("ANALYZE facts").execute(&db.admin).await?;
    Ok(())
}

#[tokio::test]
async fn vector_search_returns_full_limit_under_a_selective_filter_on_the_hnsw_index() -> TestResult
{
    let db = db().await;
    let c = ctx("t1");
    seed_selective(&db, &c).await?;

    // Disabling sort and seq scan leaves the HNSW index as the only cheap plan, which is the case
    // iterative scan exists for. Without these the planner may sort a small table and pass anyway.
    let forced = format!(
        "{}?options=-c%20enable_sort%3Doff%20-c%20enable_seqscan%3Doff",
        db.url
    );
    let store = PgStore::connect(&forced, 2).await?;
    let found = store
        .search_facts_vector(&c, &[user("rare")], &axis(0), &FactFilter::default(), 5)
        .await?;
    assert_eq!(found.len(), 5);
    assert!(found.iter().all(|s| s.fact.scope == user("rare")));
    assert!(found.windows(2).all(|w| w[0].score >= w[1].score));
    // Noise 0.7 on axis(0) gives cosine similarity near 1 / sqrt(1 + 0.7^2).
    assert!((found[0].score - 0.8192).abs() < 0.1, "{}", found[0].score);

    // Control: same plan settings without iterative scan. Proves the assertion above is meaningful.
    let mut tx = db.admin.begin().await?;
    for sql in [
        "SET LOCAL ROLE memoryos_app",
        "SET LOCAL enable_sort = off",
        "SET LOCAL enable_seqscan = off",
    ] {
        sqlx::query(sql).execute(&mut *tx).await?;
    }
    sqlx::query("SELECT set_config('app.tenant', 't1', true)")
        .execute(&mut *tx)
        .await?;
    let control = "SELECT id FROM facts WHERE scope = 'user:rare' \
                   ORDER BY embedding <=> $1 LIMIT 5";
    let plan = sqlx::query(&format!("EXPLAIN {control}"))
        .bind(Vector::from(axis(0)))
        .fetch_all(&mut *tx)
        .await?;
    let plan_text = plan
        .iter()
        .map(|r| r.try_get::<String, _>(0))
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    assert!(plan_text.contains("facts_embedding_idx"), "{plan_text}");
    let rows = sqlx::query(control)
        .bind(Vector::from(axis(0)))
        .fetch_all(&mut *tx)
        .await?;
    assert!(rows.len() < 5, "control returned {} rows", rows.len());
    Ok(())
}

#[tokio::test]
async fn vector_search_validates_input() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scopes = [user("u1")];
    let filter = FactFilter::default();
    assert!(is_invalid(
        db.store
            .search_facts_vector(&c, &scopes, &[1.0; 3], &filter, 5)
            .await
    ));
    assert!(is_invalid(
        db.store
            .search_facts_vector(&c, &scopes, &vec![0.0; EMBEDDING_DIMS], &filter, 5)
            .await
    ));
    assert!(is_invalid(
        db.store
            .search_facts_vector(&c, &scopes, &axis(0), &filter, 0)
            .await
    ));
    assert!(db
        .store
        .search_facts_vector(&c, &[], &axis(0), &filter, 5)
        .await?
        .is_empty());
    Ok(())
}

#[tokio::test]
async fn text_search_ranks_filters_scope_and_treats_query_as_data() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let other = user("u2");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let other_subject = entity(&db, &c, &other, "bob").await?;
    let (_, weak) = add_fact(
        &db,
        &c,
        new_fact(
            &scope,
            subject,
            "rust and a lot of other words about nothing much",
            date(2020, 1),
        ),
    )
    .await?;
    let (_, strong) = add_fact(
        &db,
        &c,
        new_fact(&scope, subject, "rust rust rust", date(2020, 1)),
    )
    .await?;
    add_fact(
        &db,
        &c,
        new_fact(&other, other_subject, "rust elsewhere", date(2020, 1)),
    )
    .await?;

    let filter = FactFilter::default();
    let found = db
        .store
        .search_facts_text(&c, std::slice::from_ref(&scope), "rust", &filter, 10)
        .await?;
    assert_eq!(
        found.iter().map(|s| s.fact.id).collect::<Vec<_>>(),
        vec![strong, weak]
    );
    assert!(found[0].score > found[1].score);

    // websearch syntax: quoted phrase and negation.
    let phrase = db
        .store
        .search_facts_text(
            &c,
            std::slice::from_ref(&scope),
            "\"rust and\" -nothingness",
            &filter,
            10,
        )
        .await?;
    assert_eq!(phrase.len(), 1);

    for hostile in ["'; DROP TABLE facts; --", "", "&&& ((( :*"] {
        let result = db
            .store
            .search_facts_text(&c, std::slice::from_ref(&scope), hostile, &filter, 10)
            .await?;
        assert!(result.is_empty(), "{hostile}");
    }
    assert_eq!(count(&db, "SELECT count(*) FROM facts").await?, 3);
    Ok(())
}

#[tokio::test]
async fn pending_episode_search_returns_only_unprocessed_user_and_system() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let trusted = db
        .store
        .insert_episode(&c, episode(&scope, Trust::User, "the quick fox"))
        .await?;
    db.store
        .insert_episode(&c, episode(&scope, Trust::External, "the quick poison"))
        .await?;
    db.store
        .insert_episode(&c, episode(&scope, Trust::Agent, "the quick planted"))
        .await?;
    let system = db
        .store
        .insert_episode(&c, episode(&scope, Trust::System, "the quick system"))
        .await?;
    db.store
        .insert_episode(&c, episode(&user("u2"), Trust::User, "the quick other"))
        .await?;

    let found = db
        .store
        .search_pending_episodes(&c, &scopes, "quick", 10)
        .await?;
    let ids: BTreeSet<Uuid> = found.iter().map(|s| s.episode.id).collect();
    assert_eq!(ids, BTreeSet::from([trusted.id, system.id]));
    assert!(found[0].score > 0.0);

    db.store.apply_extraction(&c, trusted.id, vec![]).await?;
    db.store.apply_extraction(&c, system.id, vec![]).await?;
    assert!(db
        .store
        .search_pending_episodes(&c, &scopes, "quick", 10)
        .await?
        .is_empty());
    Ok(())
}

// ---- Entities --------------------------------------------------------------------------------

#[tokio::test]
async fn entities_upsert_by_normalized_name_and_rank_by_similarity() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let make = |name: &str, embedding| NewEntity {
        scope: scope.clone(),
        kind: "person".into(),
        name: name.into(),
        aliases: vec![],
        embedding,
    };
    let first = db
        .store
        .upsert_entity(&c, make("Ada  Lovelace", axis(5)))
        .await?;
    let same = db
        .store
        .upsert_entity(&c, make(" ada lovelace ", axis(6)))
        .await?;
    assert_eq!(first.id, same.id);
    assert_eq!(same.name, "Ada  Lovelace");

    let far = db.store.upsert_entity(&c, make("Bob", axis(6))).await?;
    let found = db.store.similar_entities(&c, &scope, &axis(5), 10).await?;
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].0.id, first.id);
    assert!((found[0].1 - 1.0).abs() < 1e-6);
    assert_eq!(found[1].0.id, far.id);
    assert!(found[1].1.abs() < 1e-6);

    db.store
        .add_entity_alias(&c, first.id, "Countess", None)
        .await?;
    db.store
        .add_entity_alias(&c, first.id, "Countess", None)
        .await?;
    assert_eq!(
        db.store.get_entity(&c, first.id).await?.aliases,
        vec!["Countess".to_owned()]
    );
    assert!(is_not_found(
        db.store
            .add_entity_alias(&c, Uuid::now_v7(), "x", None)
            .await
    ));
    assert!(is_not_found(db.store.get_entity(&c, Uuid::now_v7()).await));
    Ok(())
}

// ---- Graph -----------------------------------------------------------------------------------

/// Applies one edge fact with its own episode.
async fn edge(db: &Db, c: &RequestContext, scope: &Scope, a: Uuid, b: Uuid) -> Res<Uuid> {
    let mut fact = new_fact(scope, a, "edge", date(2020, 1));
    fact.predicate = "links".into();
    fact.object_id = Some(b);
    Ok(add_fact(db, c, fact).await?.1)
}

#[tokio::test]
async fn neighbors_respects_depth_and_terminates_on_cycles() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let a = entity(&db, &c, &scope, "a").await?;
    let b = entity(&db, &c, &scope, "b").await?;
    let d = entity(&db, &c, &scope, "d").await?;
    let e = entity(&db, &c, &scope, "e").await?;
    // a -> b -> d -> e, and d -> a closes a cycle. e -> e is a self-loop.
    let ab = edge(&db, &c, &scope, a, b).await?;
    let bd = edge(&db, &c, &scope, b, d).await?;
    let da = edge(&db, &c, &scope, d, a).await?;
    let de = edge(&db, &c, &scope, d, e).await?;
    let ee = edge(&db, &c, &scope, e, e).await?;
    let filter = FactFilter::default();

    let entity_ids = |n: &memoryos_core::GraphNeighborhood| {
        n.entities.iter().map(|x| x.id).collect::<BTreeSet<_>>()
    };
    let edge_ids = |n: &memoryos_core::GraphNeighborhood| {
        n.edges.iter().map(|x| x.id).collect::<BTreeSet<_>>()
    };

    // Depth 1 from `a` follows edges in both directions: a->b and d->a.
    let one = db.store.neighbors(&c, &scopes, a, 1, &filter, 5000).await?;
    assert_eq!(one.root, a);
    assert_eq!(entity_ids(&one), BTreeSet::from([a, b, d]));
    assert_eq!(edge_ids(&one), BTreeSet::from([ab, da]));

    let two = db.store.neighbors(&c, &scopes, a, 2, &filter, 5000).await?;
    assert_eq!(entity_ids(&two), BTreeSet::from([a, b, d, e]));
    assert_eq!(edge_ids(&two), BTreeSet::from([ab, da, bd, de]));

    let five = db.store.neighbors(&c, &scopes, a, 5, &filter, 5000).await?;
    assert_eq!(entity_ids(&five), BTreeSet::from([a, b, d, e]));
    assert_eq!(edge_ids(&five), BTreeSet::from([ab, bd, da, de, ee]));
    assert_eq!(five.edges.len(), 5, "no duplicates after a cycle");

    for bad in [0, 6, -1] {
        assert!(is_invalid(
            db.store.neighbors(&c, &scopes, a, bad, &filter, 5000).await
        ));
    }
    Ok(())
}

#[tokio::test]
async fn neighbors_applies_scope_and_time_filter_to_edges() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let a = entity(&db, &c, &scope, "a").await?;
    let b = entity(&db, &c, &scope, "b").await?;
    let later = entity(&db, &c, &scope, "later").await?;

    let old_edge = edge(&db, &c, &scope, a, b).await?;
    let mut replacement = new_fact(&scope, a, "edge two", date(2022, 1));
    replacement.object_id = Some(later);
    let new_edge = supersede(&db, &c, old_edge, replacement).await?;

    let current = db
        .store
        .neighbors(&c, &scopes, a, 2, &FactFilter::default(), 5000)
        .await?;
    assert_eq!(
        current.edges.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![new_edge]
    );
    assert!(current.entities.iter().all(|x| x.id != b));

    let past = db
        .store
        .neighbors(&c, &scopes, a, 2, &at(date(2021, 1)), 5000)
        .await?;
    assert_eq!(
        past.edges.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![old_edge]
    );

    // Retraction removes an edge from the graph as well, and the edge it closed is current again.
    db.store.invalidate_fact(&c, new_edge, "wrong").await?;
    let after = db
        .store
        .neighbors(&c, &scopes, a, 2, &FactFilter::default(), 5000)
        .await?;
    assert_eq!(
        after.edges.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![old_edge]
    );

    // A root outside the caller's scopes is indistinguishable from a missing one.
    assert!(is_not_found(
        db.store
            .neighbors(&c, &[user("u2")], a, 1, &FactFilter::default(), 5000)
            .await
    ));
    assert!(is_not_found(
        db.store
            .neighbors(&c, &scopes, Uuid::now_v7(), 1, &FactFilter::default(), 5000)
            .await
    ));
    Ok(())
}

// ---- Purge -----------------------------------------------------------------------------------

#[tokio::test]
async fn purge_episode_keeps_shared_facts_and_removes_sole_sourced_ones() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let shared_subject = entity(&db, &c, &scope, "shared").await?;
    let lonely_subject = entity(&db, &c, &scope, "lonely").await?;
    let lonely_object = entity(&db, &c, &scope, "object").await?;

    let bad = new_episode(&db, &c, &scope, Trust::User).await?;
    let mut lonely = new_fact(&scope, lonely_subject, "only bad says so", date(2020, 1));
    lonely.object_id = Some(lonely_object);
    let inserted = db
        .store
        .apply_extraction(
            &c,
            bad,
            vec![
                FactOp::Add(new_fact(
                    &scope,
                    shared_subject,
                    "both say so",
                    date(2020, 1),
                )),
                FactOp::Add(lonely),
            ],
        )
        .await?;
    let (shared_fact, lonely_fact) = (inserted[0], inserted[1]);

    let good = new_episode(&db, &c, &scope, Trust::User).await?;
    db.store
        .apply_extraction(
            &c,
            good,
            vec![FactOp::Corroborate {
                fact_id: shared_fact,
            }],
        )
        .await?;

    let report = db.store.purge_episode(&c, bad).await?;
    assert_eq!(report.episodes_deleted, 1);
    assert_eq!(report.facts_deleted, 1);
    assert_eq!(report.entities_deleted, 2);

    let kept = db.store.get_fact(&c, shared_fact).await?;
    assert_eq!(kept.source_episode_ids, vec![good]);
    assert!(is_not_found(db.store.get_fact(&c, lonely_fact).await));
    assert!(is_not_found(db.store.get_episode(&c, bad).await));
    assert!(is_not_found(db.store.get_entity(&c, lonely_subject).await));
    assert!(is_not_found(db.store.get_entity(&c, lonely_object).await));
    db.store.get_entity(&c, shared_subject).await?;

    let row = sqlx::query("SELECT actor, target_id, detail FROM audit_events WHERE action = $1")
        .bind("episode.purge")
        .fetch_one(&db.admin)
        .await?;
    assert_eq!(row.try_get::<String, _>("actor")?, "tester");
    assert_eq!(row.try_get::<Option<Uuid>, _>("target_id")?, Some(bad));
    assert_eq!(
        row.try_get::<serde_json::Value, _>("detail")?["facts_deleted"],
        json!(1)
    );

    assert!(is_not_found(db.store.purge_episode(&c, bad).await));
    Ok(())
}

#[tokio::test]
async fn purge_episode_reopens_the_fact_a_poisoned_fact_had_closed() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, truth) = add_fact(&db, &c, new_fact(&scope, subject, "truth", date(2020, 1))).await?;

    let bad = new_episode(&db, &c, &scope, Trust::User).await?;
    let lie = db
        .store
        .apply_extraction(
            &c,
            bad,
            vec![FactOp::Supersede {
                old_fact_id: truth,
                new: new_fact(&scope, subject, "lie", date(2021, 1)),
            }],
        )
        .await?[0];
    assert_eq!(db.store.get_fact(&c, truth).await?.superseded_by, Some(lie));

    db.store.purge_episode(&c, bad).await?;
    let reopened = db.store.get_fact(&c, truth).await?;
    assert_eq!(reopened.valid_to, None);
    assert_eq!(reopened.superseded_by, None);
    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert_eq!(
        current.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![truth]
    );
    Ok(())
}

#[tokio::test]
async fn purge_episode_in_the_middle_of_a_chain_relinks_it() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, first) = add_fact(&db, &c, new_fact(&scope, subject, "a", date(2020, 1))).await?;
    let bad = new_episode(&db, &c, &scope, Trust::User).await?;
    let middle = db
        .store
        .apply_extraction(
            &c,
            bad,
            vec![FactOp::Supersede {
                old_fact_id: first,
                new: new_fact(&scope, subject, "b", date(2021, 1)),
            }],
        )
        .await?[0];
    let last = supersede(
        &db,
        &c,
        middle,
        new_fact(&scope, subject, "c", date(2022, 1)),
    )
    .await?;

    db.store.purge_episode(&c, bad).await?;
    let relinked = db.store.get_fact(&c, first).await?;
    assert_eq!(relinked.valid_to, Some(date(2022, 1)));
    assert_eq!(relinked.superseded_by, Some(last));
    assert_eq!(db.store.get_fact(&c, last).await?.valid_to, None);
    Ok(())
}

fn with_trust(mut fact: NewFact, trust: Trust) -> NewFact {
    fact.trust = trust;
    fact
}

async fn corroborate(
    db: &Db,
    c: &RequestContext,
    scope: &Scope,
    trust: Trust,
    id: Uuid,
) -> Res<Uuid> {
    let episode_id = new_episode(db, c, scope, trust).await?;
    db.store
        .apply_extraction(c, episode_id, vec![FactOp::Corroborate { fact_id: id }])
        .await?;
    Ok(episode_id)
}

#[tokio::test]
async fn purging_the_trusted_source_quarantines_a_fact_left_on_an_external_one() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (good, fact) = add_fact(&db, &c, new_fact(&scope, subject, "claim", date(2020, 1))).await?;
    let external = corroborate(&db, &c, &scope, Trust::External, fact).await?;

    let report = db.store.purge_episode(&c, good).await?;
    assert_eq!(report.facts_deleted, 0);

    let left = db.store.get_fact(&c, fact).await?;
    assert_eq!(left.status, FactStatus::Quarantined);
    assert_eq!(left.trust, Trust::External);
    assert_eq!(left.source_episode_ids, vec![external]);
    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert!(current.is_empty());
    let detail: serde_json::Value =
        sqlx::query_scalar("SELECT detail FROM audit_events WHERE action = 'episode.purge'")
            .fetch_one(&db.admin)
            .await?;
    assert_eq!(detail["facts_requarantined"], json!(1));
    Ok(())
}

#[tokio::test]
async fn purge_applies_the_agent_rule_by_scope() -> TestResult {
    let db = db().await;
    let c = ctx("t1");

    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (good, fact) = add_fact(&db, &c, new_fact(&scope, subject, "claim", date(2020, 1))).await?;
    corroborate(&db, &c, &scope, Trust::Agent, fact).await?;
    db.store.purge_episode(&c, good).await?;
    let left = db.store.get_fact(&c, fact).await?;
    assert_eq!(
        (left.status, left.trust),
        (FactStatus::Quarantined, Trust::Agent)
    );

    // In its own scope an agent's fact may stay active, at the lower trust.
    let scope = Scope::agent("a1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (good, fact) = add_fact(&db, &c, new_fact(&scope, subject, "claim", date(2020, 1))).await?;
    corroborate(&db, &c, &scope, Trust::Agent, fact).await?;
    db.store.purge_episode(&c, good).await?;
    let left = db.store.get_fact(&c, fact).await?;
    assert_eq!(
        (left.status, left.trust),
        (FactStatus::Active, Trust::Agent)
    );
    Ok(())
}

#[tokio::test]
async fn purge_recomputes_trust_as_the_highest_remaining_source() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let fact = with_trust(
        new_fact(&scope, subject, "claim", date(2020, 1)),
        Trust::System,
    );
    let (top, fact) = add_fact(&db, &c, fact).await?;
    corroborate(&db, &c, &scope, Trust::External, fact).await?;
    corroborate(&db, &c, &scope, Trust::User, fact).await?;

    db.store.purge_episode(&c, top).await?;
    let left = db.store.get_fact(&c, fact).await?;
    assert_eq!((left.status, left.trust), (FactStatus::Active, Trust::User));
    Ok(())
}

#[tokio::test]
async fn requarantined_fact_reopens_what_it_closed_and_approval_restores_it() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, truth) = add_fact(&db, &c, new_fact(&scope, subject, "truth", date(2020, 1))).await?;
    let good = new_episode(&db, &c, &scope, Trust::User).await?;
    let claim = db
        .store
        .apply_extraction(
            &c,
            good,
            vec![FactOp::Supersede {
                old_fact_id: truth,
                new: new_fact(&scope, subject, "claim", date(2021, 1)),
            }],
        )
        .await?[0];
    corroborate(&db, &c, &scope, Trust::External, claim).await?;

    db.store.purge_episode(&c, good).await?;
    let reopened = db.store.get_fact(&c, truth).await?;
    assert_eq!((reopened.valid_to, reopened.superseded_by), (None, None));
    let held = db.store.get_fact(&c, claim).await?;
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!(held.pending_supersede_of, Some(truth));
    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert_eq!(
        current.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![truth]
    );

    db.store.approve_fact(&c, claim).await?;
    let closed = db.store.get_fact(&c, truth).await?;
    assert_eq!(closed.superseded_by, Some(claim));
    assert_eq!(closed.valid_to, Some(date(2021, 1)));
    Ok(())
}

#[tokio::test]
async fn requarantined_fact_in_the_middle_of_a_chain_leaves_it_linked() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, first) = add_fact(&db, &c, new_fact(&scope, subject, "a", date(2020, 1))).await?;
    let good = new_episode(&db, &c, &scope, Trust::User).await?;
    let middle = db
        .store
        .apply_extraction(
            &c,
            good,
            vec![FactOp::Supersede {
                old_fact_id: first,
                new: new_fact(&scope, subject, "b", date(2021, 1)),
            }],
        )
        .await?[0];
    let last = supersede(
        &db,
        &c,
        middle,
        new_fact(&scope, subject, "c", date(2022, 1)),
    )
    .await?;
    corroborate(&db, &c, &scope, Trust::External, middle).await?;

    db.store.purge_episode(&c, good).await?;
    let relinked = db.store.get_fact(&c, first).await?;
    assert_eq!(relinked.superseded_by, Some(last));
    assert_eq!(relinked.valid_to, Some(date(2022, 1)));
    let held = db.store.get_fact(&c, middle).await?;
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!((held.superseded_by, held.valid_to), (None, None));
    assert_eq!(held.pending_supersede_of, Some(last));

    // Approval makes it history again and the head stays current.
    let approved = db.store.approve_fact(&c, middle).await?;
    assert_eq!(approved.superseded_by, Some(last));
    assert_eq!(approved.valid_to, Some(date(2022, 1)));
    assert_eq!(db.store.get_fact(&c, last).await?.valid_to, None);
    Ok(())
}

async fn current_ids(db: &Db, c: &RequestContext, scope: &Scope, subject: Uuid) -> Res<Vec<Uuid>> {
    let current = db
        .store
        .current_facts_for_subject(c, scope, subject)
        .await?;
    Ok(current.iter().map(|f| f.id).collect())
}

/// Like `supersede`, also returning the episode. Returns (episode id, fact id).
async fn supersede_from_episode(
    db: &Db,
    c: &RequestContext,
    old_fact_id: Uuid,
    new: NewFact,
) -> Res<(Uuid, Uuid)> {
    let episode_id = new_episode(db, c, &new.scope, new.trust).await?;
    let ids = db
        .store
        .apply_extraction(c, episode_id, vec![FactOp::Supersede { old_fact_id, new }])
        .await?;
    Ok((episode_id, ids[0]))
}

#[tokio::test]
async fn purge_never_raises_trust_above_the_current_value() -> TestResult {
    let db = db().await;
    let c = ctx("t1");

    // Active in its own scope: the system corroboration must not lift it.
    let scope = Scope::agent("a1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let fact = with_trust(
        new_fact(&scope, subject, "claim", date(2020, 1)),
        Trust::Agent,
    );
    let (agent, fact) = add_fact(&db, &c, fact).await?;
    corroborate(&db, &c, &scope, Trust::System, fact).await?;
    db.store.purge_episode(&c, agent).await?;
    let left = db.store.get_fact(&c, fact).await?;
    assert_eq!(
        (left.status, left.trust),
        (FactStatus::Active, Trust::Agent)
    );

    // Quarantined: losing the external source must not launder it into a trusted fact.
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let mut fact = with_trust(
        new_fact(&scope, subject, "claim", date(2020, 1)),
        Trust::External,
    );
    fact.status = FactStatus::Quarantined;
    let (external, fact) = add_fact(&db, &c, fact).await?;
    corroborate(&db, &c, &scope, Trust::System, fact).await?;
    db.store.purge_episode(&c, external).await?;
    let left = db.store.get_fact(&c, fact).await?;
    assert_eq!(
        (left.status, left.trust),
        (FactStatus::Quarantined, Trust::External)
    );
    Ok(())
}

#[tokio::test]
async fn pending_link_on_a_purged_head_moves_to_the_reopened_predecessor() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, truth) = add_fact(&db, &c, new_fact(&scope, subject, "truth", date(2020, 1))).await?;
    let (bad, lie) = supersede_from_episode(
        &db,
        &c,
        truth,
        new_fact(&scope, subject, "lie", date(2021, 1)),
    )
    .await?;
    let pending = pending_on(&db, &c, &scope, subject, lie).await?;

    db.store.purge_episode(&c, bad).await?;
    assert_eq!(
        db.store.get_fact(&c, pending).await?.pending_supersede_of,
        Some(truth)
    );

    db.store.approve_fact(&c, pending).await?;
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![pending]);
    assert_eq!(
        db.store.get_fact(&c, truth).await?.superseded_by,
        Some(pending)
    );
    Ok(())
}

#[tokio::test]
async fn pending_link_on_a_purged_middle_moves_to_the_current_fact() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, first) = add_fact(&db, &c, new_fact(&scope, subject, "a", date(2020, 1))).await?;
    let (bad, middle) = supersede_from_episode(
        &db,
        &c,
        first,
        new_fact(&scope, subject, "b", date(2021, 1)),
    )
    .await?;
    let last = supersede(
        &db,
        &c,
        middle,
        new_fact(&scope, subject, "c", date(2022, 1)),
    )
    .await?;
    let pending = pending_on(&db, &c, &scope, subject, middle).await?;

    db.store.purge_episode(&c, bad).await?;
    assert_eq!(
        db.store.get_fact(&c, pending).await?.pending_supersede_of,
        Some(last)
    );

    db.store.approve_fact(&c, pending).await?;
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![pending]);
    Ok(())
}

#[tokio::test]
async fn pending_link_on_a_requarantined_fact_moves_to_the_reopened_predecessor() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, truth) = add_fact(&db, &c, new_fact(&scope, subject, "truth", date(2020, 1))).await?;
    let (good, claim) = supersede_from_episode(
        &db,
        &c,
        truth,
        new_fact(&scope, subject, "claim", date(2021, 1)),
    )
    .await?;
    corroborate(&db, &c, &scope, Trust::External, claim).await?;
    let pending = pending_on(&db, &c, &scope, subject, claim).await?;

    db.store.purge_episode(&c, good).await?;
    assert_eq!(
        db.store.get_fact(&c, claim).await?.status,
        FactStatus::Quarantined
    );
    assert_eq!(
        db.store.get_fact(&c, pending).await?.pending_supersede_of,
        Some(truth)
    );

    db.store.approve_fact(&c, pending).await?;
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![pending]);
    Ok(())
}

#[tokio::test]
async fn pending_link_is_cleared_when_the_purge_leaves_nothing_to_close() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (bad, lie) = add_fact(&db, &c, new_fact(&scope, subject, "lie", date(2020, 1))).await?;
    let pending = pending_on(&db, &c, &scope, subject, lie).await?;

    db.store.purge_episode(&c, bad).await?;
    assert_eq!(
        db.store.get_fact(&c, pending).await?.pending_supersede_of,
        None
    );

    db.store.approve_fact(&c, pending).await?;
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![pending]);
    Ok(())
}

#[tokio::test]
async fn purge_that_leaves_a_fact_below_what_it_closed_quarantines_it() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let system = with_trust(
        new_fact(&scope, subject, "system says", date(2020, 1)),
        Trust::System,
    );
    let (_, truth) = add_fact(&db, &c, system).await?;
    let (source, claim) = supersede_from_episode(
        &db,
        &c,
        truth,
        with_trust(
            new_fact(&scope, subject, "claim", date(2021, 1)),
            Trust::System,
        ),
    )
    .await?;
    corroborate(&db, &c, &scope, Trust::User, claim).await?;

    db.store.purge_episode(&c, source).await?;
    let reopened = db.store.get_fact(&c, truth).await?;
    assert_eq!((reopened.valid_to, reopened.superseded_by), (None, None));
    let held = db.store.get_fact(&c, claim).await?;
    assert_eq!(
        (held.status, held.trust),
        (FactStatus::Quarantined, Trust::User)
    );
    assert_eq!(held.pending_supersede_of, Some(truth));
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![truth]);
    let detail: serde_json::Value =
        sqlx::query_scalar("SELECT detail FROM audit_events WHERE action = 'episode.purge'")
            .fetch_one(&db.admin)
            .await?;
    assert_eq!(detail["facts_requarantined"], json!(1));

    db.store.approve_fact(&c, claim).await?;
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![claim]);
    let closed = db.store.get_fact(&c, truth).await?;
    assert_eq!(closed.superseded_by, Some(claim));
    assert_eq!(closed.valid_to, Some(date(2021, 1)));
    Ok(())
}

#[tokio::test]
async fn purge_keeps_a_fact_active_when_what_it_closed_does_not_outrank_it() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;

    // The closed fact was already at the trust the closer falls to.
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
    let (source, claim) = supersede_from_episode(
        &db,
        &c,
        old,
        with_trust(
            new_fact(&scope, subject, "claim", date(2021, 1)),
            Trust::System,
        ),
    )
    .await?;
    corroborate(&db, &c, &scope, Trust::User, claim).await?;
    db.store.purge_episode(&c, source).await?;
    let left = db.store.get_fact(&c, claim).await?;
    assert_eq!((left.status, left.trust), (FactStatus::Active, Trust::User));
    assert_eq!(db.store.get_fact(&c, old).await?.superseded_by, Some(claim));

    // Both fall together, so the closed fact does not outrank the closer afterwards.
    let subject = entity(&db, &c, &scope, "grace").await?;
    let system = |text, from| with_trust(new_fact(&scope, subject, text, from), Trust::System);
    let (first_own, first) = add_fact(&db, &c, system("first", date(2020, 1))).await?;
    let (second_own, second) =
        supersede_from_episode(&db, &c, first, system("second", date(2021, 1))).await?;
    let shared = new_episode(&db, &c, &scope, Trust::System).await?;
    db.store
        .apply_extraction(
            &c,
            shared,
            vec![
                FactOp::Corroborate { fact_id: first },
                FactOp::Corroborate { fact_id: second },
            ],
        )
        .await?;
    corroborate(&db, &c, &scope, Trust::User, first).await?;
    corroborate(&db, &c, &scope, Trust::User, second).await?;
    db.store.purge_episode(&c, first_own).await?;
    db.store.purge_episode(&c, second_own).await?;
    db.store.purge_episode(&c, shared).await?;
    let left = db.store.get_fact(&c, second).await?;
    assert_eq!((left.status, left.trust), (FactStatus::Active, Trust::User));
    assert_eq!(
        db.store.get_fact(&c, first).await?.superseded_by,
        Some(second)
    );
    Ok(())
}

#[tokio::test]
async fn purge_applies_the_contradiction_rule_across_a_deleted_middle_fact() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let system = |text, from| with_trust(new_fact(&scope, subject, text, from), Trust::System);
    let (_, truth) = add_fact(&db, &c, system("truth", date(2020, 1))).await?;
    let (middle_own, middle) =
        supersede_from_episode(&db, &c, truth, system("middle", date(2021, 1))).await?;
    let (claim_own, claim) =
        supersede_from_episode(&db, &c, middle, system("claim", date(2022, 1))).await?;
    let shared = new_episode(&db, &c, &scope, Trust::System).await?;
    db.store
        .apply_extraction(
            &c,
            shared,
            vec![
                FactOp::Corroborate { fact_id: middle },
                FactOp::Corroborate { fact_id: claim },
            ],
        )
        .await?;
    corroborate(&db, &c, &scope, Trust::User, claim).await?;
    db.store.purge_episode(&c, middle_own).await?;
    db.store.purge_episode(&c, claim_own).await?;

    // Deletes the middle fact, leaving the claim, now user trust, as the closer of the system fact.
    db.store.purge_episode(&c, shared).await?;
    assert!(is_not_found(db.store.get_fact(&c, middle).await));
    let held = db.store.get_fact(&c, claim).await?;
    assert_eq!(
        (held.status, held.trust),
        (FactStatus::Quarantined, Trust::User)
    );
    assert_eq!(held.pending_supersede_of, Some(truth));
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![truth]);

    db.store.approve_fact(&c, claim).await?;
    assert_eq!(current_ids(&db, &c, &scope, subject).await?, vec![claim]);
    Ok(())
}

// ---- Trust at the head of a chain ------------------------------------------------------------

/// `old` closed by a system fact, so a writer that only saw `old` reaches a higher-trust head.
async fn chain_with_system_head(
    db: &Db,
    c: &RequestContext,
    scope: &Scope,
    subject: Uuid,
) -> Res<(Uuid, Uuid)> {
    let (_, old) = add_fact(db, c, new_fact(scope, subject, "old", date(2020, 1))).await?;
    let head = supersede(
        db,
        c,
        old,
        with_trust(
            new_fact(scope, subject, "system says", date(2021, 1)),
            Trust::System,
        ),
    )
    .await?;
    Ok((old, head))
}

#[tokio::test]
async fn supersede_that_reaches_a_higher_trust_head_is_quarantined() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (old, head) = chain_with_system_head(&db, &c, &scope, subject).await?;

    let new = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "user says", date(2022, 1)),
    )
    .await?;

    let open = db.store.get_fact(&c, head).await?;
    assert_eq!((open.valid_to, open.superseded_by), (None, None));
    let held = db.store.get_fact(&c, new).await?;
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!(held.pending_supersede_of, Some(head));
    assert_eq!((held.valid_to, held.superseded_by), (None, None));

    // Approval reviewed exactly this head, so it may close it.
    let approved = db.store.approve_fact(&c, new).await?;
    assert_eq!(approved.status, FactStatus::Active);
    let closed = db.store.get_fact(&c, head).await?;
    assert_eq!(closed.superseded_by, Some(new));
    assert_eq!(closed.valid_to, Some(date(2022, 1)));
    Ok(())
}

#[tokio::test]
async fn backdated_supersede_of_a_higher_trust_head_is_quarantined_too() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (old, head) = chain_with_system_head(&db, &c, &scope, subject).await?;
    let new = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "user says", date(2020, 6)),
    )
    .await?;
    let held = db.store.get_fact(&c, new).await?;
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!(held.pending_supersede_of, Some(head));
    Ok(())
}

#[tokio::test]
async fn supersede_that_reaches_an_equal_trust_head_closes_it() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
    let head = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "head", date(2021, 1)),
    )
    .await?;
    let new = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "new", date(2022, 1)),
    )
    .await?;
    assert_eq!(db.store.get_fact(&c, new).await?.status, FactStatus::Active);
    assert_eq!(db.store.get_fact(&c, head).await?.superseded_by, Some(new));
    Ok(())
}

/// A quarantined fact pending on `old`, as an extraction would leave it.
async fn pending_on(
    db: &Db,
    c: &RequestContext,
    scope: &Scope,
    subject: Uuid,
    old: Uuid,
) -> Res<Uuid> {
    let mut fact = new_fact(scope, subject, "pending", date(2023, 1));
    fact.status = FactStatus::Quarantined;
    supersede(db, c, old, fact).await
}

#[tokio::test]
async fn approval_is_held_when_the_target_moved_to_a_higher_trust_head() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
    let pending = pending_on(&db, &c, &scope, subject, old).await?;
    let head = supersede(
        &db,
        &c,
        old,
        with_trust(
            new_fact(&scope, subject, "system says", date(2021, 1)),
            Trust::System,
        ),
    )
    .await?;

    assert!(is_invalid(db.store.approve_fact(&c, pending).await));
    let held = db.store.get_fact(&c, pending).await?;
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!(held.pending_supersede_of, Some(head));
    let open = db.store.get_fact(&c, head).await?;
    assert_eq!((open.valid_to, open.superseded_by), (None, None));
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM audit_events WHERE action = 'fact.approve_held'"
        )
        .await?,
        1
    );

    // The second approval is a decision about this head.
    let approved = db.store.approve_fact(&c, pending).await?;
    assert_eq!(approved.status, FactStatus::Active);
    assert_eq!(
        db.store.get_fact(&c, head).await?.superseded_by,
        Some(pending)
    );
    Ok(())
}

#[tokio::test]
async fn approval_closes_a_moved_head_of_equal_trust_and_skips_a_retracted_one() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;

    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
    let pending = pending_on(&db, &c, &scope, subject, old).await?;
    let head = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "head", date(2021, 1)),
    )
    .await?;
    db.store.approve_fact(&c, pending).await?;
    assert_eq!(
        db.store.get_fact(&c, head).await?.superseded_by,
        Some(pending)
    );

    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old 2", date(2020, 1))).await?;
    let pending = pending_on(&db, &c, &scope, subject, old).await?;
    db.store.invalidate_fact(&c, old, "wrong").await?;
    let approved = db.store.approve_fact(&c, pending).await?;
    assert_eq!(approved.status, FactStatus::Active);
    assert_eq!(db.store.get_fact(&c, old).await?.superseded_by, None);
    Ok(())
}

// ---- Episode facts, entity lookup, edge limits and job release -------------------------------

#[tokio::test]
async fn facts_for_episode_lists_created_and_corroborated_facts_in_any_status() -> TestResult {
    let db = db().await;
    let (c, other) = (ctx("t1"), ctx("t2"));
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, linked) = add_fact(&db, &c, new_fact(&scope, subject, "linked", date(2020, 1))).await?;

    let mut held = new_fact(&scope, subject, "held", date(2020, 1));
    held.status = FactStatus::Quarantined;
    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;
    let inserted = db
        .store
        .apply_extraction(
            &c,
            episode_id,
            vec![
                FactOp::Add(new_fact(&scope, subject, "first", date(2020, 1))),
                FactOp::Corroborate { fact_id: linked },
                FactOp::Add(held),
            ],
        )
        .await?;

    let found = db.store.facts_for_episode(&c, episode_id).await?;
    assert_eq!(
        found.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![linked, inserted[0], inserted[1]]
    );
    assert_eq!(found[2].status, FactStatus::Quarantined);

    let empty = new_episode(&db, &c, &scope, Trust::User).await?;
    assert!(db.store.facts_for_episode(&c, empty).await?.is_empty());
    assert!(is_not_found(
        db.store.facts_for_episode(&c, Uuid::now_v7()).await
    ));
    assert!(is_not_found(
        db.store.facts_for_episode(&other, episode_id).await
    ));
    Ok(())
}

#[tokio::test]
async fn find_entity_matches_normalized_names_and_aliases_exactly() -> TestResult {
    let db = db().await;
    let (c, other) = (ctx("t1"), ctx("t2"));
    let scope = user("u1");
    let robert = entity(&db, &c, &scope, "Robert  Smith").await?;
    db.store
        .add_entity_alias(&c, robert, "Bobby  S", None)
        .await?;
    let bob = entity(&db, &c, &scope, "Bob").await?;

    let find = |name: &'static str| {
        let (store, c, scope) = (db.store.clone(), c.clone(), scope.clone());
        async move { store.find_entity(&c, &scope, "thing", name).await }
    };
    assert_eq!(find(" ROBERT smith ").await?.map(|e| e.id), Some(robert));
    assert_eq!(find("bobby s").await?.map(|e| e.id), Some(robert));
    assert_eq!(find("Bob").await?.map(|e| e.id), Some(bob));
    assert!(find("Rob").await?.is_none());

    // A name beats another entity's alias.
    db.store.add_entity_alias(&c, robert, "Bob", None).await?;
    assert_eq!(find("bob").await?.map(|e| e.id), Some(bob));

    let store = &db.store;
    assert!(store
        .find_entity(&c, &scope, "other", "Bob")
        .await?
        .is_none());
    assert!(store
        .find_entity(&c, &user("u2"), "thing", "Bob")
        .await?
        .is_none());
    assert!(store
        .find_entity(&other, &scope, "thing", "Bob")
        .await?
        .is_none());
    assert!(is_invalid(
        store.find_entity(&c, &scope, "thing", " ").await
    ));
    Ok(())
}

#[tokio::test]
async fn neighbors_returns_the_newest_edges_up_to_the_limit() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let filter = FactFilter::default();
    let hub = entity(&db, &c, &scope, "hub").await?;
    let spoke = entity(&db, &c, &scope, "spoke").await?;

    // Half point out of the hub and half into it: both directions count toward one limit.
    let mut made = Vec::new();
    for i in 0..12 {
        let fact = if i % 2 == 0 {
            edge(&db, &c, &scope, hub, spoke).await?
        } else {
            edge(&db, &c, &scope, spoke, hub).await?
        };
        made.push(fact);
    }
    made.reverse();

    let some = db.store.neighbors(&c, &scopes, hub, 1, &filter, 5).await?;
    assert_eq!(
        some.edges.iter().map(|f| f.id).collect::<Vec<_>>(),
        made[..5]
    );

    // Above the hard cap is the cap, not an error.
    let all = db
        .store
        .neighbors(&c, &scopes, hub, 1, &filter, i64::MAX)
        .await?;
    assert_eq!(all.edges.iter().map(|f| f.id).collect::<Vec<_>>(), made);

    // The limit covers the whole walk, and earlier hops are not counted twice.
    let walk = db.store.neighbors(&c, &scopes, hub, 3, &filter, 7).await?;
    assert_eq!(walk.edges.len(), 7);
    let distinct: BTreeSet<Uuid> = walk.edges.iter().map(|f| f.id).collect();
    assert_eq!(distinct.len(), 7);

    for bad in [0, -1] {
        assert!(is_invalid(
            db.store.neighbors(&c, &scopes, hub, 1, &filter, bad).await
        ));
    }
    Ok(())
}

#[tokio::test]
async fn release_job_gives_the_attempt_back_and_is_fenced() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    new_episode(&db, &c, &user("u1"), Trust::User).await?;
    let job = db.store.claim_job("w", 30).await?.expect("a job");
    assert_eq!(job.attempts, 1);

    let later = Utc::now() + chrono::Duration::hours(1);
    assert!(!db.store.release_job(&job, "other", later).await?);
    let mut stale = job.clone();
    stale.attempts = 9;
    assert!(!db.store.release_job(&stale, "w", later).await?);
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM jobs WHERE status = 'running' AND attempts = 1"
        )
        .await?,
        1
    );

    assert!(db.store.release_job(&job, "w", later).await?);
    let row = sqlx::query("SELECT status, attempts, locked_by, locked_until, last_error FROM jobs")
        .fetch_one(&db.admin)
        .await?;
    assert_eq!(row.try_get::<String, _>("status")?, "pending");
    assert_eq!(row.try_get::<i32, _>("attempts")?, 0);
    assert_eq!(row.try_get::<Option<String>, _>("locked_by")?, None);
    assert_eq!(
        row.try_get::<Option<DateTime<Utc>>, _>("locked_until")?,
        None
    );
    assert_eq!(row.try_get::<Option<String>, _>("last_error")?, None);

    // Not due until `run_after`, and a second release of the same claim does nothing.
    assert!(db.store.claim_job("w", 30).await?.is_none());
    assert!(!db.store.release_job(&job, "w", later).await?);

    sqlx::query("UPDATE jobs SET run_after = now()")
        .execute(&db.admin)
        .await?;
    let again = db.store.claim_job("w", 30).await?.expect("claimable");
    assert_eq!(again.attempts, 1);
    assert!(db.store.complete_job(&again, "w").await?);
    assert!(!db.store.release_job(&again, "w", Utc::now()).await?);
    Ok(())
}

#[tokio::test]
async fn purge_scope_removes_everything_under_it_and_nothing_else() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let doomed = Scope::agent("bot");
    let kept = user("u1");

    let doomed_subject = entity(&db, &c, &doomed, "d").await?;
    add_fact(
        &db,
        &c,
        new_fact(&doomed, doomed_subject, "gone", date(2020, 1)),
    )
    .await?;
    let kept_subject = entity(&db, &c, &kept, "k").await?;
    let (_, kept_fact) = add_fact(
        &db,
        &c,
        new_fact(&kept, kept_subject, "stays", date(2020, 1)),
    )
    .await?;

    db.store
        .put_block(&c, "bot", "persona", "p", 100, None)
        .await?;
    db.store
        .put_block(&c, "other", "persona", "p", 100, None)
        .await?;
    let agent_session = db.store.create_session(&c, "bot", &doomed).await?;
    db.store
        .append_message(&c, agent_session.id, "user", json!([]))
        .await?;
    // The agent's session held under a user scope still belongs to the agent being purged.
    let bot_user_session = db.store.create_session(&c, "bot", &kept).await?;
    let other_session = db.store.create_session(&c, "other", &kept).await?;

    let report = db.store.purge_scope(&c, &doomed).await?;
    assert_eq!(report.facts_deleted, 1);
    assert_eq!(report.episodes_deleted, 1);
    assert_eq!(report.entities_deleted, 1);

    assert!(is_not_found(
        db.store.get_session(&c, agent_session.id).await
    ));
    assert!(is_not_found(
        db.store.get_session(&c, bot_user_session.id).await
    ));
    assert!(is_not_found(db.store.get_block(&c, "bot", "persona").await));
    assert_eq!(
        count(&db, "SELECT count(*) FROM session_messages").await?,
        0
    );

    db.store.get_fact(&c, kept_fact).await?;
    db.store.get_block(&c, "other", "persona").await?;
    db.store.get_session(&c, other_session.id).await?;
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM audit_events WHERE action = 'scope.purge'"
        )
        .await?,
        1
    );

    // A user purge removes that user's sessions too, since their messages are the user's data.
    let report = db.store.purge_scope(&c, &kept).await?;
    assert_eq!(report.facts_deleted, 1);
    assert!(is_not_found(
        db.store.get_session(&c, other_session.id).await
    ));
    Ok(())
}

// ---- Blocks ----------------------------------------------------------------------------------

#[tokio::test]
async fn put_block_checks_version_and_appends_history() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let created = db
        .store
        .put_block(&c, "a", "human", "v1", 100, None)
        .await?;
    assert_eq!(created.version, 1);

    let duplicate = db.store.put_block(&c, "a", "human", "x", 100, None).await;
    assert!(matches!(
        duplicate,
        Err(Error::VersionConflict {
            expected: 0,
            current: 1
        })
    ));

    let updated = db
        .store
        .put_block(&c, "a", "human", "v2", 100, Some(1))
        .await?;
    assert_eq!(updated.version, 2);
    assert_eq!(updated.id, created.id);

    let stale = db
        .store
        .put_block(&c, "a", "human", "x", 100, Some(1))
        .await;
    assert!(matches!(
        stale,
        Err(Error::VersionConflict {
            expected: 1,
            current: 2
        })
    ));
    assert_eq!(db.store.get_block(&c, "a", "human").await?.content, "v2");

    let versions = db.store.block_versions(&c, "a", "human").await?;
    assert_eq!(
        versions
            .iter()
            .map(|v| (v.version, v.content.as_str(), v.edited_by.as_str()))
            .collect::<Vec<_>>(),
        vec![(1, "v1", "tester"), (2, "v2", "tester")]
    );

    assert!(is_not_found(
        db.store
            .put_block(&c, "a", "ghost", "x", 100, Some(1))
            .await
    ));
    assert!(is_not_found(
        db.store.block_versions(&c, "a", "ghost").await
    ));
    assert_eq!(db.store.list_blocks(&c, "a").await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn put_block_limit_counts_characters_not_bytes() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    // Five characters, fifteen bytes.
    let cjk = "\u{8a18}\u{61b6}\u{7cfb}\u{7d71}\u{5eab}";
    db.store.put_block(&c, "a", "x", cjk, 5, None).await?;

    let too_large = db
        .store
        .put_block(&c, "a", "y", "\u{8a18}".repeat(6).as_str(), 5, None)
        .await;
    assert!(matches!(
        too_large,
        Err(Error::BlockTooLarge { len: 6, max: 5 })
    ));
    assert!(is_not_found(db.store.get_block(&c, "a", "y").await));

    let update = db.store.put_block(&c, "a", "x", "abcdef", 5, Some(1)).await;
    assert!(matches!(update, Err(Error::BlockTooLarge { .. })));
    assert_eq!(db.store.get_block(&c, "a", "x").await?.version, 1);
    Ok(())
}

#[tokio::test]
async fn concurrent_put_block_has_one_winner_per_version() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    db.store.put_block(&c, "a", "x", "v1", 100, None).await?;
    let calls = (0..8).map(|i| {
        let store = db.store.clone();
        let c = c.clone();
        tokio::spawn(async move {
            store
                .put_block(&c, "a", "x", &format!("w{i}"), 100, Some(1))
                .await
        })
    });
    let results = futures::future::join_all(calls).await;
    let mut winners = 0;
    for result in results {
        match result? {
            Ok(block) => {
                assert_eq!(block.version, 2);
                winners += 1;
            }
            Err(Error::VersionConflict { expected, current }) => {
                assert_eq!((expected, current), (1, 2));
            }
            Err(other) => panic!("unexpected error: {other}"),
        }
    }
    assert_eq!(winners, 1);
    assert_eq!(db.store.block_versions(&c, "a", "x").await?.len(), 2);
    Ok(())
}

// ---- Sessions --------------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_append_message_assigns_gapless_unique_seq() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let session = db.store.create_session(&c, "a", &user("u1")).await?;
    let calls = (0..32).map(|i| {
        let store = db.store.clone();
        let c = c.clone();
        let id = session.id;
        tokio::spawn(async move { store.append_message(&c, id, "user", json!([i])).await })
    });
    let mut seqs = Vec::new();
    for result in futures::future::join_all(calls).await {
        seqs.push(result??.seq);
    }
    seqs.sort_unstable();
    assert_eq!(seqs, (1..=32).collect::<Vec<i64>>());
    assert_eq!(db.store.load_messages(&c, session.id).await?.len(), 32);
    Ok(())
}

#[tokio::test]
async fn compact_session_replaces_old_messages_with_a_summary() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let session = db.store.create_session(&c, "a", &user("u1")).await?;
    assert_eq!(db.store.session_summary(&c, session.id).await?, None);
    for i in 0..5 {
        db.store
            .append_message(&c, session.id, "user", json!([{ "n": i }]))
            .await?;
    }

    db.store
        .compact_session(&c, session.id, 3, "first three")
        .await?;
    let left = db.store.load_messages(&c, session.id).await?;
    assert_eq!(left.iter().map(|m| m.seq).collect::<Vec<_>>(), vec![4, 5]);
    assert_eq!(left[0].content, json!([{ "n": 3 }]));
    assert_eq!(
        db.store.session_summary(&c, session.id).await?.as_deref(),
        Some("first three")
    );

    // New messages continue the sequence.
    let next = db
        .store
        .append_message(&c, session.id, "user", json!([]))
        .await?;
    assert_eq!(next.seq, 6);

    assert!(is_invalid(
        db.store.compact_session(&c, session.id, 2, "stale").await
    ));
    assert!(is_invalid(
        db.store.compact_session(&c, session.id, 99, "future").await
    ));
    assert_eq!(
        db.store.session_summary(&c, session.id).await?.as_deref(),
        Some("first three")
    );

    let ghost = Uuid::now_v7();
    assert!(is_not_found(
        db.store.append_message(&c, ghost, "user", json!([])).await
    ));
    assert!(is_not_found(db.store.load_messages(&c, ghost).await));
    assert!(is_not_found(db.store.session_summary(&c, ghost).await));
    assert!(is_not_found(
        db.store.compact_session(&c, ghost, 1, "x").await
    ));
    Ok(())
}

// ---- Jobs ------------------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_claimers_never_get_the_same_job() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    for _ in 0..10 {
        new_episode(&db, &c, &scope, Trust::User).await?;
    }
    let claimers = (0..16).map(|i| {
        let store = db.store.clone();
        tokio::spawn(async move { store.claim_job(&format!("w{i}"), 60).await })
    });
    let mut claimed = Vec::new();
    for result in futures::future::join_all(claimers).await {
        if let Some(job) = result?? {
            assert_eq!(job.attempts, 1);
            claimed.push(job.id);
        }
    }
    let unique: BTreeSet<Uuid> = claimed.iter().copied().collect();
    assert_eq!(unique.len(), claimed.len());
    assert!(!claimed.is_empty());

    // SKIP LOCKED may leave a job for a later call, never hand it out twice.
    while let Some(job) = db.store.claim_job("drain", 60).await? {
        assert!(!unique.contains(&job.id));
        claimed.push(job.id);
    }
    assert_eq!(claimed.len(), 10);
    assert_eq!(claimed.iter().copied().collect::<BTreeSet<_>>().len(), 10);
    Ok(())
}

#[tokio::test]
async fn two_concurrent_claimers_of_one_job_produce_one_winner() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let (a, b) = tokio::join!(db.store.claim_job("a", 60), db.store.claim_job("b", 60));
    let winners = [a?, b?].into_iter().flatten().count();
    assert_eq!(winners, 1);
    Ok(())
}

#[tokio::test]
async fn expired_lease_is_reclaimed_with_attempts_incremented() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let first = db.store.claim_job("crashed", 1).await?.expect("job");
    assert_eq!(first.attempts, 1);
    assert!(db.store.claim_job("early", 60).await?.is_none());

    tokio::time::sleep(Duration::from_millis(1200)).await;
    let second = db.store.claim_job("rescuer", 60).await?.expect("reclaimed");
    assert_eq!(second.id, first.id);
    assert_eq!(second.attempts, 2);
    let holder: String = sqlx::query_scalar("SELECT locked_by FROM jobs WHERE id = $1")
        .bind(second.id)
        .fetch_one(&db.admin)
        .await?;
    assert_eq!(holder, "rescuer");
    Ok(())
}

#[tokio::test]
async fn fail_job_reschedules_then_goes_dead_at_the_limit() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let job = db.store.claim_job("w", 60).await?.expect("job");

    // Future run_after: not claimable yet.
    let later = Utc::now() + chrono::Duration::hours(1);
    assert!(db.store.fail_job(&job, "w", "boom\0", later, 2).await?);
    assert!(db.store.claim_job("w", 60).await?.is_none());
    let row = sqlx::query("SELECT status, last_error, locked_by FROM jobs WHERE id = $1")
        .bind(job.id)
        .fetch_one(&db.admin)
        .await?;
    assert_eq!(row.try_get::<String, _>("status")?, "pending");
    assert_eq!(row.try_get::<String, _>("last_error")?, "boom");
    assert_eq!(row.try_get::<Option<String>, _>("locked_by")?, None);

    // Due again: claimed for attempt 2, which reaches the limit of 2 on failure.
    let past = Utc::now() - chrono::Duration::hours(1);
    sqlx::query("UPDATE jobs SET run_after = $1")
        .bind(past)
        .execute(&db.admin)
        .await?;
    let retry = db.store.claim_job("w", 60).await?.expect("retry");
    assert_eq!(retry.attempts, 2);
    assert!(db.store.fail_job(&retry, "w", "boom", past, 2).await?);
    let status: String = sqlx::query_scalar("SELECT status FROM jobs WHERE id = $1")
        .bind(job.id)
        .fetch_one(&db.admin)
        .await?;
    assert_eq!(status, "dead");
    assert!(db.store.claim_job("w", 60).await?.is_none());

    Ok(())
}

async fn job_row(db: &Db, id: Uuid) -> Res<(String, Option<String>, Option<String>)> {
    let row = sqlx::query("SELECT status, locked_by, last_error FROM jobs WHERE id = $1")
        .bind(id)
        .fetch_one(&db.admin)
        .await?;
    Ok((
        row.try_get("status")?,
        row.try_get("locked_by")?,
        row.try_get("last_error")?,
    ))
}

#[tokio::test]
async fn completed_job_is_not_claimed_or_revived_by_a_late_failure() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let job = db.store.claim_job("w", 60).await?.expect("job");
    assert!(db.store.complete_job(&job, "w").await?);
    assert!(db.store.claim_job("w", 60).await?.is_none());

    // The same worker, after its claim is finished: every further call reports a lost claim.
    assert!(!db.store.complete_job(&job, "w").await?);
    assert!(!db.store.renew_job(&job, "w", 60).await?);
    assert!(!db.store.fail_job(&job, "w", "late", Utc::now(), 5).await?);
    assert_eq!(job_row(&db, job.id).await?.0, "done");
    assert!(db.store.claim_job("w", 60).await?.is_none());
    Ok(())
}

#[tokio::test]
async fn stale_worker_cannot_touch_a_job_another_worker_reclaimed() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let stale = db.store.claim_job("old", 1).await?.expect("job");
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let current = db.store.claim_job("new", 60).await?.expect("reclaimed");
    assert_eq!(current.id, stale.id);

    let later = Utc::now() + chrono::Duration::hours(1);
    assert!(!db.store.complete_job(&stale, "old").await?);
    assert!(!db.store.fail_job(&stale, "old", "x", later, 1).await?);
    assert!(!db.store.renew_job(&stale, "old", 3600).await?);
    assert_eq!(
        job_row(&db, stale.id).await?,
        ("running".to_owned(), Some("new".to_owned()), None)
    );

    // Each part of the fence holds alone: the current worker with the old attempts, and the old
    // worker with the current attempts.
    assert!(!db.store.complete_job(&stale, "new").await?);
    assert!(!db.store.complete_job(&current, "old").await?);
    assert!(!db.store.fail_job(&current, "old", "x", later, 1).await?);
    assert_eq!(job_row(&db, stale.id).await?.0, "running");

    assert!(db.store.complete_job(&current, "new").await?);
    assert_eq!(job_row(&db, stale.id).await?.0, "done");
    Ok(())
}

#[tokio::test]
async fn same_worker_id_reclaiming_is_still_fenced_by_attempts() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let stale = db.store.claim_job("w", 1).await?.expect("job");
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let current = db.store.claim_job("w", 60).await?.expect("reclaimed");
    assert!(!db.store.complete_job(&stale, "w").await?);
    assert!(db.store.complete_job(&current, "w").await?);
    Ok(())
}

#[tokio::test]
async fn renew_job_extends_the_lease_so_it_cannot_be_reclaimed() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let job = db.store.claim_job("w", 1).await?.expect("job");
    assert!(db.store.renew_job(&job, "w", 60).await?);
    assert!(is_invalid(db.store.renew_job(&job, "w", 0).await));
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert!(db.store.claim_job("other", 60).await?.is_none());
    assert!(db.store.complete_job(&job, "w").await?);
    Ok(())
}

#[tokio::test]
async fn lost_claim_on_a_missing_job_is_false() -> TestResult {
    let db = db().await;
    let ghost = memoryos_core::Job {
        id: Uuid::now_v7(),
        tenant: "t1".into(),
        kind: "extract".into(),
        payload: json!({}),
        attempts: 1,
    };
    assert!(!db.store.renew_job(&ghost, "w", 60).await?);
    assert!(!db.store.complete_job(&ghost, "w").await?);
    assert!(!db.store.fail_job(&ghost, "w", "x", Utc::now(), 5).await?);
    Ok(())
}

// ---- API keys --------------------------------------------------------------------------------

#[tokio::test]
async fn api_key_lookup_stops_after_revoke() -> TestResult {
    let db = db().await;
    let hash = "ab".repeat(32);
    let scopes = vec!["user:*".to_owned()];
    let key = db
        .store
        .create_api_key("t1", "ci", &hash, &scopes, Trust::Agent, true)
        .await?;
    assert_eq!(key.tenant, "t1");
    assert_eq!(key.max_trust, Trust::Agent);
    assert!(key.is_admin);

    let found = db.store.find_api_key(&hash).await?.expect("key");
    assert_eq!(found.id, key.id);
    assert_eq!(found.scopes, scopes);
    assert!(db.store.find_api_key(&"cd".repeat(32)).await?.is_none());

    assert!(is_invalid(
        db.store
            .create_api_key("t1", "dup", &hash, &scopes, Trust::Agent, false)
            .await
    ));
    assert!(is_invalid(
        db.store
            .create_api_key("t1", "raw", "mos_token", &scopes, Trust::Agent, false)
            .await
    ));

    db.store.revoke_api_key(key.id).await?;
    assert!(db.store.find_api_key(&hash).await?.is_none());
    db.store.revoke_api_key(key.id).await?;
    assert!(is_not_found(db.store.revoke_api_key(Uuid::now_v7()).await));
    Ok(())
}

// ---- Tenant isolation ------------------------------------------------------------------------

#[tokio::test]
async fn tenant_b_cannot_read_or_change_any_table_family_of_tenant_a() -> TestResult {
    let db = db().await;
    let a = ctx("tenant-a");
    let b = ctx("tenant-b");
    let scope = user("shared-id");
    let scopes = [scope.clone()];
    let filter = FactFilter::default();

    // Tenant A data, in the scope names and agent ids tenant B will also use.
    let subject = entity(&db, &a, &scope, "ada").await?;
    let object = entity(&db, &a, &scope, "bob").await?;
    let source = db
        .store
        .insert_episode(&a, episode(&scope, Trust::User, "secret words"))
        .await?;
    let mut secret = new_fact(&scope, subject, "secret words", date(2020, 1));
    secret.object_id = Some(object);
    let fact = db
        .store
        .apply_extraction(&a, source.id, vec![FactOp::Add(secret)])
        .await?[0];
    let pending = db
        .store
        .insert_episode(&a, episode(&scope, Trust::User, "secret pending"))
        .await?;
    let external = new_episode(&db, &a, &scope, Trust::External).await?;
    let mut held = new_fact(&scope, subject, "held", date(2020, 1));
    held.trust = Trust::External;
    held.status = FactStatus::Quarantined;
    let quarantined = db
        .store
        .apply_extraction(&a, external, vec![FactOp::Add(held)])
        .await?[0];
    db.store
        .put_block(&a, "bot", "persona", "secret", 100, None)
        .await?;
    let session = db.store.create_session(&a, "bot", &scope).await?;
    db.store
        .append_message(&a, session.id, "user", json!(["secret"]))
        .await?;
    db.store
        .compact_session(&a, session.id, 0, "secret summary")
        .await?;
    db.store
        .invalidate_fact(&a, quarantined, "audit me")
        .await?;

    // Reads by id.
    assert!(is_not_found(db.store.get_episode(&b, source.id).await));
    assert!(is_not_found(db.store.get_entity(&b, subject).await));
    assert!(is_not_found(db.store.get_fact(&b, fact).await));
    assert!(is_not_found(db.store.fact_history(&b, fact).await));
    assert!(is_not_found(db.store.get_session(&b, session.id).await));
    assert!(is_not_found(db.store.load_messages(&b, session.id).await));
    assert!(is_not_found(db.store.session_summary(&b, session.id).await));
    assert!(is_not_found(db.store.get_block(&b, "bot", "persona").await));
    assert!(is_not_found(
        db.store.block_versions(&b, "bot", "persona").await
    ));
    assert!(is_not_found(
        db.store
            .neighbors(&b, &scopes, subject, 2, &filter, 5000)
            .await
    ));

    // Reads by search, with the same scope string.
    assert!(db
        .store
        .search_facts_text(&b, &scopes, "secret", &filter, 10)
        .await?
        .is_empty());
    assert!(db
        .store
        .search_facts_vector(&b, &scopes, &axis(0), &filter, 10)
        .await?
        .is_empty());
    assert!(db
        .store
        .search_pending_episodes(&b, &scopes, "secret", 10)
        .await?
        .is_empty());
    assert!(db
        .store
        .similar_entities(&b, &scope, &axis(1), 10)
        .await?
        .is_empty());
    assert!(db
        .store
        .current_facts_for_subject(&b, &scope, subject)
        .await?
        .is_empty());
    assert!(db.store.list_quarantined(&b, &scopes, 10).await?.is_empty());
    assert!(db.store.list_blocks(&b, "bot").await?.is_empty());

    // Writes against A's rows.
    assert!(is_not_found(db.store.approve_fact(&b, quarantined).await));
    assert!(is_not_found(db.store.reject_fact(&b, quarantined).await));
    assert!(is_not_found(db.store.invalidate_fact(&b, fact, "x").await));
    assert!(is_not_found(db.store.purge_episode(&b, source.id).await));
    assert!(is_not_found(
        db.store.add_entity_alias(&b, subject, "x", None).await
    ));
    assert!(is_not_found(
        db.store
            .append_message(&b, session.id, "user", json!([]))
            .await
    ));
    assert!(is_not_found(
        db.store.compact_session(&b, session.id, 0, "x").await
    ));
    assert!(is_not_found(
        db.store
            .put_block(&b, "bot", "persona", "x", 100, Some(1))
            .await
    ));
    assert!(is_not_found(
        db.store.apply_extraction(&b, pending.id, vec![]).await
    ));

    // Foreign keys bypass RLS, so these are stopped by the store's own tenant-scoped reads.
    let b_episode = new_episode(&db, &b, &scope, Trust::User).await?;
    let b_subject = entity(&db, &b, &scope, "own").await?;
    let foreign_ops = [
        FactOp::Add(new_fact(&scope, subject, "foreign subject", date(2020, 1))),
        FactOp::Add({
            let mut f = new_fact(&scope, b_subject, "foreign object", date(2020, 1));
            f.object_id = Some(object);
            f
        }),
        FactOp::Supersede {
            old_fact_id: fact,
            new: new_fact(&scope, b_subject, "foreign supersede", date(2021, 1)),
        },
        FactOp::Corroborate { fact_id: fact },
    ];
    for op in foreign_ops {
        assert!(is_not_found(
            db.store.apply_extraction(&b, b_episode, vec![op]).await
        ));
    }

    // purge_scope in B with the same scope leaves A untouched.
    db.store.purge_scope(&b, &scope).await?;
    db.store
        .put_block(&b, "bot", "persona", "own", 100, None)
        .await?;

    let still = db.store.get_fact(&a, fact).await?;
    assert_eq!(still.invalidated_at, None);
    assert_eq!(still.source_episode_ids, vec![source.id]);
    assert_eq!(
        db.store.get_block(&a, "bot", "persona").await?.content,
        "secret"
    );
    assert_eq!(
        db.store.get_entity(&a, subject).await?.aliases,
        Vec::<String>::new()
    );
    assert_eq!(
        db.store.session_summary(&a, session.id).await?.as_deref(),
        Some("secret summary")
    );
    assert_eq!(db.store.load_messages(&a, session.id).await?.len(), 1);
    assert_eq!(
        db.store.get_fact(&a, quarantined).await?.status,
        FactStatus::Quarantined
    );
    Ok(())
}

/// Direct SQL through the same role switch, for the tables the trait cannot read back.
#[tokio::test]
async fn row_level_security_holds_under_the_superuser_login() -> TestResult {
    let db = db().await;
    let a = ctx("tenant-a");
    let scope = user("u1");
    let subject = entity(&db, &a, &scope, "ada").await?;
    let (_, fact) = add_fact(&db, &a, new_fact(&scope, subject, "f", date(2020, 1))).await?;
    db.store.invalidate_fact(&a, fact, "audit").await?;
    db.store.put_block(&a, "bot", "p", "x", 10, None).await?;
    let session = db.store.create_session(&a, "bot", &scope).await?;
    db.store
        .append_message(&a, session.id, "user", json!([]))
        .await?;

    let tables = [
        "episodes",
        "entities",
        "facts",
        "fact_sources",
        "blocks",
        "block_versions",
        "sessions",
        "session_messages",
        "audit_events",
    ];
    for table in tables {
        let select = format!("SELECT count(*) FROM {table}");
        // The superuser login sees the row: the data exists and the tests below are not vacuous.
        let total: i64 = sqlx::query_scalar(&select).fetch_one(&db.admin).await?;
        assert!(total >= 1, "{table} is empty");

        for (tenant, expected) in [(Some("tenant-b"), 0), (None, 0), (Some("tenant-a"), total)] {
            let mut tx = db.admin.begin().await?;
            sqlx::query("SET LOCAL ROLE memoryos_app")
                .execute(&mut *tx)
                .await?;
            if let Some(tenant) = tenant {
                sqlx::query("SELECT set_config('app.tenant', $1, true)")
                    .bind(tenant)
                    .execute(&mut *tx)
                    .await?;
            }
            let seen: i64 = sqlx::query_scalar(&select).fetch_one(&mut *tx).await?;
            assert_eq!(seen, expected, "{table} as {tenant:?}");
            tx.rollback().await?;
        }
    }

    // Cross-tenant insert is rejected by WITH CHECK, and jobs and api_keys are off limits.
    let mut tx = db.admin.begin().await?;
    sqlx::query("SET LOCAL ROLE memoryos_app")
        .execute(&mut *tx)
        .await?;
    sqlx::query("SELECT set_config('app.tenant', 'tenant-b', true)")
        .execute(&mut *tx)
        .await?;
    let forged = sqlx::query(
        "INSERT INTO sessions (id, tenant, agent_id, scope) VALUES ($1, 'tenant-a', 'x', 'user:x')",
    )
    .bind(Uuid::now_v7())
    .execute(&mut *tx)
    .await;
    let code = forged
        .expect_err("forged insert must fail")
        .into_database_error()
        .and_then(|e| e.code().map(|c| c.into_owned()));
    assert_eq!(code.as_deref(), Some("42501"));
    tx.rollback().await?;

    for table in ["jobs", "api_keys"] {
        let mut tx = db.admin.begin().await?;
        sqlx::query("SET LOCAL ROLE memoryos_app")
            .execute(&mut *tx)
            .await?;
        let denied = sqlx::query(&format!("SELECT count(*) FROM {table}"))
            .execute(&mut *tx)
            .await;
        assert!(denied.is_err(), "{table} must be denied");
        tx.rollback().await?;
    }
    Ok(())
}

/// Calls both search functions as memoryos_app. `tenant = None` leaves app.tenant unset.
async fn search_ids_as_app(
    pool: &PgPool,
    tenant: Option<&str>,
    scope: &str,
    word: &str,
) -> Res<(Vec<Uuid>, Vec<Uuid>)> {
    let mut tx = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE memoryos_app")
        .execute(&mut *tx)
        .await?;
    if let Some(tenant) = tenant {
        sqlx::query("SELECT set_config('app.tenant', $1, true)")
            .bind(tenant)
            .execute(&mut *tx)
            .await?;
    }
    let scopes = vec![scope.to_owned()];
    let facts: Vec<Uuid> =
        sqlx::query_scalar("SELECT id FROM search_fact_ids_text($1, $2, now(), now(), 10)")
            .bind(&scopes)
            .bind(word)
            .fetch_all(&mut *tx)
            .await?;
    let episodes: Vec<Uuid> =
        sqlx::query_scalar("SELECT id FROM search_pending_episode_ids($1, $2, 10)")
            .bind(&scopes)
            .bind(word)
            .fetch_all(&mut *tx)
            .await?;
    tx.rollback().await?;
    Ok((facts, episodes))
}

/// A fact and a pending episode for `a`, both containing "secret". Returns (fact, episode).
async fn seed_secret(store: &PgStore, a: &RequestContext, scope: &Scope) -> Res<(Uuid, Uuid)> {
    let subject = store
        .upsert_entity(
            a,
            NewEntity {
                scope: scope.clone(),
                kind: "thing".into(),
                name: "ada".into(),
                aliases: vec![],
                embedding: axis(1),
            },
        )
        .await?
        .id;
    let source = store
        .insert_episode(a, episode(scope, Trust::User, "processed"))
        .await?
        .id;
    let fact = store
        .apply_extraction(
            a,
            source,
            vec![FactOp::Add(new_fact(
                scope,
                subject,
                "secret fact",
                date(2020, 1),
            ))],
        )
        .await?[0];
    let pending = store
        .insert_episode(a, episode(scope, Trust::User, "secret pending"))
        .await?
        .id;
    Ok((fact, pending))
}

#[tokio::test]
async fn search_functions_return_nothing_for_another_or_unset_tenant() -> TestResult {
    let db = db().await;
    let a = ctx("tenant-a");
    let scope = user("u1");
    let (fact, pending) = seed_secret(&db.store, &a, &scope).await?;
    let scope_text = scope.to_string();

    let own = search_ids_as_app(&db.admin, Some("tenant-a"), &scope_text, "secret").await?;
    assert_eq!(own, (vec![fact], vec![pending]));

    for tenant in [Some("tenant-b"), Some(""), Some("%"), None] {
        let seen = search_ids_as_app(&db.admin, tenant, &scope_text, "secret").await?;
        assert_eq!(seen, (vec![], vec![]), "tenant {tenant:?}");
    }

    // Through the store as well, with tenant B holding its own row under the same scope name.
    let b = ctx("tenant-b");
    let (b_fact, b_pending) = seed_secret(&db.store, &b, &scope).await?;
    let filter = FactFilter::default();
    let scopes = [scope];
    let facts = db
        .store
        .search_facts_text(&b, &scopes, "secret", &filter, 10)
        .await?;
    assert_eq!(
        facts.iter().map(|s| s.fact.id).collect::<Vec<_>>(),
        vec![b_fact]
    );
    let episodes = db
        .store
        .search_pending_episodes(&b, &scopes, "secret", 10)
        .await?;
    assert_eq!(
        episodes.iter().map(|s| s.episode.id).collect::<Vec<_>>(),
        vec![b_pending]
    );
    Ok(())
}

#[tokio::test]
async fn text_search_keeps_time_and_status_filters_through_the_function() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(
        &db,
        &c,
        new_fact(&scope, subject, "marker old", date(2020, 1)),
    )
    .await?;
    let new = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "marker new", date(2022, 1)),
    )
    .await?;
    let mut held = new_fact(&scope, subject, "marker held", date(2020, 1));
    held.trust = Trust::External;
    held.status = FactStatus::Quarantined;
    add_fact(&db, &c, held).await?;

    let search = |filter: FactFilter| {
        let store = db.store.clone();
        let c = c.clone();
        let scopes = scopes.clone();
        async move {
            store
                .search_facts_text(&c, &scopes, "marker", &filter, 10)
                .await
                .map(|found| found.into_iter().map(|s| s.fact.id).collect::<Vec<_>>())
        }
    };
    assert_eq!(search(FactFilter::default()).await?, vec![new]);
    assert_eq!(search(at(date(2021, 1))).await?, vec![old]);
    assert_eq!(search(at(date(2019, 1))).await?, Vec::<Uuid>::new());
    let recorded = db.store.get_fact(&c, new).await?.recorded_at;
    let before = FactFilter {
        as_of: Some(date(2023, 1)),
        known_at: Some(recorded - chrono::Duration::seconds(1)),
    };
    assert_eq!(search(before).await?, Vec::<Uuid>::new());
    db.store.invalidate_fact(&c, new, "wrong").await?;
    assert_eq!(search(FactFilter::default()).await?, vec![old]);
    Ok(())
}

/// The production shape: a non-superuser login owns the tables and is granted memoryos_app. A
/// superuser only prepares the role and the extension, as a managed database would.
#[tokio::test]
async fn non_superuser_owner_login_runs_migration_tenancy_search_and_jobs() -> TestResult {
    let server = server_url();
    // The template migration created memoryos_app, which is cluster-wide.
    let bootstrap = connect_with_retry(&format!("{server}/postgres")).await;
    for sql in [
        "CREATE ROLE svc LOGIN PASSWORD 'pw' NOSUPERUSER NOBYPASSRLS",
        "GRANT memoryos_app TO svc",
        "CREATE DATABASE owned OWNER svc",
    ] {
        sqlx::query(sql).execute(&bootstrap).await?;
    }
    bootstrap.close().await;
    let superuser = connect_with_retry(&format!("{server}/owned")).await;
    sqlx::query("CREATE EXTENSION vector")
        .execute(&superuser)
        .await?;

    let host = server
        .strip_prefix("postgres://postgres:postgres@")
        .expect("server url");
    let url = format!("postgres://svc:pw@{host}/owned");
    let store = PgStore::connect(&url, 8).await?;
    store.migrate().await?;

    let owner = connect_with_retry(&url).await;
    let is_super: bool =
        sqlx::query_scalar("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
            .fetch_one(&owner)
            .await?;
    assert!(!is_super);
    let table_owner: String =
        sqlx::query_scalar("SELECT tableowner::text FROM pg_tables WHERE tablename = 'facts'")
            .fetch_one(&owner)
            .await?;
    assert_eq!(table_owner, "svc");

    let a = ctx("tenant-a");
    let b = ctx("tenant-b");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let filter = FactFilter::default();
    let (fact, pending) = seed_secret(&store, &a, &scope).await?;

    let found = store
        .search_facts_text(&a, &scopes, "secret", &filter, 10)
        .await?;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].fact.id, fact);
    assert!(found[0].score > 0.0);
    let found = store
        .search_pending_episodes(&a, &scopes, "secret", 10)
        .await?;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].episode.id, pending);

    assert!(store
        .search_facts_text(&b, &scopes, "secret", &filter, 10)
        .await?
        .is_empty());
    assert!(store
        .search_pending_episodes(&b, &scopes, "secret", 10)
        .await?
        .is_empty());
    assert!(is_not_found(store.get_fact(&b, fact).await));
    let nothing = (vec![], vec![]);
    let scope_text = scope.to_string();
    assert_eq!(
        search_ids_as_app(&owner, Some("tenant-b"), &scope_text, "secret").await?,
        nothing
    );
    assert_eq!(
        search_ids_as_app(&owner, None, &scope_text, "secret").await?,
        nothing
    );

    // The owner calling the function directly, outside the role, is held to app.tenant too.
    let direct: Vec<Uuid> =
        sqlx::query_scalar("SELECT id FROM search_fact_ids_text($1, 'secret', now(), now(), 10)")
            .bind(vec![scope_text])
            .fetch_all(&owner)
            .await?;
    assert!(direct.is_empty());

    // Writes, audit, enqueue_job and the job methods all work under this login.
    store.invalidate_fact(&a, fact, "audit").await?;
    store.put_block(&a, "bot", "persona", "p", 10, None).await?;
    let job = store.claim_job("w", 60).await?.expect("job");
    assert!(store.complete_job(&job, "w").await?);
    let report = store.purge_scope(&a, &scope).await?;
    assert_eq!(report.facts_deleted, 1);
    Ok(())
}

#[tokio::test]
async fn empty_tenant_is_rejected_before_any_query() -> TestResult {
    let db = db().await;
    assert!(is_invalid(db.store.list_blocks(&ctx(""), "a").await));
    Ok(())
}

// ---- Quarantine prefixes -----------------------------------------------------------------------

#[tokio::test]
async fn quarantine_prefixes_match_literally_and_stay_in_the_tenant() -> TestResult {
    let db = db().await;
    let c = ctx("a");
    let mut ids = Vec::new();
    for name in ["a_1", "axb", "b"] {
        let scope = user(name);
        let subject = entity(&db, &c, &scope, "Ada").await?;
        let mut fact = new_fact(&scope, subject, "held", date(2022, 1));
        fact.trust = Trust::External;
        fact.status = FactStatus::Quarantined;
        ids.push(add_fact(&db, &c, fact).await?.1);
    }
    let listed = |exact: Vec<Scope>, prefixes: Vec<&'static str>, tenant: &'static str| {
        let store = db.store.clone();
        async move {
            let prefixes: Vec<String> = prefixes.into_iter().map(str::to_owned).collect();
            let found = store
                .list_quarantined_matching(&ctx(tenant), &exact, &prefixes, 10)
                .await?;
            Ok::<_, Error>(found.iter().map(|fact| fact.id).collect::<Vec<_>>())
        }
    };

    assert_eq!(listed(vec![], vec!["user:"], "a").await?, ids);
    assert_eq!(listed(vec![], vec![""], "a").await?, ids);
    // `_` is an ordinary character here. As a LIKE pattern it would also match `user:axb`.
    assert_eq!(listed(vec![], vec!["user:a_"], "a").await?, ids[..1]);
    assert_eq!(listed(vec![user("b")], vec![], "a").await?, ids[2..]);
    assert_eq!(
        listed(vec![user("b")], vec!["user:a_"], "a").await?,
        [ids[0], ids[2]]
    );
    assert!(listed(vec![], vec![], "a").await?.is_empty());
    assert!(listed(vec![], vec![""], "b").await?.is_empty());
    Ok(())
}
