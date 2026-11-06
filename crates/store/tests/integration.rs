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
use testcontainers::core::{IntoContainerPort, WaitFor};
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
async fn corroborate_links_a_second_source_without_raising_trust() -> TestResult {
    let db = db().await;
    let c = ctx("t1");
    let scope = user("u1");
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
    assert!(db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?
        .is_empty());
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
    let pending: Option<Uuid> =
        sqlx::query_scalar("SELECT pending_supersede_of FROM facts WHERE id = $1")
            .bind(new)
            .fetch_one(&db.admin)
            .await?;
    assert_eq!(pending, Some(old));

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
            .neighbors(&c, &scopes, subject, 1, &no_filter)
            .await?
            .edges
            .len(),
        1
    );
    let listed = db.store.list_quarantined(&c, &scopes, 10).await?;
    assert_eq!(listed.iter().map(|f| f.id).collect::<Vec<_>>(), vec![new]);

    let approved = db.store.approve_fact(&c, new).await?;
    assert_eq!(approved.status, FactStatus::Active);
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
async fn pending_episode_search_excludes_external_and_processed() -> TestResult {
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
        .insert_episode(&c, episode(&user("u2"), Trust::User, "the quick other"))
        .await?;

    let found = db
        .store
        .search_pending_episodes(&c, &scopes, "quick", 10)
        .await?;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].episode.id, trusted.id);
    assert!(found[0].score > 0.0);

    db.store.apply_extraction(&c, trusted.id, vec![]).await?;
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

    db.store.add_entity_alias(&c, first.id, "Countess").await?;
    db.store.add_entity_alias(&c, first.id, "Countess").await?;
    assert_eq!(
        db.store.get_entity(&c, first.id).await?.aliases,
        vec!["Countess".to_owned()]
    );
    assert!(is_not_found(
        db.store.add_entity_alias(&c, Uuid::now_v7(), "x").await
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
    let one = db.store.neighbors(&c, &scopes, a, 1, &filter).await?;
    assert_eq!(one.root, a);
    assert_eq!(entity_ids(&one), BTreeSet::from([a, b, d]));
    assert_eq!(edge_ids(&one), BTreeSet::from([ab, da]));

    let two = db.store.neighbors(&c, &scopes, a, 2, &filter).await?;
    assert_eq!(entity_ids(&two), BTreeSet::from([a, b, d, e]));
    assert_eq!(edge_ids(&two), BTreeSet::from([ab, da, bd, de]));

    let five = db.store.neighbors(&c, &scopes, a, 5, &filter).await?;
    assert_eq!(entity_ids(&five), BTreeSet::from([a, b, d, e]));
    assert_eq!(edge_ids(&five), BTreeSet::from([ab, bd, da, de, ee]));
    assert_eq!(five.edges.len(), 5, "no duplicates after a cycle");

    for bad in [0, 6, -1] {
        assert!(is_invalid(
            db.store.neighbors(&c, &scopes, a, bad, &filter).await
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
        .neighbors(&c, &scopes, a, 2, &FactFilter::default())
        .await?;
    assert_eq!(
        current.edges.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![new_edge]
    );
    assert!(current.entities.iter().all(|x| x.id != b));

    let past = db
        .store
        .neighbors(&c, &scopes, a, 2, &at(date(2021, 1)))
        .await?;
    assert_eq!(
        past.edges.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![old_edge]
    );

    // Retraction removes an edge from the graph as well.
    db.store.invalidate_fact(&c, new_edge, "wrong").await?;
    assert!(db
        .store
        .neighbors(&c, &scopes, a, 2, &FactFilter::default())
        .await?
        .edges
        .is_empty());

    // A root outside the caller's scopes is indistinguishable from a missing one.
    assert!(is_not_found(
        db.store
            .neighbors(&c, &[user("u2")], a, 1, &FactFilter::default())
            .await
    ));
    assert!(is_not_found(
        db.store
            .neighbors(&c, &scopes, Uuid::now_v7(), 1, &FactFilter::default())
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
    db.store.fail_job(job.id, "boom\0", later, 2).await?;
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
    db.store.fail_job(retry.id, "boom", past, 2).await?;
    let status: String = sqlx::query_scalar("SELECT status FROM jobs WHERE id = $1")
        .bind(job.id)
        .fetch_one(&db.admin)
        .await?;
    assert_eq!(status, "dead");
    assert!(db.store.claim_job("w", 60).await?.is_none());

    assert!(is_not_found(
        db.store.fail_job(Uuid::now_v7(), "x", past, 2).await
    ));
    assert!(is_not_found(db.store.complete_job(Uuid::now_v7()).await));
    Ok(())
}

#[tokio::test]
async fn completed_job_is_not_claimed_or_revived_by_a_late_failure() -> TestResult {
    let db = db().await;
    new_episode(&db, &ctx("t1"), &user("u1"), Trust::User).await?;
    let job = db.store.claim_job("w", 60).await?.expect("job");
    db.store.complete_job(job.id).await?;
    assert!(db.store.claim_job("w", 60).await?.is_none());

    db.store
        .fail_job(job.id, "stale worker", Utc::now(), 5)
        .await?;
    let status: String = sqlx::query_scalar("SELECT status FROM jobs WHERE id = $1")
        .bind(job.id)
        .fetch_one(&db.admin)
        .await?;
    assert_eq!(status, "done");
    assert!(db.store.claim_job("w", 60).await?.is_none());
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
        db.store.neighbors(&b, &scopes, subject, 2, &filter).await
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
        db.store.add_entity_alias(&b, subject, "x").await
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

#[tokio::test]
async fn empty_tenant_is_rejected_before_any_query() -> TestResult {
    let db = db().await;
    assert!(is_invalid(db.store.list_blocks(&ctx(""), "a").await));
    Ok(())
}
