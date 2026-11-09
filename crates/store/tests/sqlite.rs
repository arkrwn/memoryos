//! Real-file SQLite integration tests for `SqliteStore`.
//!
//! Every test owns a temporary directory and database file. The suite exercises the same public
//! `MemoryStore` contract as the PostgreSQL integration suite, without Docker or direct SQL.

#![cfg(feature = "integration-sqlite")]

use std::collections::BTreeSet;
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use memoryos_core::{
    Error, Fact, FactFilter, FactListState, FactOp, FactStatus, MemoryStore, NewEntity, NewEpisode,
    NewFact, RequestContext, Scope, ScoredFact, Trust, EMBEDDING_DIMS,
};
use memoryos_store::SqliteStore;
use serde_json::json;
use tempfile::TempDir;
use uuid::Uuid;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// `claim_episode` for tests that do not need to distinguish creation from a replay.
trait InsertEpisode {
    async fn insert_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> memoryos_core::Result<memoryos_core::Episode>;
}

impl InsertEpisode for SqliteStore {
    async fn insert_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> memoryos_core::Result<memoryos_core::Episode> {
        self.claim_episode(ctx, episode).await.map(|(row, _)| row)
    }
}

struct Db {
    // Dropped before `_dir`, so SQLite has closed its file handles before cleanup.
    store: Arc<SqliteStore>,
    path: String,
    _dir: TempDir,
}

async fn db() -> Res<Db> {
    let dir = tempfile::tempdir()?;
    // The directory is unique per test; a fixed file name keeps connection strings simple.
    let path = dir
        .path()
        .join("memoryos.sqlite3")
        .to_string_lossy()
        .into_owned();
    let store = SqliteStore::connect(&path).await?;
    store.migrate().await?;
    Ok(Db {
        store: Arc::new(store),
        path,
        _dir: dir,
    })
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

/// A unit vector on one axis. Distinct axes have cosine similarity zero.
fn axis(i: usize) -> Vec<f32> {
    assert!(i < EMBEDDING_DIMS);
    let mut vector = vec![0.0; EMBEDDING_DIMS];
    vector[i] = 1.0;
    vector
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
        expires_at: None,
    }
}

fn with_trust(mut fact: NewFact, trust: Trust) -> NewFact {
    fact.trust = trust;
    fact
}

fn at(as_of: DateTime<Utc>) -> FactFilter {
    FactFilter {
        as_of: Some(as_of),
        known_at: None,
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
        .apply_commit(c, episode_id, vec![FactOp::Add(fact)])
        .await?;
    assert_eq!(ids.len(), 1);
    Ok((episode_id, ids[0]))
}

async fn supersede(db: &Db, c: &RequestContext, old_fact_id: Uuid, new: NewFact) -> Res<Uuid> {
    let episode_id = new_episode(db, c, &new.scope, new.trust).await?;
    let ids = db
        .store
        .apply_commit(c, episode_id, vec![FactOp::Supersede { old_fact_id, new }])
        .await?;
    assert_eq!(ids.len(), 1);
    Ok(ids[0])
}

async fn corroborate(
    db: &Db,
    c: &RequestContext,
    scope: &Scope,
    trust: Trust,
    fact_id: Uuid,
) -> Res<Uuid> {
    let episode_id = new_episode(db, c, scope, trust).await?;
    db.store
        .apply_commit(c, episode_id, vec![FactOp::Corroborate { fact_id }])
        .await?;
    Ok(episode_id)
}

/// Applies one graph edge fact with its own episode.
async fn edge(db: &Db, c: &RequestContext, scope: &Scope, a: Uuid, b: Uuid) -> Res<Uuid> {
    let mut fact = new_fact(scope, a, "edge", date(2020, 1));
    fact.predicate = "links".into();
    fact.object_id = Some(b);
    Ok(add_fact(db, c, fact).await?.1)
}

fn scored_ids(found: Vec<ScoredFact>) -> Vec<Uuid> {
    found.into_iter().map(|hit| hit.fact.id).collect()
}

fn fact_ids(facts: &[Fact]) -> Vec<Uuid> {
    facts.iter().map(|fact| fact.id).collect()
}

fn is_not_found<T>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::NotFound(_)))
}

fn is_invalid<T>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::InvalidInput(_)))
}

// ---- Connectivity, migrations and persistence -------------------------------------------------

#[tokio::test]
async fn fresh_and_repeat_migration_succeed_and_data_survives_restart() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    db.store.ping().await?;
    db.store.migrate().await?;

    let subject = entity(&db, &c, &scope, "Ada").await?;
    let (source, fact) = add_fact(
        &db,
        &c,
        new_fact(&scope, subject, "persistent", date(2020, 1)),
    )
    .await?;
    db.store
        .put_block(&c, "bot", "persona", "persistent block", 100, None)
        .await?;
    let hash = "ab".repeat(32);
    let key = db
        .store
        .create_api_key(
            "t1",
            "restart",
            &hash,
            &["user:*".into()],
            Trust::User,
            true,
        )
        .await?;

    let path = db.path.clone();
    drop(db.store);
    let reopened = SqliteStore::connect(&path).await?;
    reopened.migrate().await?;
    reopened.ping().await?;
    assert_eq!(reopened.get_episode(&c, source).await?.id, source);
    assert_eq!(reopened.get_entity(&c, subject).await?.name, "Ada");
    assert_eq!(reopened.get_fact(&c, fact).await?.statement, "persistent");
    assert_eq!(
        reopened.get_block(&c, "bot", "persona").await?.content,
        "persistent block"
    );
    assert_eq!(reopened.find_api_key(&hash).await?.expect("key").id, key.id);
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn sqlite_creates_private_files_without_changing_existing_parent_directories() -> TestResult {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir()?;
    let existing = dir.path().join("existing");
    std::fs::create_dir(&existing)?;
    std::fs::set_permissions(&existing, std::fs::Permissions::from_mode(0o755))?;
    let nested = existing.join("nested");
    let path = nested.join("memoryos.db");
    let url = path.to_string_lossy().into_owned();
    let store = SqliteStore::connect(&url).await?;
    store.migrate().await?;
    assert_eq!(existing.metadata()?.permissions().mode() & 0o777, 0o755);
    assert_eq!(nested.metadata()?.permissions().mode() & 0o777, 0o700);
    assert_eq!(path.metadata()?.permissions().mode() & 0o777, 0o600);
    for suffix in ["-wal", "-shm"] {
        let auxiliary = std::path::PathBuf::from(format!("{}{suffix}", path.display()));
        if auxiliary.exists() {
            assert_eq!(auxiliary.metadata()?.permissions().mode() & 0o777, 0o600);
        }
    }

    drop(store);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
    let reopened = SqliteStore::connect(&url).await?;
    assert_eq!(path.metadata()?.permissions().mode() & 0o777, 0o600);
    drop(reopened);
    Ok(())
}

#[tokio::test]
async fn in_memory_databases_are_refused() {
    for url in [
        "sqlite::memory:",
        ":memory:",
        "sqlite://:memory:",
        "SQLITE::MEMORY:",
        "sqlite:file:memoryos?mode=memory&cache=shared",
    ] {
        assert!(is_invalid(SqliteStore::connect(url).await), "{url}");
    }
}

// ---- Episodes and atomic commits --------------------------------------------------------------

#[tokio::test]
async fn episode_claim_is_idempotent_per_tenant_and_applies_defaults() -> TestResult {
    let db = db().await?;
    let a = ctx("t1");
    let scope = user("u1");
    let mut new = episode(&scope, Trust::User, "hello");
    new.idempotency_key = Some("key-1".into());
    new.metadata = serde_json::Value::Null;

    let (first, created) = db.store.claim_episode(&a, new.clone()).await?;
    assert!(created);
    assert_eq!(first.metadata, json!({}));
    assert_eq!(first.occurred_at, first.created_at);
    let (second, created) = db.store.claim_episode(&a, new.clone()).await?;
    assert!(!created);
    assert_eq!(first.id, second.id);

    // The idempotency namespace includes the tenant, not just the key.
    let (other, created) = db.store.claim_episode(&ctx("t2"), new).await?;
    assert!(created);
    assert_ne!(other.id, first.id);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_episode_claims_with_one_key_create_exactly_one_row() -> TestResult {
    let db = db().await?;
    let mut new = episode(&user("u1"), Trust::User, "hello");
    new.idempotency_key = Some("shared-key".into());
    let calls = (0..8).map(|_| {
        let store = db.store.clone();
        let new = new.clone();
        tokio::spawn(async move { store.claim_episode(&ctx("t1"), new).await })
    });
    let results = futures::future::join_all(calls).await;
    let mut ids = BTreeSet::new();
    let mut creators = 0;
    for result in results {
        let (episode, created) = result??;
        ids.insert(episode.id);
        creators += usize::from(created);
    }
    assert_eq!(ids.len(), 1);
    assert_eq!(creators, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_apply_commit_applies_exactly_once() -> TestResult {
    let db = db().await?;
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
        tokio::spawn(async move { store.apply_commit(&c, episode_id, ops).await })
    });
    let results = futures::future::join_all(calls).await;
    let mut applied = 0;
    for result in results {
        if !result??.is_empty() {
            applied += 1;
        }
    }
    assert_eq!(applied, 1);
    let all = db
        .store
        .list_facts(&c, &scope, FactListState::All, 10, None)
        .await?;
    assert_eq!(all.len(), 3, "old fact plus the two committed facts");
    assert_eq!(db.store.facts_for_episode(&c, episode_id).await?.len(), 2);
    assert!(db
        .store
        .get_episode(&c, episode_id)
        .await?
        .processed_at
        .is_some());
    Ok(())
}

#[tokio::test]
async fn failed_commit_rolls_back_all_operations_and_can_retry() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;

    let failed = db
        .store
        .apply_commit(
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
    assert!(is_not_found(failed));
    assert!(db
        .store
        .list_facts(&c, &scope, FactListState::All, 10, None)
        .await?
        .is_empty());
    assert!(db
        .store
        .get_episode(&c, episode_id)
        .await?
        .processed_at
        .is_none());

    let retry = db
        .store
        .apply_commit(
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
    assert!(db
        .store
        .apply_commit(
            &c,
            episode_id,
            vec![FactOp::Add(new_fact(
                &scope,
                subject,
                "duplicate replay",
                date(2020, 1),
            ))],
        )
        .await?
        .is_empty());
    assert_eq!(db.store.facts_for_episode(&c, episode_id).await?.len(), 1);
    Ok(())
}

// ---- Tenant isolation -------------------------------------------------------------------------

#[tokio::test]
async fn tenant_b_cannot_read_search_change_or_reference_tenant_a_data() -> TestResult {
    let db = db().await?;
    let a = ctx("tenant-a");
    let b = ctx("tenant-b");
    let scope = user("shared-id");
    let scopes = [scope.clone()];
    let filter = FactFilter::default();

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
        .apply_commit(&a, source.id, vec![FactOp::Add(secret)])
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
        .apply_commit(&a, external, vec![FactOp::Add(held)])
        .await?[0];
    db.store
        .put_block(&a, "bot", "persona", "secret", 100, None)
        .await?;

    // Reads by id and all scoped read families hide A's rows from B.
    assert!(is_not_found(db.store.get_episode(&b, source.id).await));
    assert!(is_not_found(
        db.store.facts_for_episode(&b, source.id).await
    ));
    assert!(is_not_found(db.store.get_entity(&b, subject).await));
    assert!(is_not_found(db.store.get_fact(&b, fact).await));
    assert!(is_not_found(db.store.fact_history(&b, fact).await));
    assert!(is_not_found(db.store.get_block(&b, "bot", "persona").await));
    assert!(is_not_found(
        db.store.block_versions(&b, "bot", "persona").await
    ));
    assert!(is_not_found(
        db.store
            .neighbors(&b, &scopes, subject, 2, &filter, 5000)
            .await
    ));
    assert!(db.store.current_fact(&b, &scope, fact).await?.is_none());
    assert!(db
        .store
        .current_facts_for_subject(&b, &scope, subject)
        .await?
        .is_empty());
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
        .similar_entities(&b, &scope, &axis(1), 10)
        .await?
        .is_empty());
    assert!(db.store.list_quarantined(&b, &scopes, 10).await?.is_empty());
    assert!(db.store.list_blocks(&b, "bot").await?.is_empty());
    assert!(db
        .store
        .list_facts(&b, &scope, FactListState::All, 10, None)
        .await?
        .is_empty());
    assert!(db.store.scope_summaries(&b).await?.is_empty());
    assert_eq!(db.store.highest_trust_in_scope(&b, &scope).await?, None);

    // Writes against A's rows fail as missing rather than revealing or changing them.
    assert!(is_not_found(db.store.approve_fact(&b, quarantined).await));
    assert!(is_not_found(db.store.reject_fact(&b, quarantined).await));
    assert!(is_not_found(db.store.invalidate_fact(&b, fact, "x").await));
    assert!(is_not_found(db.store.purge_episode(&b, source.id).await));
    assert!(is_not_found(
        db.store.add_entity_alias(&b, subject, "x", None).await
    ));
    assert!(is_not_found(
        db.store
            .add_entity_alias(&b, subject, "x", Some(source.id))
            .await
    ));
    assert!(is_not_found(
        db.store
            .put_block(&b, "bot", "persona", "x", 100, Some(1))
            .await
    ));
    assert!(is_not_found(
        db.store.apply_commit(&b, pending.id, vec![]).await
    ));

    // Tenant-aware foreign keys are not enough: every cross-tenant reference must be rejected.
    let b_episode = new_episode(&db, &b, &scope, Trust::User).await?;
    let b_subject = entity(&db, &b, &scope, "own").await?;
    let foreign_ops = [
        FactOp::Add(new_fact(&scope, subject, "foreign subject", date(2020, 1))),
        FactOp::Add({
            let mut foreign = new_fact(&scope, b_subject, "foreign object", date(2020, 1));
            foreign.object_id = Some(object);
            foreign
        }),
        FactOp::Supersede {
            old_fact_id: fact,
            new: new_fact(&scope, b_subject, "foreign supersede", date(2021, 1)),
        },
        FactOp::Corroborate { fact_id: fact },
        FactOp::SupersedeWithExisting {
            old_fact_id: fact,
            fact_id: fact,
        },
    ];
    for op in foreign_ops {
        assert!(is_not_found(
            db.store.apply_commit(&b, b_episode, vec![op]).await
        ));
    }
    // All failures rolled back, leaving B's episode usable and A's episode unprocessed.
    db.store
        .apply_commit(
            &b,
            b_episode,
            vec![FactOp::Add(new_fact(
                &scope,
                b_subject,
                "own fact",
                date(2020, 1),
            ))],
        )
        .await?;
    assert!(db
        .store
        .get_episode(&a, pending.id)
        .await?
        .processed_at
        .is_none());

    // Purging the identically named scope in B cannot affect A. Blocks have independent tenancy.
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
        db.store.get_fact(&a, quarantined).await?.status,
        FactStatus::Quarantined
    );
    Ok(())
}

// ---- Supersession, history, bitemporal visibility and expiry ----------------------------------

#[tokio::test]
async fn supersession_preserves_history_and_invalidation_reopens_the_predecessor() -> TestResult {
    let db = db().await?;
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
    assert!(db.store.current_fact(&c, &scope, old).await?.is_none());
    assert_eq!(
        fact_ids(
            &db.store
                .current_facts_for_subject(&c, &scope, subject)
                .await?
        ),
        vec![new]
    );
    assert_eq!(
        fact_ids(&db.store.fact_history(&c, new).await?),
        vec![old, new]
    );
    assert_eq!(
        fact_ids(&db.store.fact_history(&c, old).await?),
        vec![old, new]
    );

    db.store.invalidate_fact(&c, new, "wrong").await?;
    let reopened = db.store.get_fact(&c, old).await?;
    assert_eq!((reopened.valid_to, reopened.superseded_by), (None, None));
    assert_eq!(
        fact_ids(
            &db.store
                .current_facts_for_subject(&c, &scope, subject)
                .await?
        ),
        vec![old]
    );
    assert!(db.store.current_fact(&c, &scope, new).await?.is_none());

    // Repeating invalidation is harmless and cannot close the reopened predecessor again.
    db.store.invalidate_fact(&c, new, "wrong again").await?;
    assert_eq!(db.store.get_fact(&c, old).await?.superseded_by, None);
    Ok(())
}

#[tokio::test]
async fn valid_time_knowledge_time_and_retraction_filters_match_the_contract() -> TestResult {
    let db = db().await?;
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
        assert_eq!(scored_ids(found), expected, "as_of {when}");
    }

    let recorded = db.store.get_fact(&c, new).await?.recorded_at;
    let before_recorded = FactFilter {
        as_of: Some(date(2023, 1)),
        known_at: Some(recorded - chrono::Duration::microseconds(1)),
    };
    assert!(db
        .store
        .search_facts_vector(&c, &scopes, &axis(0), &before_recorded, 10)
        .await?
        .is_empty());
    let at_recorded = FactFilter {
        as_of: Some(date(2023, 1)),
        known_at: Some(recorded),
    };
    assert_eq!(
        scored_ids(
            db.store
                .search_facts_text(&c, &scopes, "london", &at_recorded, 10)
                .await?
        ),
        vec![new]
    );

    let invalidated = db.store.invalidate_fact(&c, new, "wrong").await?;
    let retracted_at = invalidated.invalidated_at.expect("retraction time");
    let before_retraction = FactFilter {
        as_of: None,
        known_at: Some(retracted_at - chrono::Duration::microseconds(1)),
    };
    assert_eq!(
        scored_ids(
            db.store
                .search_facts_text(&c, &scopes, "london", &before_retraction, 10)
                .await?
        ),
        vec![new]
    );
    let after_retraction = FactFilter {
        as_of: None,
        known_at: Some(retracted_at),
    };
    assert!(db
        .store
        .search_facts_text(&c, &scopes, "london", &after_retraction, 10)
        .await?
        .is_empty());
    assert_eq!(
        fact_ids(
            &db.store
                .current_facts_for_subject(&c, &scope, subject)
                .await?
        ),
        vec![old]
    );
    Ok(())
}

#[tokio::test]
async fn expiry_is_an_exclusive_visibility_boundary_and_expired_facts_are_past() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let subject = entity(&db, &c, &scope, "ada").await?;
    let mut temporary = new_fact(&scope, subject, "temporary leave", date(2020, 1));
    temporary.expires_at = Some(date(2022, 1));
    let (_, id) = add_fact(&db, &c, temporary).await?;

    for when in [date(2020, 1), date(2021, 12)] {
        assert_eq!(
            scored_ids(
                db.store
                    .search_facts_vector(&c, &scopes, &axis(0), &at(when), 10)
                    .await?
            ),
            vec![id]
        );
        assert_eq!(
            scored_ids(
                db.store
                    .search_facts_text(&c, &scopes, "temporary", &at(when), 10)
                    .await?
            ),
            vec![id]
        );
    }
    for when in [date(2022, 1), date(2023, 1)] {
        assert!(db
            .store
            .search_facts_vector(&c, &scopes, &axis(0), &at(when), 10)
            .await?
            .is_empty());
        assert!(db
            .store
            .search_facts_text(&c, &scopes, "temporary", &at(when), 10)
            .await?
            .is_empty());
    }
    assert!(db.store.current_fact(&c, &scope, id).await?.is_none());
    assert!(db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?
        .is_empty());
    assert_eq!(
        db.store
            .list_facts(&c, &scope, FactListState::Past, 10, None)
            .await?
            .iter()
            .map(|view| view.fact.id)
            .collect::<Vec<_>>(),
        vec![id]
    );

    for expiry in [date(2019, 1), date(2020, 1)] {
        let mut invalid = new_fact(&scope, subject, "bad expiry", date(2020, 1));
        invalid.expires_at = Some(expiry);
        let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;
        assert!(is_invalid(
            db.store
                .apply_commit(&c, episode_id, vec![FactOp::Add(invalid)])
                .await
        ));
    }
    Ok(())
}

// ---- Quarantine and trust rules ---------------------------------------------------------------

#[tokio::test]
async fn quarantined_supersede_requires_approval_or_rejection() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];

    // Approval activates the proposal and performs exactly the close that was reviewed.
    let subject = entity(&db, &c, &scope, "approved").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "paris", date(2020, 1))).await?;
    let mut proposal = new_fact(&scope, subject, "london", date(2022, 1));
    proposal.trust = Trust::External;
    proposal.status = FactStatus::Quarantined;
    let new = supersede(&db, &c, old, proposal).await?;
    let open = db.store.get_fact(&c, old).await?;
    assert_eq!((open.valid_to, open.superseded_by), (None, None));
    assert_eq!(
        db.store.get_fact(&c, new).await?.pending_supersede_of,
        Some(old)
    );
    assert_eq!(
        fact_ids(&db.store.list_quarantined(&c, &scopes, 10).await?),
        vec![new]
    );
    assert!(db
        .store
        .search_facts_text(&c, &scopes, "london", &FactFilter::default(), 10)
        .await?
        .is_empty());
    assert_eq!(
        scored_ids(
            db.store
                .search_facts_vector(&c, &scopes, &axis(0), &FactFilter::default(), 10)
                .await?
        ),
        vec![old]
    );

    let approved = db.store.approve_fact(&c, new).await?;
    assert_eq!(approved.status, FactStatus::Active);
    assert_eq!(approved.pending_supersede_of, None);
    let closed = db.store.get_fact(&c, old).await?;
    assert_eq!(closed.valid_to, Some(date(2022, 1)));
    assert_eq!(closed.superseded_by, Some(new));
    assert!(db.store.list_quarantined(&c, &scopes, 10).await?.is_empty());
    assert!(is_invalid(db.store.approve_fact(&c, new).await));

    // Rejection leaves its predecessor open and the rejected fact out of quarantine and recall.
    let subject = entity(&db, &c, &scope, "rejected").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "truth", date(2020, 1))).await?;
    let mut proposal = new_fact(&scope, subject, "falsehood", date(2022, 1));
    proposal.trust = Trust::External;
    proposal.status = FactStatus::Quarantined;
    let rejected_id = supersede(&db, &c, old, proposal).await?;
    let rejected = db.store.reject_fact(&c, rejected_id).await?;
    assert_eq!(rejected.status, FactStatus::Rejected);
    assert_eq!(
        (
            db.store.get_fact(&c, old).await?.valid_to,
            db.store.get_fact(&c, old).await?.superseded_by
        ),
        (None, None)
    );
    assert!(db.store.list_quarantined(&c, &scopes, 10).await?.is_empty());
    assert!(db
        .store
        .search_facts_text(&c, &scopes, "falsehood", &FactFilter::default(), 10)
        .await?
        .is_empty());
    assert!(is_invalid(db.store.approve_fact(&c, rejected_id).await));
    Ok(())
}

#[tokio::test]
async fn commits_enforce_source_trust_scope_embedding_and_time_rules() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let other_scope = user("u2");
    let other_subject = entity(&db, &c, &other_scope, "bob").await?;
    let good = |statement| new_fact(&scope, subject, statement, date(2020, 1));

    let external = new_episode(&db, &c, &scope, Trust::External).await?;
    let mut poisoned = good("poison");
    poisoned.trust = Trust::External;
    assert!(is_invalid(
        db.store
            .apply_commit(&c, external, vec![FactOp::Add(poisoned)])
            .await
    ));

    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;
    let mut raised = good("raised");
    raised.trust = Trust::System;
    let mut wrong_scope = new_fact(&other_scope, other_subject, "wrong scope", date(2020, 1));
    wrong_scope.trust = Trust::User;
    let foreign_entity = new_fact(&scope, other_subject, "foreign entity", date(2020, 1));
    let mut short_embedding = good("short");
    short_embedding.embedding = vec![1.0; 3];
    let mut nonfinite = good("nonfinite");
    nonfinite.embedding[0] = f32::NAN;
    let mut zero = good("zero");
    zero.embedding = vec![0.0; EMBEDDING_DIMS];
    let mut reversed = good("reversed");
    reversed.valid_to = Some(date(2019, 1));
    let mut rejected = good("rejected");
    rejected.status = FactStatus::Rejected;
    for bad in [
        raised,
        wrong_scope,
        foreign_entity,
        short_embedding,
        nonfinite,
        zero,
        reversed,
        rejected,
    ] {
        assert!(is_invalid(
            db.store
                .apply_commit(&c, episode_id, vec![FactOp::Add(bad)])
                .await
        ));
    }
    assert!(db
        .store
        .list_facts(&c, &scope, FactListState::All, 10, None)
        .await?
        .is_empty());

    // Agent content must be reviewed outside an agent scope, but may be active in its own scope.
    for scope in [user("u1"), Scope::shared("team")] {
        let subject = entity(&db, &c, &scope, "agent fact").await?;
        let episode_id = new_episode(&db, &c, &scope, Trust::Agent).await?;
        let mut planted = new_fact(&scope, subject, "planted", date(2020, 1));
        planted.trust = Trust::Agent;
        assert!(is_invalid(
            db.store
                .apply_commit(&c, episode_id, vec![FactOp::Add(planted.clone())])
                .await
        ));
        planted.status = FactStatus::Quarantined;
        let held = add_fact(&db, &c, planted).await?.1;
        assert_eq!(
            db.store.get_fact(&c, held).await?.status,
            FactStatus::Quarantined
        );
    }
    let scope = Scope::agent("bot");
    let subject = entity(&db, &c, &scope, "own").await?;
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
async fn lower_trust_supersede_of_a_higher_trust_head_is_quarantined() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
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
    let held = supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "user says", date(2022, 1)),
    )
    .await?;

    let open = db.store.get_fact(&c, head).await?;
    assert_eq!((open.valid_to, open.superseded_by), (None, None));
    let held_fact = db.store.get_fact(&c, held).await?;
    assert_eq!(held_fact.status, FactStatus::Quarantined);
    assert_eq!(held_fact.pending_supersede_of, Some(head));
    assert_eq!((held_fact.valid_to, held_fact.superseded_by), (None, None));

    let approved = db.store.approve_fact(&c, held).await?;
    assert_eq!(approved.status, FactStatus::Active);
    assert_eq!(db.store.get_fact(&c, head).await?.superseded_by, Some(held));
    assert_eq!(
        db.store.get_fact(&c, head).await?.valid_to,
        Some(date(2022, 1))
    );
    Ok(())
}

// ---- Purge -------------------------------------------------------------------------------------

#[tokio::test]
async fn purge_keeps_shared_facts_and_deletes_sole_sourced_facts_and_orphan_entities() -> TestResult
{
    let db = db().await?;
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
        .apply_commit(
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
    let good = corroborate(&db, &c, &scope, Trust::User, shared_fact).await?;

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
    assert!(is_not_found(db.store.purge_episode(&c, bad).await));
    Ok(())
}

#[tokio::test]
async fn purge_recomputes_trust_requarantines_and_reopens_closed_facts() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "ada").await?;

    // Losing the sole trusted source leaves an external fact quarantined.
    let (_, truth) = add_fact(&db, &c, new_fact(&scope, subject, "truth", date(2020, 1))).await?;
    let (good, claim) = {
        let source = new_episode(&db, &c, &scope, Trust::User).await?;
        let claim = db
            .store
            .apply_commit(
                &c,
                source,
                vec![FactOp::Supersede {
                    old_fact_id: truth,
                    new: new_fact(&scope, subject, "claim", date(2021, 1)),
                }],
            )
            .await?[0];
        (source, claim)
    };
    let external = corroborate(&db, &c, &scope, Trust::External, claim).await?;
    let report = db.store.purge_episode(&c, good).await?;
    assert_eq!(report.facts_deleted, 0);
    let reopened = db.store.get_fact(&c, truth).await?;
    assert_eq!((reopened.valid_to, reopened.superseded_by), (None, None));
    let held = db.store.get_fact(&c, claim).await?;
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!(held.trust, Trust::External);
    assert_eq!(held.source_episode_ids, vec![external]);
    assert_eq!(held.pending_supersede_of, Some(truth));
    assert_eq!(
        fact_ids(
            &db.store
                .current_facts_for_subject(&c, &scope, subject)
                .await?
        ),
        vec![truth]
    );
    db.store.approve_fact(&c, claim).await?;
    assert_eq!(
        db.store.get_fact(&c, truth).await?.superseded_by,
        Some(claim)
    );

    // Trust is recomputed from remaining sources rather than retained from the purged maximum.
    let other = entity(&db, &c, &scope, "grace").await?;
    let (top, recomputed) = add_fact(
        &db,
        &c,
        with_trust(
            new_fact(&scope, other, "recomputed", date(2020, 1)),
            Trust::System,
        ),
    )
    .await?;
    corroborate(&db, &c, &scope, Trust::External, recomputed).await?;
    let user_source = corroborate(&db, &c, &scope, Trust::User, recomputed).await?;
    db.store.purge_episode(&c, top).await?;
    let left = db.store.get_fact(&c, recomputed).await?;
    assert_eq!((left.status, left.trust), (FactStatus::Active, Trust::User));
    assert!(left.source_episode_ids.contains(&user_source));
    Ok(())
}

// ---- Entities ---------------------------------------------------------------------------------

#[tokio::test]
async fn entity_upsert_aliases_lookup_and_cosine_ordering_match_the_contract() -> TestResult {
    let db = db().await?;
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
        vec!["Countess"]
    );
    assert_eq!(
        db.store
            .find_entity(&c, &scope, "person", " ADA   lovelace ")
            .await?
            .map(|entity| entity.id),
        Some(first.id)
    );
    assert_eq!(
        db.store
            .find_entity(&c, &scope, "person", "countess")
            .await?
            .map(|entity| entity.id),
        Some(first.id)
    );
    // A real name takes precedence over another entity's alias.
    db.store.add_entity_alias(&c, first.id, "Bob", None).await?;
    assert_eq!(
        db.store
            .find_entity(&c, &scope, "person", "bob")
            .await?
            .map(|entity| entity.id),
        Some(far.id)
    );
    assert!(db
        .store
        .find_entity(&c, &scope, "other", "bob")
        .await?
        .is_none());
    assert!(db
        .store
        .find_entity(&c, &user("u2"), "person", "bob")
        .await?
        .is_none());
    assert!(is_not_found(
        db.store
            .add_entity_alias(&c, Uuid::now_v7(), "x", None)
            .await
    ));
    assert!(is_not_found(db.store.get_entity(&c, Uuid::now_v7()).await));
    Ok(())
}

#[tokio::test]
async fn purging_the_last_alias_source_removes_only_that_alias() -> TestResult {
    let db = db().await?;
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
    assert_eq!(
        db.store.get_entity(&c, subject).await?.aliases,
        ["Bob", "Rob"]
    );
    db.store.purge_episode(&c, third).await?;
    assert_eq!(db.store.get_entity(&c, subject).await?.aliases, ["Rob"]);
    assert!(db
        .store
        .find_entity(&c, &scope, "thing", "bob")
        .await?
        .is_none());
    db.store.get_episode(&c, first).await?;
    Ok(())
}

/// Matches PostgreSQL: discard drops the alias provenance by cascade but keeps the alias itself.
/// Through ingest an unprocessed episode has earned no alias, so this only shows via the store API.
#[tokio::test]
async fn discarding_an_unprocessed_episode_keeps_its_aliases_as_postgres_does() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Robert").await?;
    let source = new_episode(&db, &c, &scope, Trust::User).await?;
    db.store
        .add_entity_alias(&c, subject, "Bob", Some(source))
        .await?;

    assert!(db.store.discard_unprocessed_episode(&c, source).await?);
    assert_eq!(db.store.get_entity(&c, subject).await?.aliases, ["Bob"]);
    assert!(db
        .store
        .find_entity(&c, &scope, "thing", "bob")
        .await?
        .is_some());
    assert!(is_not_found(db.store.get_episode(&c, source).await));
    Ok(())
}

/// Matches PostgreSQL: purge_scope leaves aliases on entities outside the scope. Through ingest an
/// alias is only earned on an entity in the episode's own scope, so this only shows via the store API.
#[tokio::test]
async fn purging_a_scope_keeps_aliases_it_earned_in_other_scopes_as_postgres_does() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let source_scope = user("source");
    let entity_scope = user("entity");
    let subject = entity(&db, &c, &entity_scope, "Robert").await?;
    let source = new_episode(&db, &c, &source_scope, Trust::User).await?;
    db.store
        .add_entity_alias(&c, subject, "Bob", Some(source))
        .await?;

    db.store.purge_scope(&c, &source_scope).await?;
    assert_eq!(db.store.get_entity(&c, subject).await?.aliases, ["Bob"]);
    assert!(db
        .store
        .find_entity(&c, &entity_scope, "thing", "bob")
        .await?
        .is_some());
    Ok(())
}

// ---- Vector search -----------------------------------------------------------------------------

/// `axis(0)` plus a deterministic pseudo-random vector of magnitude `noise`.
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

#[tokio::test]
async fn selective_vector_search_returns_the_full_limit_in_cosine_order() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let big = user("big");
    let rare = user("rare");
    let big_subject = entity(&db, &c, &big, "big").await?;
    let rare_subject = entity(&db, &c, &rare, "rare").await?;

    // Interleave a large unselected scope with a smaller selected one. Filtering before the limit
    // must not let the unselected vectors starve the requested result set.
    for batch in 0..12u64 {
        let mut rare_fact = new_fact(&rare, rare_subject, "rare fact", date(2020, 1));
        rare_fact.embedding = noisy(10_000 + batch, 0.7);
        add_fact(&db, &c, rare_fact).await?;
        let episode_id = new_episode(&db, &c, &big, Trust::User).await?;
        let ops = (0..10u64)
            .map(|i| {
                let mut fact = new_fact(&big, big_subject, "big fact", date(2020, 1));
                fact.embedding = noisy(batch * 100 + i, 0.5);
                FactOp::Add(fact)
            })
            .collect();
        db.store.apply_commit(&c, episode_id, ops).await?;
    }

    let found = db
        .store
        .search_facts_vector(
            &c,
            std::slice::from_ref(&rare),
            &axis(0),
            &FactFilter::default(),
            5,
        )
        .await?;
    assert_eq!(found.len(), 5);
    assert!(found.iter().all(|hit| hit.fact.scope == rare));
    assert!(found.windows(2).all(|pair| pair[0].score >= pair[1].score));
    assert!((found[0].score - 0.8192).abs() < 0.1, "{}", found[0].score);
    assert!(db
        .store
        .search_facts_vector(&c, &[user("missing")], &axis(0), &FactFilter::default(), 5)
        .await?
        .is_empty());
    Ok(())
}

#[tokio::test]
async fn vector_search_rejects_wrong_width_nonfinite_zero_and_bad_limits() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scopes = [user("u1")];
    let filter = FactFilter::default();
    for bad in [vec![1.0; 3], vec![], vec![1.0; EMBEDDING_DIMS + 1]] {
        assert!(is_invalid(
            db.store
                .search_facts_vector(&c, &scopes, &bad, &filter, 5)
                .await
        ));
        assert!(is_invalid(
            db.store.similar_entities(&c, &scopes[0], &bad, 5).await
        ));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = axis(0);
        bad[7] = value;
        assert!(is_invalid(
            db.store
                .search_facts_vector(&c, &scopes, &bad, &filter, 5)
                .await
        ));
        assert!(is_invalid(
            db.store.similar_entities(&c, &scopes[0], &bad, 5).await
        ));
    }
    let zero = vec![0.0; EMBEDDING_DIMS];
    assert!(is_invalid(
        db.store
            .search_facts_vector(&c, &scopes, &zero, &filter, 5)
            .await
    ));
    assert!(is_invalid(
        db.store.similar_entities(&c, &scopes[0], &zero, 5).await
    ));
    assert!(is_invalid(
        db.store
            .search_facts_vector(&c, &scopes, &axis(0), &filter, 0)
            .await
    ));
    assert!(is_invalid(
        db.store
            .search_facts_vector(&c, &scopes, &axis(0), &filter, -1)
            .await
    ));
    assert!(db
        .store
        .search_facts_vector(&c, &[], &axis(0), &filter, 5)
        .await?
        .is_empty());
    Ok(())
}

// ---- Text search -------------------------------------------------------------------------------

#[tokio::test]
async fn text_search_ranks_filters_scope_and_treats_punctuation_as_data() -> TestResult {
    let db = db().await?;
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
        found.iter().map(|hit| hit.fact.id).collect::<Vec<_>>(),
        vec![strong, weak]
    );
    assert!(found[0].score > found[1].score);
    let phrase = db
        .store
        .search_facts_text(
            &c,
            std::slice::from_ref(&scope),
            "\"rust and\"",
            &filter,
            10,
        )
        .await?;
    assert_eq!(scored_ids(phrase), vec![weak]);

    // None of these may become SQL or an unhandled FTS expression, and none match stored text.
    for hostile in [
        "'; DROP TABLE facts; --",
        "",
        "&&& ((( :*",
        "\"",
        "NOT",
        "NEAR(",
        "-nothingness",
    ] {
        let result = db
            .store
            .search_facts_text(&c, std::slice::from_ref(&scope), hostile, &filter, 10)
            .await?;
        assert!(result.is_empty(), "{hostile}");
    }

    // A punctuation-only word is dropped, as PostgreSQL drops it, instead of matching nothing.
    let punctuated = db
        .store
        .search_facts_text(
            &c,
            std::slice::from_ref(&scope),
            "rust & nothing --",
            &filter,
            10,
        )
        .await?;
    assert_eq!(scored_ids(punctuated), vec![weak]);
    // `or` between terms is OR, as in PostgreSQL websearch syntax.
    let either = db
        .store
        .search_facts_text(
            &c,
            std::slice::from_ref(&scope),
            "\"rust rust\" or nothing",
            &filter,
            10,
        )
        .await?;
    let mut either = scored_ids(either);
    either.sort();
    let mut both = vec![strong, weak];
    both.sort();
    assert_eq!(either, both);
    assert_eq!(
        db.store
            .list_facts(&c, &scope, FactListState::All, 10, None)
            .await?
            .len(),
        2
    );
    Ok(())
}

#[tokio::test]
async fn text_ranking_is_not_affected_by_another_tenants_corpus() -> TestResult {
    let db = db().await?;
    let a = ctx("tenant-a");
    let b = ctx("tenant-b");
    let scope = user("shared");
    let subject = entity(&db, &a, &scope, "Ada").await?;
    add_fact(
        &db,
        &a,
        new_fact(
            &scope,
            subject,
            "rust rust makes local matches",
            date(2020, 1),
        ),
    )
    .await?;
    let before = db
        .store
        .search_facts_text(
            &a,
            std::slice::from_ref(&scope),
            "rust",
            &FactFilter::default(),
            10,
        )
        .await?
        .into_iter()
        .map(|hit| (hit.fact.id, hit.score))
        .collect::<Vec<_>>();

    let other_subject = entity(&db, &b, &scope, "Bob").await?;
    for index in 0..12 {
        add_fact(
            &db,
            &b,
            new_fact(
                &scope,
                other_subject,
                &format!("rust rust rust rust unrelated document {index}"),
                date(2020, 1),
            ),
        )
        .await?;
    }
    let after = db
        .store
        .search_facts_text(
            &a,
            std::slice::from_ref(&scope),
            "rust",
            &FactFilter::default(),
            10,
        )
        .await?
        .into_iter()
        .map(|hit| (hit.fact.id, hit.score))
        .collect::<Vec<_>>();
    assert_eq!(after, before);
    Ok(())
}

#[tokio::test]
async fn text_search_applies_time_scope_and_quarantine_filters() -> TestResult {
    let db = db().await?;
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

    let current = db
        .store
        .search_facts_text(&c, &scopes, "marker", &FactFilter::default(), 10)
        .await?;
    assert_eq!(scored_ids(current), vec![new]);
    assert_eq!(
        scored_ids(
            db.store
                .search_facts_text(&c, &scopes, "marker", &at(date(2021, 1)), 10)
                .await?
        ),
        vec![old]
    );
    assert!(db
        .store
        .search_facts_text(&c, &scopes, "marker", &at(date(2019, 1)), 10)
        .await?
        .is_empty());
    assert!(db
        .store
        .search_facts_text(&c, &[user("u2")], "marker", &FactFilter::default(), 10)
        .await?
        .is_empty());
    let recorded = db.store.get_fact(&c, new).await?.recorded_at;
    let before = FactFilter {
        as_of: Some(date(2023, 1)),
        known_at: Some(recorded - chrono::Duration::microseconds(1)),
    };
    assert!(db
        .store
        .search_facts_text(&c, &scopes, "marker", &before, 10)
        .await?
        .is_empty());
    db.store.invalidate_fact(&c, new, "wrong").await?;
    assert_eq!(
        scored_ids(
            db.store
                .search_facts_text(&c, &scopes, "marker", &FactFilter::default(), 10)
                .await?
        ),
        vec![old]
    );
    Ok(())
}

// ---- Graph -------------------------------------------------------------------------------------

#[tokio::test]
async fn graph_respects_depth_terminates_on_cycles_and_limits_edges() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let scopes = [scope.clone()];
    let a = entity(&db, &c, &scope, "a").await?;
    let b = entity(&db, &c, &scope, "b").await?;
    let d = entity(&db, &c, &scope, "d").await?;
    let e = entity(&db, &c, &scope, "e").await?;
    // a -> b -> d -> e, d -> a closes a cycle, and e -> e is a self-loop.
    let ab = edge(&db, &c, &scope, a, b).await?;
    let bd = edge(&db, &c, &scope, b, d).await?;
    let da = edge(&db, &c, &scope, d, a).await?;
    let de = edge(&db, &c, &scope, d, e).await?;
    let ee = edge(&db, &c, &scope, e, e).await?;
    let filter = FactFilter::default();

    let entity_ids = |graph: &memoryos_core::GraphNeighborhood| {
        graph
            .entities
            .iter()
            .map(|entity| entity.id)
            .collect::<BTreeSet<_>>()
    };
    let edge_ids = |graph: &memoryos_core::GraphNeighborhood| {
        graph
            .edges
            .iter()
            .map(|fact| fact.id)
            .collect::<BTreeSet<_>>()
    };
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
    assert_eq!(five.edges.len(), 5, "cycles must not duplicate edges");
    let limited = db.store.neighbors(&c, &scopes, a, 5, &filter, 3).await?;
    assert_eq!(limited.edges.len(), 3);
    assert_eq!(edge_ids(&limited).len(), 3);
    for bad in [0, 6, -1] {
        assert!(is_invalid(
            db.store.neighbors(&c, &scopes, a, bad, &filter, 5000).await
        ));
    }
    for bad in [0, -1] {
        assert!(is_invalid(
            db.store.neighbors(&c, &scopes, a, 1, &filter, bad).await
        ));
    }
    Ok(())
}

#[tokio::test]
async fn graph_applies_scope_time_retraction_and_expiry_to_edges() -> TestResult {
    let db = db().await?;
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
    assert_eq!(fact_ids(&current.edges), vec![new_edge]);
    assert!(current.entities.iter().all(|entity| entity.id != b));
    let past = db
        .store
        .neighbors(&c, &scopes, a, 2, &at(date(2021, 1)), 5000)
        .await?;
    assert_eq!(fact_ids(&past.edges), vec![old_edge]);
    db.store.invalidate_fact(&c, new_edge, "wrong").await?;
    let reopened = db
        .store
        .neighbors(&c, &scopes, a, 2, &FactFilter::default(), 5000)
        .await?;
    assert_eq!(fact_ids(&reopened.edges), vec![old_edge]);

    let expiring_entity = entity(&db, &c, &scope, "expiring").await?;
    let mut temporary = new_fact(&scope, a, "temporary edge", date(2020, 1));
    temporary.object_id = Some(expiring_entity);
    temporary.expires_at = Some(date(2022, 1));
    let temporary = add_fact(&db, &c, temporary).await?.1;
    assert!(fact_ids(
        &db.store
            .neighbors(&c, &scopes, a, 1, &at(date(2021, 1)), 5000)
            .await?
            .edges
    )
    .contains(&temporary));
    assert!(!fact_ids(
        &db.store
            .neighbors(&c, &scopes, a, 1, &at(date(2022, 1)), 5000)
            .await?
            .edges
    )
    .contains(&temporary));

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

// ---- Blocks ------------------------------------------------------------------------------------

#[tokio::test]
async fn block_updates_use_optimistic_versions_and_append_history() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let created = db
        .store
        .put_block(&c, "a", "human", "v1", 100, None)
        .await?;
    assert_eq!(created.version, 1);
    assert!(matches!(
        db.store.put_block(&c, "a", "human", "x", 100, None).await,
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
    assert!(matches!(
        db.store
            .put_block(&c, "a", "human", "x", 100, Some(1))
            .await,
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
            .map(|version| (
                version.version,
                version.content.as_str(),
                version.edited_by.as_str()
            ))
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

    // Character limits count Unicode scalar values, not UTF-8 bytes, and failed writes are atomic.
    let cjk = "\u{8a18}\u{61b6}\u{7cfb}\u{7d71}\u{5eab}";
    db.store.put_block(&c, "a", "cjk", cjk, 5, None).await?;
    assert!(matches!(
        db.store
            .put_block(
                &c,
                "a",
                "large",
                "\u{8a18}\u{8a18}\u{8a18}\u{8a18}\u{8a18}\u{8a18}",
                5,
                None
            )
            .await,
        Err(Error::BlockTooLarge { len: 6, max: 5 })
    ));
    assert!(is_not_found(db.store.get_block(&c, "a", "large").await));
    assert!(matches!(
        db.store
            .put_block(&c, "a", "cjk", "abcdef", 5, Some(1))
            .await,
        Err(Error::BlockTooLarge { .. })
    ));
    assert_eq!(db.store.get_block(&c, "a", "cjk").await?.version, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_block_updates_have_exactly_one_winner_per_version() -> TestResult {
    let db = db().await?;
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
    assert_eq!(db.store.get_block(&c, "a", "x").await?.version, 2);
    assert_eq!(db.store.block_versions(&c, "a", "x").await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn block_history_is_pruned_to_the_latest_100_oldest_first() -> TestResult {
    let db = db().await?;
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
    assert_eq!(block.version, 105);
    let versions = db.store.block_versions(&c, "a1", "notes").await?;
    assert_eq!(versions.len(), 100);
    assert_eq!(versions[0].version, 6);
    assert_eq!(versions[0].content, "v6");
    assert_eq!(versions[99].version, 105);
    assert_eq!(versions[99].content, "v105");
    Ok(())
}

// ---- API keys ----------------------------------------------------------------------------------

#[tokio::test]
async fn api_keys_create_lookup_validate_duplicate_and_revoke() -> TestResult {
    let db = db().await?;
    let hash = "ab".repeat(32);
    let scopes = vec!["user:*".to_owned()];
    let key = db
        .store
        .create_api_key("t1", "ci", &hash, &scopes, Trust::Agent, true)
        .await?;
    assert_eq!(key.tenant, "t1");
    assert_eq!(key.name, "ci");
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

// ---- Administrative/listing methods ------------------------------------------------------------

#[tokio::test]
async fn list_facts_filters_states_enriches_entities_and_paginates_by_id() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let object = entity(&db, &c, &scope, "London").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
    let mut replacement = new_fact(&scope, subject, "new", date(2022, 1));
    replacement.object_id = Some(object);
    let new = supersede(&db, &c, old, replacement).await?;
    let (_, independent) = add_fact(
        &db,
        &c,
        new_fact(&scope, subject, "independent", date(2023, 1)),
    )
    .await?;
    let mut held = new_fact(&scope, subject, "held", date(2020, 1));
    held.trust = Trust::External;
    held.status = FactStatus::Quarantined;
    add_fact(&db, &c, held).await?;

    let current = db
        .store
        .list_facts(&c, &scope, FactListState::Current, 10, None)
        .await?;
    assert_eq!(
        current
            .iter()
            .map(|view| view.fact.id)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([new, independent])
    );
    let past = db
        .store
        .list_facts(&c, &scope, FactListState::Past, 10, None)
        .await?;
    assert_eq!(
        past.iter().map(|view| view.fact.id).collect::<Vec<_>>(),
        vec![old]
    );
    let all = db
        .store
        .list_facts(&c, &scope, FactListState::All, 10, None)
        .await?;
    let mut expected = vec![old, new, independent];
    expected.sort_by(|a, b| b.cmp(a));
    assert_eq!(
        all.iter().map(|view| view.fact.id).collect::<Vec<_>>(),
        expected
    );
    let new_view = all
        .iter()
        .find(|view| view.fact.id == new)
        .expect("new view");
    assert_eq!(new_view.subject_name, "Ada");
    assert_eq!(new_view.subject_kind, "thing");
    assert_eq!(new_view.object_name.as_deref(), Some("London"));
    assert_eq!(new_view.object_kind.as_deref(), Some("thing"));

    let first_page = db
        .store
        .list_facts(&c, &scope, FactListState::All, 2, None)
        .await?;
    assert_eq!(first_page.len(), 2);
    let cursor = first_page.last().expect("two rows").fact.id;
    let second_page = db
        .store
        .list_facts(&c, &scope, FactListState::All, 2, Some(cursor))
        .await?;
    assert_eq!(second_page.len(), 1);
    assert_eq!(second_page[0].fact.id, expected[2]);
    assert_eq!(first_page[1].fact.id, cursor);
    assert!(second_page.iter().all(|view| view.fact.id < cursor));
    Ok(())
}

#[tokio::test]
async fn scope_summaries_and_highest_trust_include_all_storage_families() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let (_, old) = add_fact(&db, &c, new_fact(&scope, subject, "old", date(2020, 1))).await?;
    supersede(
        &db,
        &c,
        old,
        new_fact(&scope, subject, "new", date(2022, 1)),
    )
    .await?;
    let mut held = new_fact(&scope, subject, "held", date(2020, 1));
    held.trust = Trust::External;
    held.status = FactStatus::Quarantined;
    add_fact(&db, &c, held).await?;
    // An episode alone contributes to the maximum trust in a scope.
    new_episode(&db, &c, &scope, Trust::System).await?;
    db.store
        .put_block(&c, "bot", "persona", "p", 100, None)
        .await?;
    entity(&db, &c, &Scope::shared("entities-only"), "Only").await?;

    let summaries = db.store.scope_summaries(&c).await?;
    let user_summary = summaries
        .iter()
        .find(|summary| summary.scope == scope)
        .expect("user summary");
    assert_eq!(user_summary.current, 1);
    assert_eq!(user_summary.past, 1);
    assert_eq!(user_summary.quarantined, 1);
    assert_eq!(user_summary.entities, 1);
    assert_eq!(user_summary.blocks, 0);
    assert!(user_summary.last_recorded_at.is_some());
    let agent_summary = summaries
        .iter()
        .find(|summary| summary.scope == Scope::agent("bot"))
        .expect("agent summary");
    assert_eq!(agent_summary.blocks, 1);
    assert_eq!(agent_summary.entities, 0);
    let entity_summary = summaries
        .iter()
        .find(|summary| summary.scope == Scope::shared("entities-only"))
        .expect("entity-only summary");
    assert_eq!(entity_summary.entities, 1);
    assert_eq!(entity_summary.current, 0);

    assert_eq!(
        db.store.highest_trust_in_scope(&c, &scope).await?,
        Some(Trust::System)
    );
    assert_eq!(
        db.store
            .highest_trust_in_scope(&c, &Scope::shared("entities-only"))
            .await?,
        None
    );
    assert_eq!(
        db.store.highest_trust_in_scope(&ctx("t2"), &scope).await?,
        None
    );
    Ok(())
}

#[tokio::test]
async fn quarantine_prefixes_are_literal_composable_and_tenant_scoped() -> TestResult {
    let db = db().await?;
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
    let listed = |exact: Vec<Scope>, prefixes: Vec<&str>, tenant: &str| {
        let store = db.store.clone();
        let exact = exact.clone();
        let prefixes: Vec<String> = prefixes.iter().map(|prefix| (*prefix).to_owned()).collect();
        let tenant = tenant.to_owned();
        async move {
            let found = store
                .list_quarantined_matching(&ctx(&tenant), &exact, &prefixes, 10)
                .await?;
            Ok::<_, Error>(fact_ids(&found))
        }
    };

    assert_eq!(listed(vec![], vec!["user:"], "a").await?, ids);
    assert_eq!(listed(vec![], vec![""], "a").await?, ids);
    // `_` is literal, not a SQL LIKE wildcard that would also match user:axb.
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

#[tokio::test]
async fn subject_query_returns_latest_200_current_facts_oldest_first() -> TestResult {
    let db = db().await?;
    let c = ctx("t1");
    let scope = user("u1");
    let subject = entity(&db, &c, &scope, "Ada").await?;
    let ops = (0..201)
        .map(|i| FactOp::Add(new_fact(&scope, subject, &format!("f{i}"), date(2020, 1))))
        .collect();
    let episode_id = new_episode(&db, &c, &scope, Trust::User).await?;
    let ids = db.store.apply_commit(&c, episode_id, ops).await?;
    let current = db
        .store
        .current_facts_for_subject(&c, &scope, subject)
        .await?;
    assert_eq!(fact_ids(&current), ids[1..]);
    Ok(())
}

// ---- Invalid input and empty tenants -----------------------------------------------------------

#[tokio::test]
async fn empty_tenant_and_malformed_inputs_are_domain_errors() -> TestResult {
    let db = db().await?;
    let empty = ctx("");
    let scope = user("u1");
    assert!(is_invalid(db.store.list_blocks(&empty, "a").await));
    assert!(is_invalid(
        db.store.get_episode(&empty, Uuid::now_v7()).await
    ));
    assert!(is_invalid(db.store.scope_summaries(&empty).await));
    assert!(is_invalid(
        db.store
            .list_facts(&empty, &scope, FactListState::All, 10, None)
            .await
    ));
    assert!(is_invalid(
        db.store.highest_trust_in_scope(&empty, &scope).await
    ));
    assert!(is_invalid(
        db.store
            .search_facts_text(
                &empty,
                std::slice::from_ref(&scope),
                "x",
                &FactFilter::default(),
                10
            )
            .await
    ));
    assert!(is_invalid(
        db.store
            .create_api_key(
                "",
                "bad",
                &"ab".repeat(32),
                &["*".into()],
                Trust::User,
                false
            )
            .await
    ));

    let c = ctx("t1");
    assert!(is_invalid(
        db.store
            .insert_episode(&c, episode(&scope, Trust::User, "a\0b"))
            .await
    ));
    assert!(is_invalid(
        db.store.find_entity(&c, &scope, "thing", " ").await
    ));
    assert!(is_invalid(
        db.store
            .list_facts(&c, &scope, FactListState::All, 0, None)
            .await
    ));
    assert!(is_invalid(
        db.store
            .list_quarantined(&c, std::slice::from_ref(&scope), 0)
            .await
    ));
    assert!(is_invalid(
        db.store
            .search_facts_text(
                &c,
                std::slice::from_ref(&scope),
                "x",
                &FactFilter::default(),
                0
            )
            .await
    ));
    assert!(is_invalid(
        db.store.similar_entities(&c, &scope, &axis(0), 0).await
    ));

    let bad_entity = NewEntity {
        scope: scope.clone(),
        kind: "thing".into(),
        name: "bad".into(),
        aliases: vec![],
        embedding: vec![1.0; 3],
    };
    assert!(is_invalid(db.store.upsert_entity(&c, bad_entity).await));
    Ok(())
}
