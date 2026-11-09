//! Retrieval pipeline scenarios against real SQLite files, with no container dependency.

#![cfg(feature = "integration-sqlite")]

use std::sync::Arc;

use memoryos_core::{RequestContext, Scope, EMBEDDING_DIMS};
use memoryos_embed::TestEmbedder;
use memoryos_retrieve::RecallResponse;
use memoryos_store::SqliteStore;
use tempfile::TempDir;
use uuid::Uuid;

mod common;

struct Fixture {
    store: Arc<SqliteStore>,
    embedder: Arc<TestEmbedder>,
    _dir: TempDir,
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("recall.sqlite");
    let url = format!("sqlite://{}", path.display());
    let store = Arc::new(
        SqliteStore::connect(&url)
            .await
            .expect("connect SQLite store"),
    );
    store.migrate().await.expect("migrate SQLite store");
    assert!(path.exists(), "the fixture uses a real SQLite file");

    let embedder = Arc::new(TestEmbedder::new());
    let mut query = vec![0.0; EMBEDDING_DIMS];
    query[0] = 1.0;
    embedder.set_vector(common::QUERY, query);
    Fixture {
        store,
        embedder,
        _dir: dir,
    }
}

impl Fixture {
    fn scenario(&self) -> common::Scenario {
        common::Scenario::new(
            self.store.clone(),
            self.embedder.clone(),
            RequestContext {
                tenant: "tenant-a".into(),
                actor: "test".into(),
            },
            Scope::user("ada"),
        )
    }
}

fn shape(response: &RecallResponse) -> Vec<(Uuid, u64)> {
    response
        .facts
        .iter()
        .map(|scored| (scored.fact.id, scored.score.to_bits()))
        .collect()
}

#[tokio::test]
async fn sqlite_recall_is_deterministic_and_finds_a_fact_only_by_vector() {
    let fixture = fixture().await;
    let scenario = fixture.scenario();
    let (_, first) = scenario.finds_a_fact_only_by_vector().await;
    let second = scenario.recall(common::QUERY).await;

    assert_eq!(shape(&first), shape(&second));
    assert_eq!(
        fixture.embedder.batches(),
        [[common::QUERY], [common::QUERY]],
        "each recall uses the same deterministic query embedding input"
    );
}

#[tokio::test]
async fn sqlite_superseded_fact_is_absent_now_and_present_with_as_of() {
    let fixture = fixture().await;
    fixture
        .scenario()
        .superseded_fact_is_absent_now_and_present_with_as_of()
        .await;
}

#[tokio::test]
async fn sqlite_other_tenant_is_absent() {
    let fixture = fixture().await;
    fixture.scenario().other_tenant_is_absent().await;
}
