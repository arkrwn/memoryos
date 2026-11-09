//! Ingest pipeline scenarios against real SQLite files, with no container dependency.

#![cfg(feature = "integration-sqlite")]

use std::sync::Arc;

use memoryos_embed::TestEmbedder;
use memoryos_ingest::HostCommitter;
use memoryos_store::SqliteStore;
use tempfile::TempDir;

mod common;

struct Fixture {
    store: Arc<SqliteStore>,
    embedder: Arc<TestEmbedder>,
    host: HostCommitter,
    _dir: TempDir,
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("ingest.sqlite");
    let url = format!("sqlite://{}", path.display());
    let store = Arc::new(
        SqliteStore::connect(&url)
            .await
            .expect("connect SQLite store"),
    );
    store.migrate().await.expect("migrate SQLite store");
    assert!(path.exists(), "the fixture uses a real SQLite file");

    let embedder = Arc::new(TestEmbedder::new());
    let host = HostCommitter::new(store.clone(), embedder.clone());
    Fixture {
        store,
        embedder,
        host,
        _dir: dir,
    }
}

#[tokio::test]
async fn sqlite_host_commit_adds_facts_in_one_embedding_batch() {
    let fixture = fixture().await;
    common::host_commit_adds_facts_in_one_embedding_batch(
        fixture.store.clone(),
        &fixture.host,
        &fixture.embedder,
    )
    .await;
}

#[tokio::test]
async fn sqlite_host_supersede_closes_the_old_fact() {
    let fixture = fixture().await;
    common::host_supersede_closes_the_old_fact(fixture.store.clone(), &fixture.host).await;
}

#[tokio::test]
async fn sqlite_matching_never_crosses_scopes_or_tenants() {
    let fixture = fixture().await;
    common::matching_never_crosses_scopes_or_tenants(fixture.store.clone(), &fixture.host).await;
}
