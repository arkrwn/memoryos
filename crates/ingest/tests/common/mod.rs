//! Backend-independent ingest pipeline scenarios, run against PostgreSQL and SQLite fixtures.

use std::sync::Arc;

use chrono::{TimeZone, Utc};
use memoryos_core::{FactFilter, FactStatus, MemoryStore, RequestContext, Scope, Trust};
use memoryos_embed::{vector_for, TestEmbedder};
use memoryos_ingest::{CommitFact, CommitRequest, CommittedFact, EntityRef, HostCommitter};
use uuid::Uuid;

fn ctx() -> RequestContext {
    tenant_ctx("acme")
}

fn tenant_ctx(tenant: &str) -> RequestContext {
    RequestContext {
        tenant: tenant.into(),
        actor: "tester".into(),
    }
}

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

async fn commit_one(host: &HostCommitter, scope: &Scope, text: &str, subject: &str) -> Uuid {
    let request = commit_request(
        scope,
        &format!("source of {text}"),
        vec![add_fact(text, subject)],
    );
    let response = host
        .commit(&ctx(), request, Trust::Agent)
        .await
        .expect("commit");
    response.facts[0].id
}

async fn commit_as(
    host: &HostCommitter,
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
    let mut response = host
        .commit(ctx, request, Trust::Agent)
        .await
        .expect("commit");
    assert_eq!(response.facts.len(), 1);
    response.facts.remove(0)
}

pub async fn host_commit_adds_facts_in_one_embedding_batch(
    store: Arc<dyn MemoryStore>,
    host: &HostCommitter,
    embedder: &TestEmbedder,
) {
    let scope = Scope::agent("a1");
    let mut knows = add_fact("Ada knows Bob", "Ada");
    knows.object = Some(EntityRef {
        name: "Bob".into(),
        kind: "person".into(),
    });
    let mut tea = add_fact("Ada likes tea", "ada");
    tea.valid_from = Some(
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0)
            .single()
            .expect("valid date"),
    );
    let response = host
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
        .all(|fact| fact.status == FactStatus::Active));
    let episode = store
        .get_episode(&ctx(), response.episode_id)
        .await
        .expect("episode");
    assert_eq!(
        (episode.trust, episode.content.as_str()),
        (Trust::Agent, "chat text")
    );
    assert!(episode.processed_at.is_some());

    let knows = store
        .get_fact(&ctx(), response.facts[0].id)
        .await
        .expect("fact");
    assert_eq!(knows.valid_from, episode.occurred_at);
    assert!(knows.object_id.is_some());
    let tea = store
        .get_fact(&ctx(), response.facts[1].id)
        .await
        .expect("fact");
    assert_eq!(
        tea.valid_from,
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0)
            .single()
            .expect("valid date")
    );
    assert_eq!(tea.subject_id, knows.subject_id);

    // One provider call, statements first, then each distinct entity.
    assert_eq!(
        embedder.batches(),
        [["Ada knows Bob", "Ada likes tea", "ada", "bob"]]
    );
}

pub async fn host_supersede_closes_the_old_fact(store: Arc<dyn MemoryStore>, host: &HostCommitter) {
    let scope = Scope::agent("a1");
    let old_id = commit_one(host, &scope, "Ada lives in Rome", "Ada").await;

    let mut fact = add_fact("Ada lives in Oslo", "Ada");
    fact.supersedes = Some(old_id);
    let response = host
        .commit(
            &ctx(),
            commit_request(&scope, "moved", vec![fact]),
            Trust::Agent,
        )
        .await
        .expect("commit");
    assert_eq!(response.facts[0].status, FactStatus::Active);

    let old = store.get_fact(&ctx(), old_id).await.expect("old");
    assert_eq!(old.superseded_by, Some(response.facts[0].id));
    assert!(old.valid_to.is_some());
    let recalled = store
        .search_facts_vector(
            &ctx(),
            std::slice::from_ref(&scope),
            &vector_for("Ada lives in Rome"),
            &FactFilter::default(),
            20,
        )
        .await
        .expect("search")
        .into_iter()
        .map(|scored| scored.fact.id)
        .collect::<Vec<_>>();
    assert_eq!(recalled, [response.facts[0].id]);
}

pub async fn matching_never_crosses_scopes_or_tenants(
    store: Arc<dyn MemoryStore>,
    host: &HostCommitter,
) {
    let scope = Scope::agent("a1");
    let globex = tenant_ctx("globex");
    let other_scope = commit_as(host, &ctx(), &Scope::agent("a2"), "Ada plays go", "Ada").await;
    let other_tenant = commit_as(host, &globex, &scope, "Ada plays go", "Ada").await;

    let mine = commit_as(host, &ctx(), &scope, "Ada plays go", "Ada").await;
    assert!(!mine.corroborated);
    assert_ne!(mine.id, other_scope.id);
    assert_ne!(mine.id, other_tenant.id);
    assert!(mine.related.is_empty(), "{:?}", mine.related);
    for (ctx, foreign) in [(ctx(), other_scope.id), (globex, other_tenant.id)] {
        let fact = store.get_fact(&ctx, foreign).await.expect("fact");
        assert_eq!(fact.source_episode_ids.len(), 1);
    }
}
