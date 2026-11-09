//! Backend-independent retrieval pipeline scenarios, run against PostgreSQL and SQLite fixtures.

use std::collections::BTreeSet;
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use memoryos_core::{
    FactFilter, FactOp, FactStatus, MemoryStore, NewEntity, NewEpisode, NewFact, RequestContext,
    Scope, ScoredFact, Trust, EMBEDDING_DIMS,
};
use memoryos_embed::TestEmbedder;
use memoryos_retrieve::{RecallRequest, RecallResponse, Retriever};
use serde_json::json;
use uuid::Uuid;

pub const QUERY: &str = "zebra";

pub struct Scenario {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<TestEmbedder>,
    ctx: RequestContext,
    scope: Scope,
}

impl Scenario {
    pub fn new(
        store: Arc<dyn MemoryStore>,
        embedder: Arc<TestEmbedder>,
        ctx: RequestContext,
        scope: Scope,
    ) -> Self {
        Self {
            store,
            embedder,
            ctx,
            scope,
        }
    }

    async fn entity_in(&self, ctx: &RequestContext, scope: &Scope, name: &str) -> Uuid {
        self.store
            .upsert_entity(
                ctx,
                NewEntity {
                    scope: scope.clone(),
                    kind: "thing".into(),
                    name: name.into(),
                    aliases: Vec::new(),
                    embedding: at_angle(0.0),
                },
            )
            .await
            .expect("upsert entity")
            .id
    }

    async fn entity(&self, name: &str) -> Uuid {
        self.entity_in(&self.ctx, &self.scope, name).await
    }

    async fn apply_in(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        trust: Trust,
        ops: Vec<FactOp>,
    ) -> Vec<Uuid> {
        let (source, _) = self
            .store
            .claim_episode(ctx, episode(scope, trust, "source"))
            .await
            .expect("insert episode");
        self.store
            .apply_commit(ctx, source.id, ops)
            .await
            .expect("apply commit")
    }

    async fn add(&self, subject: Uuid, statement: &str, degrees: f32) -> Uuid {
        let fact = new_fact(&self.scope, subject, statement, degrees);
        self.apply_in(&self.ctx, &self.scope, Trust::User, vec![FactOp::Add(fact)])
            .await[0]
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

    pub async fn recall(&self, query: &str) -> RecallResponse {
        Retriever::new(self.store.clone(), self.embedder.clone())
            .recall(&self.ctx, self.request(query))
            .await
            .expect("recall")
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

    pub async fn finds_a_fact_only_by_vector(&self) -> (Uuid, RecallResponse) {
        let subject = self.entity("a").await;
        let target = self
            .add(subject, "nothing in common with the query", 1.0)
            .await;

        let filter = FactFilter::default();
        assert!(self.vector_hits(&filter).await.contains(&target));
        assert!(!self.text_hits(&filter).await.contains(&target));

        let response = self.recall(QUERY).await;
        assert!(ids(&response.facts).contains(&target));
        (target, response)
    }

    pub async fn superseded_fact_is_absent_now_and_present_with_as_of(&self) {
        let subject = self.entity("a").await;
        let old = self.add(subject, "the zebra lives in Paris", 1.0).await;
        let mut replacement = new_fact(&self.scope, subject, "the zebra lives in Rome", 1.0);
        replacement.valid_from = date(2022);
        let op = FactOp::Supersede {
            old_fact_id: old,
            new: replacement,
        };
        let new = self
            .apply_in(&self.ctx, &self.scope, Trust::User, vec![op])
            .await[0];

        let now = ids(&self.recall(QUERY).await.facts);
        assert!(now.contains(&new) && !now.contains(&old));

        let mut past = self.request(QUERY);
        past.as_of = Some(date(2021));
        let then = Retriever::new(self.store.clone(), self.embedder.clone())
            .recall(&self.ctx, past)
            .await
            .expect("recall");
        let then = ids(&then.facts);
        assert!(then.contains(&old) && !then.contains(&new));
    }

    pub async fn other_tenant_is_absent(&self) {
        let other = RequestContext {
            tenant: "tenant-b".into(),
            actor: "test".into(),
        };
        let mine = self
            .add(self.entity("a").await, "a zebra was seen", 1.0)
            .await;
        let subject = self.entity_in(&other, &self.scope, "a").await;
        let fact = new_fact(&self.scope, subject, "a zebra was seen", 1.0);
        let theirs = self
            .apply_in(&other, &self.scope, Trust::User, vec![FactOp::Add(fact)])
            .await[0];

        let response = self.recall(QUERY).await;
        assert_eq!(ids(&response.facts), BTreeSet::from([mine]));

        let response = Retriever::new(self.store.clone(), self.embedder.clone())
            .recall(&other, self.request(QUERY))
            .await
            .expect("recall");
        assert_eq!(ids(&response.facts), BTreeSet::from([theirs]));
    }
}

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
        expires_at: None,
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

fn ids(facts: &[ScoredFact]) -> BTreeSet<Uuid> {
    facts.iter().map(|scored| scored.fact.id).collect()
}
