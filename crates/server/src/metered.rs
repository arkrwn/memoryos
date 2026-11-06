//! Counts job outcomes for `/metrics`. The worker reports them only through the store's fenced job
//! methods, so the count wraps those, and every other method passes through unchanged.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use memoryos_core::{
    ApiKey, Block, BlockVersion, Entity, Episode, Fact, FactFilter, FactOp, GraphNeighborhood, Job,
    MemoryStore, NewEntity, NewEpisode, PurgeReport, RequestContext, Result, Scope, ScoredEpisode,
    ScoredFact, Session, SessionMessage, Trust,
};
use uuid::Uuid;

use crate::telemetry::Metrics;

pub struct MeteredStore {
    inner: Arc<dyn MemoryStore>,
    metrics: Metrics,
}

impl MeteredStore {
    pub fn new(inner: Arc<dyn MemoryStore>, metrics: Metrics) -> Self {
        Self { inner, metrics }
    }
}

#[async_trait]
impl MemoryStore for MeteredStore {
    async fn complete_job(&self, job: &Job, worker_id: &str) -> Result<bool> {
        let applied = self.inner.complete_job(job, worker_id).await?;
        if applied {
            self.metrics.job_finished("completed");
        }
        Ok(applied)
    }

    async fn fail_job(
        &self,
        job: &Job,
        worker_id: &str,
        error: &str,
        run_after: DateTime<Utc>,
        max_attempts: i32,
    ) -> Result<bool> {
        let applied = self
            .inner
            .fail_job(job, worker_id, error, run_after, max_attempts)
            .await?;
        if applied {
            self.metrics.job_finished("failed");
        }
        Ok(applied)
    }

    async fn insert_episode(&self, ctx: &RequestContext, episode: NewEpisode) -> Result<Episode> {
        self.inner.insert_episode(ctx, episode).await
    }
    async fn insert_unqueued_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> Result<Episode> {
        self.inner.insert_unqueued_episode(ctx, episode).await
    }
    async fn get_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<Episode> {
        self.inner.get_episode(ctx, id).await
    }
    async fn purge_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<PurgeReport> {
        self.inner.purge_episode(ctx, id).await
    }
    async fn purge_scope(&self, ctx: &RequestContext, scope: &Scope) -> Result<PurgeReport> {
        self.inner.purge_scope(ctx, scope).await
    }
    async fn apply_extraction(
        &self,
        ctx: &RequestContext,
        episode_id: Uuid,
        ops: Vec<FactOp>,
    ) -> Result<Vec<Uuid>> {
        self.inner.apply_extraction(ctx, episode_id, ops).await
    }
    async fn upsert_entity(&self, ctx: &RequestContext, entity: NewEntity) -> Result<Entity> {
        self.inner.upsert_entity(ctx, entity).await
    }
    async fn get_entity(&self, ctx: &RequestContext, id: Uuid) -> Result<Entity> {
        self.inner.get_entity(ctx, id).await
    }
    async fn similar_entities(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        embedding: &[f32],
        limit: i64,
    ) -> Result<Vec<(Entity, f64)>> {
        self.inner
            .similar_entities(ctx, scope, embedding, limit)
            .await
    }
    async fn add_entity_alias(&self, ctx: &RequestContext, id: Uuid, alias: &str) -> Result<()> {
        self.inner.add_entity_alias(ctx, id, alias).await
    }
    async fn find_entity(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        kind: &str,
        name: &str,
    ) -> Result<Option<Entity>> {
        self.inner.find_entity(ctx, scope, kind, name).await
    }
    async fn get_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        self.inner.get_fact(ctx, id).await
    }
    async fn facts_for_episode(&self, ctx: &RequestContext, episode_id: Uuid) -> Result<Vec<Fact>> {
        self.inner.facts_for_episode(ctx, episode_id).await
    }
    async fn fact_history(&self, ctx: &RequestContext, id: Uuid) -> Result<Vec<Fact>> {
        self.inner.fact_history(ctx, id).await
    }
    async fn invalidate_fact(&self, ctx: &RequestContext, id: Uuid, reason: &str) -> Result<Fact> {
        self.inner.invalidate_fact(ctx, id, reason).await
    }
    async fn current_facts_for_subject(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        subject_id: Uuid,
    ) -> Result<Vec<Fact>> {
        self.inner
            .current_facts_for_subject(ctx, scope, subject_id)
            .await
    }
    async fn search_facts_vector(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        embedding: &[f32],
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        self.inner
            .search_facts_vector(ctx, scopes, embedding, filter, limit)
            .await
    }
    async fn search_facts_text(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        self.inner
            .search_facts_text(ctx, scopes, query, filter, limit)
            .await
    }
    async fn search_pending_episodes(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        limit: i64,
    ) -> Result<Vec<ScoredEpisode>> {
        self.inner
            .search_pending_episodes(ctx, scopes, query, limit)
            .await
    }
    async fn list_quarantined(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        limit: i64,
    ) -> Result<Vec<Fact>> {
        self.inner.list_quarantined(ctx, scopes, limit).await
    }
    async fn approve_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        self.inner.approve_fact(ctx, id).await
    }
    async fn reject_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        self.inner.reject_fact(ctx, id).await
    }
    async fn neighbors(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        root: Uuid,
        depth: i32,
        filter: &FactFilter,
        edge_limit: i64,
    ) -> Result<GraphNeighborhood> {
        self.inner
            .neighbors(ctx, scopes, root, depth, filter, edge_limit)
            .await
    }
    async fn list_blocks(&self, ctx: &RequestContext, agent_id: &str) -> Result<Vec<Block>> {
        self.inner.list_blocks(ctx, agent_id).await
    }
    async fn get_block(&self, ctx: &RequestContext, agent_id: &str, label: &str) -> Result<Block> {
        self.inner.get_block(ctx, agent_id, label).await
    }
    async fn put_block(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        label: &str,
        content: &str,
        max_chars: i32,
        expected_version: Option<i64>,
    ) -> Result<Block> {
        self.inner
            .put_block(ctx, agent_id, label, content, max_chars, expected_version)
            .await
    }
    async fn block_versions(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        label: &str,
    ) -> Result<Vec<BlockVersion>> {
        self.inner.block_versions(ctx, agent_id, label).await
    }
    async fn create_session(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        scope: &Scope,
    ) -> Result<Session> {
        self.inner.create_session(ctx, agent_id, scope).await
    }
    async fn get_session(&self, ctx: &RequestContext, id: Uuid) -> Result<Session> {
        self.inner.get_session(ctx, id).await
    }
    async fn append_message(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
        role: &str,
        content: serde_json::Value,
    ) -> Result<SessionMessage> {
        self.inner
            .append_message(ctx, session_id, role, content)
            .await
    }
    async fn load_messages(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
    ) -> Result<Vec<SessionMessage>> {
        self.inner.load_messages(ctx, session_id).await
    }
    async fn compact_session(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
        through_seq: i64,
        summary: &str,
    ) -> Result<()> {
        self.inner
            .compact_session(ctx, session_id, through_seq, summary)
            .await
    }
    async fn session_summary(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
    ) -> Result<Option<String>> {
        self.inner.session_summary(ctx, session_id).await
    }
    async fn claim_job(&self, worker_id: &str, lease_secs: i64) -> Result<Option<Job>> {
        self.inner.claim_job(worker_id, lease_secs).await
    }
    async fn renew_job(&self, job: &Job, worker_id: &str, lease_secs: i64) -> Result<bool> {
        self.inner.renew_job(job, worker_id, lease_secs).await
    }
    async fn release_job(
        &self,
        job: &Job,
        worker_id: &str,
        run_after: DateTime<Utc>,
    ) -> Result<bool> {
        self.inner.release_job(job, worker_id, run_after).await
    }
    async fn find_api_key(&self, key_hash: &str) -> Result<Option<ApiKey>> {
        self.inner.find_api_key(key_hash).await
    }
    async fn create_api_key(
        &self,
        tenant: &str,
        name: &str,
        key_hash: &str,
        scopes: &[String],
        max_trust: Trust,
        is_admin: bool,
    ) -> Result<ApiKey> {
        self.inner
            .create_api_key(tenant, name, key_hash, scopes, max_trust, is_admin)
            .await
    }
    async fn revoke_api_key(&self, id: Uuid) -> Result<()> {
        self.inner.revoke_api_key(id).await
    }
    async fn ping(&self) -> Result<()> {
        self.inner.ping().await
    }
}
