use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;
use crate::types::*;

/// Persistence contract. Frozen after Wave 0: changes need an entry in docs/PROGRESS.md.
///
/// Every `ctx` method must run in a transaction with row-level security applied for
/// `ctx.tenant` (see docs/SPEC.md section 6). Reads never return another tenant's rows.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    // Episodes
    /// Inserts the episode and its `extract` job in one transaction. A repeated
    /// `idempotency_key` returns the existing episode and enqueues nothing.
    async fn insert_episode(&self, ctx: &RequestContext, episode: NewEpisode) -> Result<Episode>;
    async fn get_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<Episode>;
    /// Deletes the episode. Facts it alone supports are deleted; facts with other sources
    /// lose this source and stay. Entities left with no facts are deleted.
    async fn purge_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<PurgeReport>;
    /// Hard-deletes everything stored under `scope`, including sessions when it is an agent scope.
    async fn purge_scope(&self, ctx: &RequestContext, scope: &Scope) -> Result<PurgeReport>;

    // Extraction
    /// Atomically applies `ops` and sets `episodes.processed_at`. If the episode is already
    /// processed this is a no-op returning an empty list, which makes job retries idempotent.
    /// Returns ids of inserted facts.
    async fn apply_extraction(
        &self,
        ctx: &RequestContext,
        episode_id: Uuid,
        ops: Vec<FactOp>,
    ) -> Result<Vec<Uuid>>;

    // Entities
    /// Returns the entity with the same scope, kind and normalized name, or creates it.
    async fn upsert_entity(&self, ctx: &RequestContext, entity: NewEntity) -> Result<Entity>;
    async fn get_entity(&self, ctx: &RequestContext, id: Uuid) -> Result<Entity>;
    /// Nearest entities by cosine similarity, highest first, with similarity in 0..=1.
    async fn similar_entities(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        embedding: &[f32],
        limit: i64,
    ) -> Result<Vec<(Entity, f64)>>;
    async fn add_entity_alias(&self, ctx: &RequestContext, id: Uuid, alias: &str) -> Result<()>;

    // Facts
    async fn get_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact>;
    /// All versions in the supersession chain containing `id`, oldest first.
    async fn fact_history(&self, ctx: &RequestContext, id: Uuid) -> Result<Vec<Fact>>;
    /// Marks the fact retracted by setting `invalidated_at`. Writes an audit event.
    async fn invalidate_fact(&self, ctx: &RequestContext, id: Uuid, reason: &str) -> Result<Fact>;
    /// Currently valid, active facts about `subject_id`, for reconciliation.
    async fn current_facts_for_subject(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        subject_id: Uuid,
    ) -> Result<Vec<Fact>>;
    /// Vector search over active facts passing `filter`. Must still return up to `limit`
    /// rows under selective filters (use pgvector iterative scan).
    async fn search_facts_vector(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        embedding: &[f32],
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>>;
    /// Full-text search over active facts passing `filter`.
    async fn search_facts_text(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>>;
    /// Full-text search over episodes not yet processed, so writes are readable immediately.
    /// `external` episodes are excluded: they could carry poisoned content.
    async fn search_pending_episodes(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        limit: i64,
    ) -> Result<Vec<ScoredEpisode>>;

    // Quarantine
    async fn list_quarantined(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        limit: i64,
    ) -> Result<Vec<Fact>>;
    async fn approve_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact>;
    async fn reject_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact>;

    // Graph
    /// Entities and active edges within `depth` hops of `root` through `scopes`. `depth` is 1..=5.
    async fn neighbors(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        root: Uuid,
        depth: i32,
        filter: &FactFilter,
    ) -> Result<GraphNeighborhood>;

    // Blocks
    async fn list_blocks(&self, ctx: &RequestContext, agent_id: &str) -> Result<Vec<Block>>;
    async fn get_block(&self, ctx: &RequestContext, agent_id: &str, label: &str) -> Result<Block>;
    /// `expected_version = None` creates the block and fails with `VersionConflict` if it
    /// exists. `Some(v)` updates only when the current version is `v`, and appends a version row.
    async fn put_block(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        label: &str,
        content: &str,
        max_chars: i32,
        expected_version: Option<i64>,
    ) -> Result<Block>;
    async fn block_versions(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        label: &str,
    ) -> Result<Vec<BlockVersion>>;

    // Sessions
    async fn create_session(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        scope: &Scope,
    ) -> Result<Session>;
    async fn get_session(&self, ctx: &RequestContext, id: Uuid) -> Result<Session>;
    /// Assigns the next `seq` and appends. Safe under concurrent callers.
    async fn append_message(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
        role: &str,
        content: serde_json::Value,
    ) -> Result<SessionMessage>;
    /// Messages with `seq` greater than the session's compaction point, oldest first.
    async fn load_messages(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
    ) -> Result<Vec<SessionMessage>>;
    /// Replaces messages up to and including `through_seq` with `summary`.
    async fn compact_session(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
        through_seq: i64,
        summary: &str,
    ) -> Result<()>;
    async fn session_summary(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
    ) -> Result<Option<String>>;

    // Jobs: cross-tenant, used only by the worker
    /// Claims one due job with `FOR UPDATE SKIP LOCKED`, leased until now + `lease_secs`.
    /// Jobs whose lease expired are claimable, which recovers work from crashed workers.
    async fn claim_job(&self, worker_id: &str, lease_secs: i64) -> Result<Option<Job>>;
    async fn complete_job(&self, job_id: Uuid) -> Result<()>;
    /// Reschedules at `run_after`, or marks `dead` once `attempts` reaches `max_attempts`.
    async fn fail_job(
        &self,
        job_id: Uuid,
        error: &str,
        run_after: DateTime<Utc>,
        max_attempts: i32,
    ) -> Result<()>;

    // API keys: cross-tenant, looked up before the tenant is known
    /// `key_hash` is the hex SHA-256 of the presented token. Revoked keys return `None`.
    async fn find_api_key(&self, key_hash: &str) -> Result<Option<ApiKey>>;
    async fn create_api_key(
        &self,
        tenant: &str,
        name: &str,
        key_hash: &str,
        scopes: &[String],
        max_trust: Trust,
        is_admin: bool,
    ) -> Result<ApiKey>;
    async fn revoke_api_key(&self, id: Uuid) -> Result<()>;

    /// Verifies database connectivity for `/readyz`.
    async fn ping(&self) -> Result<()>;
}
