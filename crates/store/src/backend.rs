use async_trait::async_trait;
use memoryos_core::{
    ApiKey, Block, BlockVersion, Entity, Episode, Fact, FactFilter, FactListState, FactOp,
    FactView, GraphNeighborhood, MemoryStore, NewEntity, NewEpisode, PurgeReport, RequestContext,
    Result, Scope, ScopeSummary, ScoredFact, Trust,
};
use uuid::Uuid;

use crate::{PgStore, SqliteStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Postgres,
    Sqlite,
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "postgres" | "postgresql" => Ok(Self::Postgres),
            "sqlite" => Ok(Self::Sqlite),
            _ => Err("MEMORYOS_STORE must be postgres or sqlite".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Store {
    Postgres(PgStore),
    Sqlite(SqliteStore),
}

macro_rules! dispatch {
    ($self:ident, $method:ident($($arg:ident),*)) => {
        match $self {
            Store::Postgres(store) => store.$method($($arg),*).await,
            Store::Sqlite(store) => store.$method($($arg),*).await,
        }
    };
}

impl Store {
    pub async fn connect(
        backend: Backend,
        database_url: &str,
        max_connections: u32,
    ) -> Result<Self> {
        if max_connections == 0 {
            return Err(memoryos_core::Error::InvalidInput(
                "max_connections must be at least 1".into(),
            ));
        }
        match backend {
            Backend::Postgres => PgStore::connect(database_url, max_connections)
                .await
                .map(Self::Postgres),
            Backend::Sqlite => SqliteStore::connect(database_url).await.map(Self::Sqlite),
        }
    }

    pub async fn migrate(&self) -> Result<()> {
        dispatch!(self, migrate())
    }

    pub async fn scope_summaries(&self, ctx: &RequestContext) -> Result<Vec<ScopeSummary>> {
        dispatch!(self, scope_summaries(ctx))
    }

    pub async fn list_facts(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        state: FactListState,
        limit: i64,
        before: Option<Uuid>,
    ) -> Result<Vec<FactView>> {
        dispatch!(self, list_facts(ctx, scope, state, limit, before))
    }

    pub async fn list_quarantined_matching(
        &self,
        ctx: &RequestContext,
        exact: &[Scope],
        prefixes: &[String],
        limit: i64,
    ) -> Result<Vec<Fact>> {
        dispatch!(self, list_quarantined_matching(ctx, exact, prefixes, limit))
    }

    pub async fn highest_trust_in_scope(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
    ) -> Result<Option<Trust>> {
        dispatch!(self, highest_trust_in_scope(ctx, scope))
    }
}

#[async_trait]
impl MemoryStore for Store {
    async fn claim_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> Result<(Episode, bool)> {
        dispatch!(self, claim_episode(ctx, episode))
    }

    async fn discard_unprocessed_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<bool> {
        dispatch!(self, discard_unprocessed_episode(ctx, id))
    }

    async fn get_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<Episode> {
        dispatch!(self, get_episode(ctx, id))
    }

    async fn purge_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<PurgeReport> {
        dispatch!(self, purge_episode(ctx, id))
    }

    async fn purge_scope(&self, ctx: &RequestContext, scope: &Scope) -> Result<PurgeReport> {
        dispatch!(self, purge_scope(ctx, scope))
    }

    async fn apply_commit(
        &self,
        ctx: &RequestContext,
        episode_id: Uuid,
        ops: Vec<FactOp>,
    ) -> Result<Vec<Uuid>> {
        dispatch!(self, apply_commit(ctx, episode_id, ops))
    }

    async fn upsert_entity(&self, ctx: &RequestContext, entity: NewEntity) -> Result<Entity> {
        dispatch!(self, upsert_entity(ctx, entity))
    }

    async fn get_entity(&self, ctx: &RequestContext, id: Uuid) -> Result<Entity> {
        dispatch!(self, get_entity(ctx, id))
    }

    async fn similar_entities(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        embedding: &[f32],
        limit: i64,
    ) -> Result<Vec<(Entity, f64)>> {
        dispatch!(self, similar_entities(ctx, scope, embedding, limit))
    }

    async fn add_entity_alias(
        &self,
        ctx: &RequestContext,
        id: Uuid,
        alias: &str,
        source_episode: Option<Uuid>,
    ) -> Result<()> {
        dispatch!(self, add_entity_alias(ctx, id, alias, source_episode))
    }

    async fn find_entity(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        kind: &str,
        name: &str,
    ) -> Result<Option<Entity>> {
        dispatch!(self, find_entity(ctx, scope, kind, name))
    }

    async fn get_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        dispatch!(self, get_fact(ctx, id))
    }

    async fn facts_for_episode(&self, ctx: &RequestContext, episode_id: Uuid) -> Result<Vec<Fact>> {
        dispatch!(self, facts_for_episode(ctx, episode_id))
    }

    async fn fact_history(&self, ctx: &RequestContext, id: Uuid) -> Result<Vec<Fact>> {
        dispatch!(self, fact_history(ctx, id))
    }

    async fn invalidate_fact(&self, ctx: &RequestContext, id: Uuid, reason: &str) -> Result<Fact> {
        dispatch!(self, invalidate_fact(ctx, id, reason))
    }

    async fn current_fact(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        id: Uuid,
    ) -> Result<Option<Fact>> {
        dispatch!(self, current_fact(ctx, scope, id))
    }

    async fn current_facts_for_subject(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        subject_id: Uuid,
    ) -> Result<Vec<Fact>> {
        dispatch!(self, current_facts_for_subject(ctx, scope, subject_id))
    }

    async fn search_facts_vector(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        embedding: &[f32],
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        dispatch!(
            self,
            search_facts_vector(ctx, scopes, embedding, filter, limit)
        )
    }

    async fn search_facts_text(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        dispatch!(self, search_facts_text(ctx, scopes, query, filter, limit))
    }

    async fn list_quarantined(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        limit: i64,
    ) -> Result<Vec<Fact>> {
        dispatch!(self, list_quarantined(ctx, scopes, limit))
    }

    async fn approve_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        dispatch!(self, approve_fact(ctx, id))
    }

    async fn reject_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        dispatch!(self, reject_fact(ctx, id))
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
        dispatch!(
            self,
            neighbors(ctx, scopes, root, depth, filter, edge_limit)
        )
    }

    async fn list_blocks(&self, ctx: &RequestContext, agent_id: &str) -> Result<Vec<Block>> {
        dispatch!(self, list_blocks(ctx, agent_id))
    }

    async fn get_block(&self, ctx: &RequestContext, agent_id: &str, label: &str) -> Result<Block> {
        dispatch!(self, get_block(ctx, agent_id, label))
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
        dispatch!(
            self,
            put_block(ctx, agent_id, label, content, max_chars, expected_version)
        )
    }

    async fn block_versions(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        label: &str,
    ) -> Result<Vec<BlockVersion>> {
        dispatch!(self, block_versions(ctx, agent_id, label))
    }

    async fn find_api_key(&self, key_hash: &str) -> Result<Option<ApiKey>> {
        dispatch!(self, find_api_key(key_hash))
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
        dispatch!(
            self,
            create_api_key(tenant, name, key_hash, scopes, max_trust, is_admin)
        )
    }

    async fn revoke_api_key(&self, id: Uuid) -> Result<()> {
        dispatch!(self, revoke_api_key(id))
    }

    async fn ping(&self) -> Result<()> {
        dispatch!(self, ping())
    }
}
