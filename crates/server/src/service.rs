//! Every operation of the API, with its authorization. REST handlers and MCP tools both call these,
//! so the two surfaces cannot disagree on a check (SPEC sections 6 and 14).

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use memoryos_agent::{AgentRuntime, MemoryRecall, TurnResult};
use memoryos_core::{
    Block, BlockVersion, Entity, Episode, Fact, FactFilter, GraphNeighborhood, MemoryStore,
    NewEpisode, PurgeReport, RequestContext, Result, Scope, ScoredFact, Session, SessionMessage,
    Trust,
};
use memoryos_ingest::{CommitRequest, CommitResponse, HostCommitter};
use memoryos_retrieve::{RecallRequest, RecallResponse, Retriever};
use memoryos_store::PgStore;
use serde::Deserialize;
use uuid::Uuid;

use crate::auth::Caller;
use crate::config::Mode;
use crate::error::{ApiError, ApiResult};
use crate::telemetry::Metrics;

const MAX_CONTENT_CHARS: usize = 100_000;
const DEFAULT_MAX_BLOCK_CHARS: i32 = 2000;
const MAX_BLOCK_CHARS: i32 = 100_000;
const DEFAULT_CANDIDATES: i64 = 10;
const DEFAULT_QUARANTINE_PAGE: i64 = 50;
const MAX_QUARANTINE_PAGE: i64 = 200;
const DEFAULT_DEPTH: i32 = 1;
/// Edges one neighbors request may read. A hub entity would otherwise answer with the store's cap
/// of 5000, each edge carrying its full statement.
const NEIGHBORS_EDGE_LIMIT: i64 = 500;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidatesRequest {
    pub scopes: Vec<Scope>,
    pub statement: String,
    #[serde(default)]
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PutBlockRequest {
    pub content: String,
    #[serde(default)]
    pub max_chars: Option<i32>,
    #[serde(default)]
    pub expected_version: Option<i64>,
}

pub struct Services {
    pub store: Arc<dyn MemoryStore>,
    /// The same database as `store`. Prefix listing is not part of `MemoryStore`.
    pub pg: PgStore,
    pub retriever: Arc<Retriever>,
    pub committer: HostCommitter,
    /// `Some` exactly in worker mode, the only mode that has an LLM.
    pub runtime: Option<AgentRuntime>,
    pub mode: Mode,
    pub metrics: Metrics,
}

impl Services {
    // Episodes

    pub async fn create_episode(&self, caller: &Caller, new: NewEpisode) -> ApiResult<Episode> {
        if self.mode == Mode::Host {
            return Err(ApiError::invalid_input(
                "episodes need extraction, which host mode does not run: use POST /v1/commits",
            ));
        }
        caller.require_scope(&new.scope)?;
        if new.trust > caller.key.max_trust {
            return Err(ApiError::forbidden(format!(
                "trust {} is above this key's maximum of {}",
                new.trust.as_str(),
                caller.key.max_trust.as_str()
            )));
        }
        if new.content.chars().count() > MAX_CONTENT_CHARS {
            return Err(ApiError::invalid_input(format!(
                "content exceeds {MAX_CONTENT_CHARS} characters"
            )));
        }
        if new.idempotency_key.as_deref().is_some_and(is_blank) {
            return Err(ApiError::invalid_input("idempotency_key must not be blank"));
        }
        let requested_scope = new.scope.clone();
        let episode = self.store.insert_episode(&caller.ctx, new).await?;
        // A key is unique per tenant, so a replay can return a row from a scope this caller named
        // no right to. The check above covers only the scope in the request.
        if episode.scope != requested_scope {
            return Err(ApiError::invalid_input(
                "idempotency_key was already used for a different request",
            ));
        }
        Ok(episode)
    }

    pub async fn get_episode(&self, caller: &Caller, id: Uuid) -> ApiResult<Episode> {
        let episode = self.store.get_episode(&caller.ctx, id).await?;
        caller.require_visible(&episode.scope)?;
        Ok(episode)
    }

    pub async fn purge_episode(&self, caller: &Caller, id: Uuid) -> ApiResult<PurgeReport> {
        caller.require_admin()?;
        // Episode trust covers what the purge can touch: a fact's trust is at most that of its
        // highest source, and a fact with another source only loses this link.
        let episode = self.get_episode(caller, id).await?;
        require_trust(caller, "episode", episode.trust)?;
        Ok(self.store.purge_episode(&caller.ctx, id).await?)
    }

    // Search

    pub async fn recall(&self, caller: &Caller, req: RecallRequest) -> ApiResult<RecallResponse> {
        caller.require_scopes(&req.scopes)?;
        let started = Instant::now();
        let response = self.retriever.recall(&caller.ctx, req).await?;
        self.metrics
            .recall_finished(started.elapsed().as_secs_f64());
        Ok(response)
    }

    pub async fn candidates(
        &self,
        caller: &Caller,
        req: CandidatesRequest,
    ) -> ApiResult<Vec<ScoredFact>> {
        caller.require_scopes(&req.scopes)?;
        Ok(self
            .retriever
            .candidates(
                &caller.ctx,
                &req.scopes,
                &req.statement,
                req.limit.unwrap_or(DEFAULT_CANDIDATES),
            )
            .await?)
    }

    pub async fn commit(&self, caller: &Caller, req: CommitRequest) -> ApiResult<CommitResponse> {
        caller.require_scope(&req.scope)?;
        Ok(self
            .committer
            .commit(&caller.ctx, req, caller.key.max_trust)
            .await?)
    }

    // Facts

    pub async fn get_fact(&self, caller: &Caller, id: Uuid) -> ApiResult<Fact> {
        let fact = self.store.get_fact(&caller.ctx, id).await?;
        caller.require_visible(&fact.scope)?;
        Ok(fact)
    }

    pub async fn fact_history(&self, caller: &Caller, id: Uuid) -> ApiResult<Vec<Fact>> {
        self.get_fact(caller, id).await?;
        Ok(self.store.fact_history(&caller.ctx, id).await?)
    }

    pub async fn invalidate_fact(
        &self,
        caller: &Caller,
        id: Uuid,
        reason: &str,
    ) -> ApiResult<Fact> {
        let fact = self.get_fact(caller, id).await?;
        require_trust(caller, "fact", fact.trust)?;
        // A key that cannot add here without review must not delete here by retraction either.
        if caller.key.max_trust < Trust::User && !matches!(fact.scope, Scope::Agent(_)) {
            return Err(ApiError::forbidden(
                "invalidating a fact in a user or shared scope requires user trust",
            ));
        }
        Ok(self.store.invalidate_fact(&caller.ctx, id, reason).await?)
    }

    // Graph

    pub async fn get_entity(&self, caller: &Caller, id: Uuid) -> ApiResult<Entity> {
        let entity = self.store.get_entity(&caller.ctx, id).await?;
        caller.require_visible(&entity.scope)?;
        Ok(entity)
    }

    pub async fn neighbors(
        &self,
        caller: &Caller,
        id: Uuid,
        scopes: &[Scope],
        depth: Option<i32>,
        as_of: Option<DateTime<Utc>>,
    ) -> ApiResult<GraphNeighborhood> {
        caller.require_scopes(scopes)?;
        self.get_entity(caller, id).await?;
        let filter = FactFilter {
            as_of,
            known_at: None,
        };
        Ok(self
            .store
            .neighbors(
                &caller.ctx,
                scopes,
                id,
                depth.unwrap_or(DEFAULT_DEPTH),
                &filter,
                NEIGHBORS_EDGE_LIMIT,
            )
            .await?)
    }

    // Quarantine

    pub async fn list_quarantine(
        &self,
        caller: &Caller,
        scopes: &[String],
        limit: Option<i64>,
    ) -> ApiResult<Vec<Fact>> {
        caller.require_admin()?;
        let selection = caller.select_scopes(scopes)?;
        let limit = limit.unwrap_or(DEFAULT_QUARANTINE_PAGE);
        if !(1..=MAX_QUARANTINE_PAGE).contains(&limit) {
            return Err(ApiError::invalid_input(format!(
                "limit must be in 1..={MAX_QUARANTINE_PAGE}, got {limit}"
            )));
        }
        let mut facts = self
            .pg
            .list_quarantined_matching(&caller.ctx, &selection.exact, &selection.prefixes, limit)
            .await?;
        // `select_scopes` already narrowed the query. This holds even if that logic is ever wrong.
        facts.retain(|fact| caller.may_read(&fact.scope));
        Ok(facts)
    }

    /// When the chain moved to a higher-trust head, the store holds the fact and its `InvalidInput`
    /// message tells the caller to review again (SPEC section 5).
    pub async fn approve(&self, caller: &Caller, id: Uuid) -> ApiResult<Fact> {
        caller.require_admin()?;
        let fact = self.get_fact(caller, id).await?;
        require_trust(caller, "fact", fact.trust)?;
        Ok(self.store.approve_fact(&caller.ctx, id).await?)
    }

    pub async fn reject(&self, caller: &Caller, id: Uuid) -> ApiResult<Fact> {
        caller.require_admin()?;
        let fact = self.get_fact(caller, id).await?;
        require_trust(caller, "fact", fact.trust)?;
        Ok(self.store.reject_fact(&caller.ctx, id).await?)
    }

    // Blocks

    pub async fn list_blocks(&self, caller: &Caller, agent_id: &str) -> ApiResult<Vec<Block>> {
        require_agent(caller, agent_id)?;
        Ok(self.store.list_blocks(&caller.ctx, agent_id).await?)
    }

    pub async fn get_block(
        &self,
        caller: &Caller,
        agent_id: &str,
        label: &str,
    ) -> ApiResult<Block> {
        require_agent(caller, agent_id)?;
        Ok(self.store.get_block(&caller.ctx, agent_id, label).await?)
    }

    pub async fn put_block(
        &self,
        caller: &Caller,
        agent_id: &str,
        label: &str,
        req: PutBlockRequest,
    ) -> ApiResult<Block> {
        require_agent(caller, agent_id)?;
        if caller.key.max_trust < Trust::Agent {
            return Err(ApiError::forbidden(
                "writing a block requires at least agent trust",
            ));
        }
        if req.max_chars.is_some_and(|chars| chars > MAX_BLOCK_CHARS) {
            return Err(ApiError::invalid_input(format!(
                "max_chars must be at most {MAX_BLOCK_CHARS}"
            )));
        }
        let max_chars = match (req.max_chars, req.expected_version) {
            (Some(max_chars), _) => max_chars,
            (None, None) => DEFAULT_MAX_BLOCK_CHARS,
            // An edit that does not mention the limit must not reset it to the default.
            (None, Some(_)) => {
                self.store
                    .get_block(&caller.ctx, agent_id, label)
                    .await?
                    .max_chars
            }
        };
        Ok(self
            .store
            .put_block(
                &caller.ctx,
                agent_id,
                label,
                &req.content,
                max_chars,
                req.expected_version,
            )
            .await?)
    }

    pub async fn block_versions(
        &self,
        caller: &Caller,
        agent_id: &str,
        label: &str,
    ) -> ApiResult<Vec<BlockVersion>> {
        require_agent(caller, agent_id)?;
        Ok(self
            .store
            .block_versions(&caller.ctx, agent_id, label)
            .await?)
    }

    // Sessions

    fn runtime(&self) -> ApiResult<&AgentRuntime> {
        self.runtime.as_ref().ok_or_else(|| {
            ApiError::invalid_input(
                "sessions run an agent loop, which host mode does not have: \
                 the host's own model is the agent",
            )
        })
    }

    pub async fn create_session(
        &self,
        caller: &Caller,
        agent_id: &str,
        scope: &Scope,
    ) -> ApiResult<Session> {
        self.runtime()?;
        require_agent(caller, agent_id)?;
        caller.require_scope(scope)?;
        Ok(self
            .store
            .create_session(&caller.ctx, agent_id, scope)
            .await?)
    }

    /// A session acts in two scopes, its memory scope and its agent's, and the key needs both.
    async fn visible_session(&self, caller: &Caller, id: Uuid) -> ApiResult<Session> {
        let session = self.store.get_session(&caller.ctx, id).await?;
        let agent_scope: Scope = format!("agent:{}", session.agent_id)
            .parse()
            .map_err(|_| ApiError::not_found())?;
        caller.require_visible(&agent_scope)?;
        caller.require_visible(&session.scope)?;
        Ok(session)
    }

    pub async fn session_messages(
        &self,
        caller: &Caller,
        id: Uuid,
    ) -> ApiResult<Vec<SessionMessage>> {
        self.runtime()?;
        self.visible_session(caller, id).await?;
        Ok(self.store.load_messages(&caller.ctx, id).await?)
    }

    pub async fn send_message(
        &self,
        caller: &Caller,
        id: Uuid,
        text: String,
    ) -> ApiResult<TurnResult> {
        let runtime = self.runtime()?;
        self.visible_session(caller, id).await?;
        // The turn's `memory_remember` writes episodes with trust `agent`.
        if caller.key.max_trust < Trust::Agent {
            return Err(ApiError::forbidden(
                "an agent turn writes agent trust, which is above this key's maximum",
            ));
        }
        Ok(runtime.run_turn(&caller.ctx, id, text).await?)
    }

    // Users

    pub async fn delete_user_memory(
        &self,
        caller: &Caller,
        user_id: &str,
    ) -> ApiResult<PurgeReport> {
        caller.require_admin()?;
        let scope = parse_scope("user", user_id)?;
        caller.require_scope(&scope)?;
        if let Some(highest) = self.pg.highest_trust_in_scope(&caller.ctx, &scope).await? {
            require_trust(caller, "stored memory", highest)?;
        }
        Ok(self.store.purge_scope(&caller.ctx, &scope).await?)
    }
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn require_trust(caller: &Caller, what: &str, trust: Trust) -> ApiResult<()> {
    if trust > caller.key.max_trust {
        return Err(ApiError::forbidden(format!(
            "{what} trust {} is above this key's maximum of {}",
            trust.as_str(),
            caller.key.max_trust.as_str()
        )));
    }
    Ok(())
}

fn parse_scope(kind: &str, id: &str) -> ApiResult<Scope> {
    Ok(format!("{kind}:{id}").parse::<Scope>()?)
}

fn require_agent(caller: &Caller, agent_id: &str) -> ApiResult<()> {
    caller.require_scope(&parse_scope("agent", agent_id)?)
}

/// `AgentRuntime`'s view of retrieval. `memory_recall` wants current facts, which is what
/// `candidates` returns: recall without pending episodes.
pub struct RetrieverRecall(pub Arc<Retriever>);

#[async_trait]
impl MemoryRecall for RetrieverRecall {
    async fn recall(
        &self,
        ctx: &RequestContext,
        scopes: Vec<Scope>,
        query: String,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        self.0.candidates(ctx, &scopes, &query, limit).await
    }
}
