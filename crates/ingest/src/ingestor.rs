//! Worker-mode processing of one episode (SPEC section 3).

use std::sync::Arc;

use memoryos_core::{
    Embedder, Episode, Error, Fact, FactOp, Llm, MemoryStore, RequestContext, Result,
};
use uuid::Uuid;

use crate::entities::{add_earned_aliases, embed_and_resolve, ResolvedBatch};
use crate::extract::{
    extraction_request, parse_decision, parse_extraction, reconciliation_request, Decision,
    ExtractedStatement,
};
use crate::gate::{build_op, FactDraft, Resolution, Source};
use crate::host::EntityRef;

/// Most recent current facts offered to reconciliation for one subject.
const MAX_RECONCILE_CANDIDATES: usize = 50;

pub struct Ingestor {
    store: Arc<dyn MemoryStore>,
    llm: Arc<dyn Llm>,
    embedder: Arc<dyn Embedder>,
}

impl Ingestor {
    pub fn new(
        store: Arc<dyn MemoryStore>,
        llm: Arc<dyn Llm>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            store,
            llm,
            embedder,
        }
    }

    /// Extracts, resolves, reconciles, gates and applies. No fact is written until the final atomic
    /// `apply_extraction`, so any failure can simply be retried. Entities resolved before a failure
    /// stay and are found again on the retry.
    pub async fn process_episode(&self, ctx: &RequestContext, episode_id: Uuid) -> Result<()> {
        let episode = self.store.get_episode(ctx, episode_id).await?;
        // A retry after a crash that followed the commit must not pay for the LLM again.
        if episode.processed_at.is_some() {
            return Ok(());
        }

        let request = extraction_request(&episode.content, episode.occurred_at);
        let statements = parse_extraction(&self.llm.complete(request).await?)?;

        let texts = statements.iter().map(|s| s.statement.clone()).collect();
        let entities: Vec<&EntityRef> = statements
            .iter()
            .flat_map(|s| std::iter::once(&s.subject).chain(&s.object))
            .collect();
        let mut batch = embed_and_resolve(
            self.store.as_ref(),
            self.embedder.as_ref(),
            ctx,
            &episode.scope,
            texts,
            &entities,
        )
        .await?;
        let merges = std::mem::take(&mut batch.merges);

        let ops = self.reconcile_all(ctx, &episode, statements, batch).await?;
        let ops_applied = ops.len();
        let inserted = self.store.apply_extraction(ctx, episode_id, ops).await?;
        add_earned_aliases(self.store.as_ref(), ctx, &episode, merges, &inserted).await;
        tracing::info!(
            %episode_id,
            ops_applied,
            facts_inserted = inserted.len(),
            "episode processed"
        );
        Ok(())
    }

    async fn reconcile_all(
        &self,
        ctx: &RequestContext,
        episode: &Episode,
        statements: Vec<ExtractedStatement>,
        mut batch: ResolvedBatch,
    ) -> Result<Vec<FactOp>> {
        let source = Source {
            scope: &episode.scope,
            trust: episode.trust,
        };
        let embeddings = std::mem::take(&mut batch.statement_embeddings);
        let mut ops = Vec::with_capacity(statements.len());
        for (statement, embedding) in statements.into_iter().zip(embeddings) {
            let subject_id = batch.entity_id(&statement.subject)?;
            let object_id = statement
                .object
                .as_ref()
                .map(|object| batch.entity_id(object))
                .transpose()?;

            let candidates = self.candidates(ctx, episode, subject_id).await?;
            let decision = self.decide(&statement.statement, &candidates).await?;

            let draft = FactDraft {
                subject_id,
                predicate: statement.predicate,
                object_id,
                statement: statement.statement,
                embedding,
                valid_from: statement.valid_from.unwrap_or(episode.occurred_at),
            };
            let resolution = match decision {
                Decision::Ignore => continue,
                Decision::Add => Resolution::Add(draft),
                Decision::Corroborate(fact_id) => Resolution::Corroborate { fact_id },
                Decision::Supersede(old_fact_id) => {
                    let old = candidates
                        .iter()
                        .find(|fact| fact.id == old_fact_id)
                        .ok_or_else(|| Error::Internal("decision outside candidates".into()))?;
                    Resolution::Supersede {
                        old_fact_id,
                        old_trust: old.trust,
                        draft,
                    }
                }
            };
            ops.push(build_op(&source, resolution));
        }
        Ok(ops)
    }

    async fn candidates(
        &self,
        ctx: &RequestContext,
        episode: &Episode,
        subject_id: Uuid,
    ) -> Result<Vec<Fact>> {
        let mut facts = self
            .store
            .current_facts_for_subject(ctx, &episode.scope, subject_id)
            .await?;
        // Ordered oldest first by the store. A hub entity must not produce an unbounded prompt.
        if facts.len() > MAX_RECONCILE_CANDIDATES {
            facts.drain(..facts.len() - MAX_RECONCILE_CANDIDATES);
        }
        Ok(facts)
    }

    /// With nothing to compare against the statement is new, and no LLM call is needed.
    async fn decide(&self, statement: &str, candidates: &[Fact]) -> Result<Decision> {
        if candidates.is_empty() {
            return Ok(Decision::Add);
        }
        let response = self
            .llm
            .complete(reconciliation_request(statement, candidates))
            .await?;
        parse_decision(&response, candidates)
    }
}
