//! Hybrid retrieval over facts: vector, text and graph lists merged by reciprocal rank fusion.
//! See docs/SPEC.md sections 2, 7 and 14.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::future::join_all;
use memoryos_core::{
    Embedder, Error, Fact, FactFilter, MemoryStore, RequestContext, Result, Scope, ScoredFact,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod fusion;

pub const DEFAULT_LIMIT: i64 = 10;
pub const MAX_LIMIT: i64 = 100;

const MAX_SCOPES: usize = 50;
const CANDIDATES_PER_LIST: i64 = 50;
const GRAPH_SEED_FACTS: usize = 5;
const GRAPH_DEPTH: i32 = 1;
/// Bounds how many edges the database reads for one subject, however many it has.
const GRAPH_EDGES_PER_SUBJECT: i64 = 50;
/// Bounds how many edges reach fusion, however many a subject has.
const MAX_GRAPH_CANDIDATES: usize = 50;
/// A slow embedder must not hold up recall, since text search can answer without it.
const DEFAULT_EMBED_TIMEOUT: Duration = Duration::from_secs(2);

/// What a ranking is for, which sets its time filter and how the fused list is finished.
#[derive(Debug, Clone, Copy)]
enum Purpose<'a> {
    /// Facts at the requested time. Source count nudges the order and near copies give way to
    /// distinct facts.
    Recall(&'a FactFilter),
    /// Current facts related to a statement. Plain fusion, because finding near copies of the
    /// statement is the point.
    Candidates,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallRequest {
    pub scopes: Vec<Scope>,
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: i64,
    #[serde(default)]
    pub as_of: Option<DateTime<Utc>>,
    #[serde(default)]
    pub known_at: Option<DateTime<Utc>>,
}

fn default_limit() -> i64 {
    DEFAULT_LIMIT
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallResponse {
    pub facts: Vec<ScoredFact>,
}

pub struct Retriever {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn Embedder>,
    embed_timeout: Duration,
}

impl Retriever {
    pub fn new(store: Arc<dyn MemoryStore>, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            store,
            embedder,
            embed_timeout: DEFAULT_EMBED_TIMEOUT,
        }
    }

    /// How long to wait for the query embedding before using text search only. Defaults to 2 seconds.
    pub fn with_embed_timeout(mut self, timeout: Duration) -> Self {
        self.embed_timeout = timeout;
        self
    }

    /// Ranked facts at the time the request asks about.
    pub async fn recall(&self, ctx: &RequestContext, req: RecallRequest) -> Result<RecallResponse> {
        let limit = validate(&req.scopes, &req.query, req.limit)?;
        let filter = FactFilter {
            as_of: req.as_of,
            known_at: req.known_at,
        };
        let embedding = self.embed_query(&req.query).await;
        let facts = self
            .ranked_facts(
                ctx,
                &req.scopes,
                &req.query,
                embedding.as_deref(),
                limit,
                Purpose::Recall(&filter),
            )
            .await?;
        Ok(RecallResponse { facts })
    }

    /// Related current facts for a statement the host may commit.
    pub async fn candidates(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        statement: &str,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        let limit = validate(scopes, statement, limit)?;
        let embedding = self.embed_query(statement).await;
        self.ranked_facts(
            ctx,
            scopes,
            statement,
            embedding.as_deref(),
            limit,
            Purpose::Candidates,
        )
        .await
    }

    /// `candidates` for a statement the caller has already embedded, so a commit that embeds its
    /// statements anyway does not pay for a second provider call.
    pub async fn candidates_with_embedding(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        statement: &str,
        embedding: &[f32],
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        let limit = validate(scopes, statement, limit)?;
        self.ranked_facts(
            ctx,
            scopes,
            statement,
            Some(embedding),
            limit,
            Purpose::Candidates,
        )
        .await
    }

    async fn ranked_facts(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        embedding: Option<&[f32]>,
        limit: usize,
        purpose: Purpose<'_>,
    ) -> Result<Vec<ScoredFact>> {
        let current = FactFilter::default();
        let filter = match purpose {
            Purpose::Recall(filter) => filter,
            Purpose::Candidates => &current,
        };
        let (vector, text) = tokio::join!(
            self.vector_list(ctx, scopes, embedding, filter),
            self.store
                .search_facts_text(ctx, scopes, query, filter, CANDIDATES_PER_LIST),
        );
        // Text search is the fallback for vector search, so nothing is left if it fails too.
        let text = facts_only(text?);

        // Fused twice on purpose: the first pass only picks which subjects to expand.
        let subjects = fusion::top_subjects(&fusion::fuse(&[&vector, &text]), GRAPH_SEED_FACTS);
        let graph = self.graph_list(ctx, scopes, &subjects, filter).await;

        let fused = fusion::fuse(&[&vector, &text, &graph]);
        let ranked = match purpose {
            Purpose::Recall(_) => fusion::without_near_copies(fusion::boost_by_sources(fused)),
            Purpose::Candidates => fused,
        };
        Ok(ranked
            .into_iter()
            .take(limit)
            .map(|(fact, score)| ScoredFact {
                fact: fact.clone(),
                score,
            })
            .collect())
    }

    async fn embed_query(&self, query: &str) -> Option<Vec<f32>> {
        let input = [query.to_owned()];
        let embedded = tokio::time::timeout(self.embed_timeout, self.embedder.embed(&input)).await;
        match embedded {
            Ok(Ok(vectors)) => {
                let first = vectors.into_iter().next();
                if first.is_none() {
                    tracing::warn!("embedder returned no vector, using text search only");
                }
                first
            }
            Ok(Err(error)) => {
                let kind = fusion::error_kind(&error);
                tracing::warn!(kind, "query embedding failed, using text search only");
                None
            }
            Err(_) => {
                let timeout_ms = self.embed_timeout.as_millis() as u64;
                tracing::warn!(
                    timeout_ms,
                    "query embedding timed out, using text search only"
                );
                None
            }
        }
    }

    async fn vector_list(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        embedding: Option<&[f32]>,
        filter: &FactFilter,
    ) -> Vec<Fact> {
        let Some(embedding) = embedding else {
            return Vec::new();
        };
        let found = self
            .store
            .search_facts_vector(ctx, scopes, embedding, filter, CANDIDATES_PER_LIST)
            .await;
        match found {
            Ok(scored) => facts_only(scored),
            Err(error) => {
                let kind = fusion::error_kind(&error);
                tracing::warn!(kind, "vector search failed, using text search only");
                Vec::new()
            }
        }
    }

    async fn graph_list(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        subjects: &[Uuid],
        filter: &FactFilter,
    ) -> Vec<Fact> {
        let expansions = subjects.iter().map(|&subject| async move {
            let neighborhood = self
                .store
                .neighbors(
                    ctx,
                    scopes,
                    subject,
                    GRAPH_DEPTH,
                    filter,
                    GRAPH_EDGES_PER_SUBJECT,
                )
                .await;
            (subject, neighborhood)
        });
        let results = join_all(expansions).await;
        fusion::interleave(fusion::usable_edges(results), MAX_GRAPH_CANDIDATES)
    }
}

fn facts_only(scored: Vec<ScoredFact>) -> Vec<Fact> {
    scored.into_iter().map(|scored| scored.fact).collect()
}

/// Checks the request and returns `limit` as a count. Messages never quote the query.
fn validate(scopes: &[Scope], query: &str, limit: i64) -> Result<usize> {
    if scopes.is_empty() || scopes.len() > MAX_SCOPES {
        return Err(Error::InvalidInput(format!(
            "scopes must have 1 to {MAX_SCOPES} entries, got {}",
            scopes.len()
        )));
    }
    if query.trim().is_empty() {
        return Err(Error::InvalidInput("query must not be empty".into()));
    }
    if !(1..=MAX_LIMIT).contains(&limit) {
        return Err(Error::InvalidInput(format!(
            "limit must be in 1..={MAX_LIMIT}, got {limit}"
        )));
    }
    Ok(limit as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scopes(count: usize) -> Vec<Scope> {
        (0..count).map(|i| Scope::user(format!("u{i}"))).collect()
    }

    fn is_invalid(result: Result<usize>) -> bool {
        matches!(result, Err(Error::InvalidInput(_)))
    }

    fn parse(text: &str) -> RecallRequest {
        serde_json::from_str(text).expect("valid request")
    }

    #[test]
    fn unknown_fields_are_refused() {
        let attempt = r#"{"scopes": ["user:u"], "query": "q", "limitt": 5}"#;
        assert!(serde_json::from_str::<RecallRequest>(attempt).is_err());
    }

    #[test]
    fn scope_count_must_be_1_to_50() {
        assert!(is_invalid(validate(&scopes(0), "q", 10)));
        assert!(is_invalid(validate(&scopes(51), "q", 10)));
        assert_eq!(validate(&scopes(1), "q", 10).ok(), Some(10));
        assert_eq!(validate(&scopes(50), "q", 10).ok(), Some(10));
    }

    #[test]
    fn query_must_have_content() {
        for blank in ["", "   ", " \n\t "] {
            assert!(is_invalid(validate(&scopes(1), blank, 10)));
        }
    }

    #[test]
    fn limit_must_be_1_to_100() {
        for bad in [i64::MIN, -1, 0, 101, i64::MAX] {
            assert!(is_invalid(validate(&scopes(1), "q", bad)), "{bad}");
        }
        assert_eq!(validate(&scopes(1), "q", 1).ok(), Some(1));
        assert_eq!(validate(&scopes(1), "q", 100).ok(), Some(100));
    }

    #[test]
    fn request_defaults_limit_and_times() {
        let request = parse(r#"{"scopes":["user:a"],"query":"q"}"#);
        assert_eq!(request.limit, DEFAULT_LIMIT);
        assert!(request.as_of.is_none());
        assert!(request.known_at.is_none());
        assert_eq!(request.scopes, [Scope::user("a")]);

        let request = parse(
            r#"{"scopes":[],"query":"q","limit":3,"as_of":"2024-01-02T03:04:05Z","known_at":null}"#,
        );
        assert_eq!(request.limit, 3);
        assert!(request.as_of.is_some());
        assert!(request.known_at.is_none());
    }

    #[test]
    fn request_rejects_a_bad_scope() {
        let parsed = serde_json::from_str::<RecallRequest>(r#"{"scopes":["team:a"],"query":"q"}"#);
        assert!(parsed.is_err());
    }

    #[test]
    fn response_serializes_the_openapi_fields() {
        let response = RecallResponse { facts: vec![] };
        let json = serde_json::to_value(&response).expect("serializes");
        assert_eq!(json, serde_json::json!({"facts": []}));
    }
}
