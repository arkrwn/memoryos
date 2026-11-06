//! Host-mode commits (SPEC section 14): the host decides the facts, and MemoryOS validates and
//! gates them with the same code as the worker.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use memoryos_core::{
    Embedder, Episode, Error, Fact, FactOp, FactStatus, MemoryStore, NewEpisode, RequestContext,
    Result, Scope, Trust,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entities::embed_and_resolve;
use crate::gate::{build_op, FactDraft, Resolution, Source};
use crate::validate::{is_blank_or_unstorable, MAX_CONTENT_CHARS};

const MAX_FACTS: usize = 50;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityRef {
    pub name: String,
    pub kind: String,
}

/// Unknown fields are refused: a misspelled `supersedes` would otherwise silently become an add.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitFact {
    pub statement: String,
    pub subject: EntityRef,
    pub predicate: String,
    #[serde(default)]
    pub object: Option<EntityRef>,
    #[serde(default)]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(default)]
    pub supersedes: Option<Uuid>,
    #[serde(default)]
    pub corroborates: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitRequest {
    pub scope: Scope,
    pub source: String,
    pub content: String,
    #[serde(default)]
    pub idempotency_key: Option<String>,
    pub facts: Vec<CommitFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommittedFact {
    pub id: Uuid,
    pub status: FactStatus,
}

/// `facts` has one entry per request fact, in order. It is empty when the idempotency key was
/// already processed: the store has no way to list an episode's facts, and nothing was applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    pub episode_id: Uuid,
    pub facts: Vec<CommittedFact>,
}

/// One request fact's place in the response, known before the ops move into the store.
enum Slot {
    New(FactStatus),
    /// `require_current` established that this fact is active.
    Corroborated(Uuid),
}

pub struct HostCommitter {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn Embedder>,
}

impl HostCommitter {
    pub fn new(store: Arc<dyn MemoryStore>, embedder: Arc<dyn Embedder>) -> Self {
        Self { store, embedder }
    }

    /// `max_trust` is the calling key's limit. Commit trust is `external` for an `external` key
    /// and `agent` for every other key, because the host model also reads untrusted content.
    pub async fn commit(
        &self,
        ctx: &RequestContext,
        req: CommitRequest,
        max_trust: Trust,
    ) -> Result<CommitResponse> {
        validate_request(&req)?;
        let trust = max_trust.min(Trust::Agent);

        // The key can only be looked up by inserting, and a replay must be recognized before the
        // references are checked: the original commit may have closed the fact it superseded.
        let episode = self
            .store
            .insert_unqueued_episode(
                ctx,
                NewEpisode {
                    scope: req.scope.clone(),
                    source: req.source.clone(),
                    trust,
                    content: req.content.clone(),
                    occurred_at: None,
                    metadata: serde_json::json!({}),
                    idempotency_key: req.idempotency_key.clone(),
                },
            )
            .await?;
        require_same_request(&episode, &req, trust)?;
        if episode.processed_at.is_some() {
            return Ok(replayed(&episode));
        }

        let ops = match self.build_ops(ctx, &req, &episode).await {
            Ok(ops) => ops,
            Err(err) => {
                self.discard(ctx, episode.id).await;
                return Err(err);
            }
        };
        let slots: Vec<Slot> = ops.iter().map(slot_of).collect();
        let inserted = self.store.apply_extraction(ctx, episode.id, ops).await?;
        committed(&episode, slots, inserted)
    }

    async fn build_ops(
        &self,
        ctx: &RequestContext,
        req: &CommitRequest,
        episode: &Episode,
    ) -> Result<Vec<FactOp>> {
        // Checked before any provider call or entity write, so a rejected commit costs neither.
        let mut targets = Vec::with_capacity(req.facts.len());
        for fact in &req.facts {
            let target = match fact.supersedes.or(fact.corroborates) {
                Some(id) => Some(self.require_current(ctx, &req.scope, id).await?),
                None => None,
            };
            targets.push(target);
        }

        // A corroboration only links a source, so it needs no embedding and no entities.
        let new_facts = || req.facts.iter().filter(|fact| fact.corroborates.is_none());
        let entities: Vec<&EntityRef> = new_facts()
            .flat_map(|fact| std::iter::once(&fact.subject).chain(&fact.object))
            .collect();
        let mut batch = embed_and_resolve(
            self.store.as_ref(),
            self.embedder.as_ref(),
            ctx,
            &req.scope,
            new_facts().map(|fact| fact.statement.clone()).collect(),
            &entities,
        )
        .await?;
        let mut embeddings = std::mem::take(&mut batch.statement_embeddings).into_iter();

        let source = Source {
            scope: &episode.scope,
            trust: episode.trust,
        };
        let mut ops = Vec::with_capacity(req.facts.len());
        for (fact, target) in req.facts.iter().zip(targets) {
            let resolution = if let Some(fact_id) = fact.corroborates {
                Resolution::Corroborate { fact_id }
            } else {
                let embedding = embeddings
                    .next()
                    .ok_or_else(|| Error::Internal("statement embedding is missing".into()))?;
                let draft = FactDraft {
                    subject_id: batch.entity_id(&fact.subject)?,
                    predicate: fact.predicate.clone(),
                    object_id: fact
                        .object
                        .as_ref()
                        .map(|object| batch.entity_id(object))
                        .transpose()?,
                    statement: fact.statement.clone(),
                    embedding,
                    valid_from: fact.valid_from.unwrap_or(episode.occurred_at),
                };
                match target {
                    Some(old) => Resolution::Supersede {
                        old_fact_id: old.id,
                        old_trust: old.trust,
                        draft,
                    },
                    None => Resolution::Add(draft),
                }
            };
            ops.push(build_op(&source, resolution));
        }
        Ok(ops)
    }

    /// One message for every failure, so a caller cannot probe which ids exist elsewhere. Row-level
    /// security makes another tenant's fact `NotFound`. "Current" is the store's own filter.
    async fn require_current(&self, ctx: &RequestContext, scope: &Scope, id: Uuid) -> Result<Fact> {
        let rejected = || Error::InvalidInput(format!("fact {id} is not current in this scope"));
        let fact = match self.store.get_fact(ctx, id).await {
            Ok(fact) => fact,
            Err(Error::NotFound(_)) => return Err(rejected()),
            Err(err) => return Err(err),
        };
        if fact.scope != *scope || fact.superseded_by.is_some() {
            return Err(rejected());
        }
        let current = self
            .store
            .current_facts_for_subject(ctx, scope, fact.subject_id)
            .await?;
        if current.iter().any(|candidate| candidate.id == id) {
            Ok(fact)
        } else {
            Err(rejected())
        }
    }

    /// A commit refused before `apply_extraction` must leave no source behind. The episode holds no
    /// facts, since `apply_extraction` is what links them. If this fails, what remains is an
    /// unprocessed `agent` or `external` episode, which pending search never returns.
    async fn discard(&self, ctx: &RequestContext, episode_id: Uuid) {
        if let Err(err) = self.store.purge_episode(ctx, episode_id).await {
            tracing::warn!(%episode_id, error = %err, "could not discard a refused commit");
        }
    }
}

fn validate_request(req: &CommitRequest) -> Result<()> {
    if req.facts.len() > MAX_FACTS {
        return Err(Error::InvalidInput(format!(
            "at most {MAX_FACTS} facts per commit"
        )));
    }
    if req.content.chars().count() > MAX_CONTENT_CHARS {
        return Err(Error::InvalidInput(format!(
            "content exceeds {MAX_CONTENT_CHARS} characters"
        )));
    }
    non_empty("source", &req.source)?;
    non_empty("content", &req.content)?;
    if let Some(key) = &req.idempotency_key {
        non_empty("idempotency_key", key)?;
    }
    for fact in &req.facts {
        non_empty("statement", &fact.statement)?;
        non_empty("predicate", &fact.predicate)?;
        for entity in std::iter::once(&fact.subject).chain(&fact.object) {
            non_empty("entity name", &entity.name)?;
            non_empty("entity kind", &entity.kind)?;
        }
        if fact.supersedes.is_some() && fact.corroborates.is_some() {
            return Err(Error::InvalidInput(
                "a fact can set at most one of supersedes and corroborates".into(),
            ));
        }
    }
    Ok(())
}

fn non_empty(field: &str, value: &str) -> Result<()> {
    if is_blank_or_unstorable(value) {
        return Err(Error::InvalidInput(format!(
            "{field} must not be empty or contain NUL"
        )));
    }
    Ok(())
}

/// A key names one request. Reuse with anything else would attach these facts to another source or
/// run them under another trust. The message does not reveal what the key points at.
fn require_same_request(episode: &Episode, req: &CommitRequest, trust: Trust) -> Result<()> {
    let same = episode.scope == req.scope
        && episode.trust == trust
        && episode.source == req.source
        && episode.content == req.content;
    if same {
        Ok(())
    } else {
        Err(Error::InvalidInput(
            "idempotency_key was already used for a different request".into(),
        ))
    }
}

fn replayed(episode: &Episode) -> CommitResponse {
    CommitResponse {
        episode_id: episode.id,
        facts: Vec::new(),
    }
}

fn slot_of(op: &FactOp) -> Slot {
    match op {
        FactOp::Add(new) | FactOp::Supersede { new, .. } => Slot::New(new.status),
        FactOp::Corroborate { fact_id } => Slot::Corroborated(*fact_id),
    }
}

/// `inserted` holds only new facts, in op order. An empty list when some were expected can only
/// mean a concurrent request with the same key won the episode lock and this one was a no-op.
fn committed(episode: &Episode, slots: Vec<Slot>, inserted: Vec<Uuid>) -> Result<CommitResponse> {
    let expected = slots.iter().filter(|s| matches!(s, Slot::New(_))).count();
    if inserted.is_empty() && expected > 0 {
        return Ok(replayed(episode));
    }
    if inserted.len() != expected {
        return Err(Error::Internal(format!(
            "store inserted {} facts, expected {expected}",
            inserted.len()
        )));
    }
    let mut inserted = inserted.into_iter();
    let mut facts = Vec::with_capacity(slots.len());
    for slot in slots {
        let fact = match slot {
            Slot::New(status) => CommittedFact {
                id: inserted
                    .next()
                    .ok_or_else(|| Error::Internal("inserted id is missing".into()))?,
                status,
            },
            Slot::Corroborated(id) => CommittedFact {
                id,
                status: FactStatus::Active,
            },
        };
        facts.push(fact);
    }
    Ok(CommitResponse {
        episode_id: episode.id,
        facts,
    })
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use serde_json::json;

    use super::*;

    fn fact() -> CommitFact {
        CommitFact {
            statement: "s".into(),
            subject: EntityRef {
                name: "n".into(),
                kind: "k".into(),
            },
            predicate: "p".into(),
            object: None,
            valid_from: None,
            supersedes: None,
            corroborates: None,
        }
    }

    fn request(facts: Vec<CommitFact>) -> CommitRequest {
        CommitRequest {
            scope: Scope::user("u"),
            source: "chat".into(),
            content: "c".into(),
            idempotency_key: None,
            facts,
        }
    }

    fn episode() -> Episode {
        Episode {
            id: Uuid::now_v7(),
            scope: Scope::user("u"),
            source: "chat".into(),
            trust: Trust::Agent,
            content: "c".into(),
            metadata: json!({}),
            occurred_at: Utc::now(),
            created_at: Utc::now(),
            processed_at: None,
        }
    }

    #[test]
    fn accepts_the_limit_and_refuses_past_it() {
        assert!(validate_request(&request(vec![fact(); 50])).is_ok());
        assert!(validate_request(&request(vec![fact(); 51])).is_err());
        assert!(validate_request(&request(Vec::new())).is_ok());
    }

    #[test]
    fn refuses_blank_fields() {
        type Change = fn(&mut CommitRequest);
        let changes: [Change; 8] = [
            |r| r.source = " ".into(),
            |r| r.content = "".into(),
            |r| r.idempotency_key = Some("".into()),
            |r| r.facts[0].statement = "\n".into(),
            |r| r.facts[0].predicate = "".into(),
            |r| r.facts[0].subject.name = "".into(),
            |r| r.facts[0].subject.kind = " ".into(),
            |r| {
                r.facts[0].object = Some(EntityRef {
                    name: "".into(),
                    kind: "k".into(),
                })
            },
        ];
        for (index, change) in changes.iter().enumerate() {
            let mut req = request(vec![fact()]);
            change(&mut req);
            assert!(
                matches!(validate_request(&req), Err(Error::InvalidInput(_))),
                "change {index}"
            );
        }
    }

    #[test]
    fn refuses_content_over_the_limit_counted_in_characters() {
        let mut req = request(Vec::new());
        req.content = "é".repeat(MAX_CONTENT_CHARS);
        assert!(validate_request(&req).is_ok());
        req.content.push('é');
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn refuses_both_supersedes_and_corroborates() {
        let mut both = fact();
        both.supersedes = Some(Uuid::now_v7());
        both.corroborates = Some(Uuid::now_v7());
        assert!(validate_request(&request(vec![both])).is_err());
    }

    #[test]
    fn serde_names_match_openapi() {
        let id = Uuid::now_v7();
        let parsed: CommitRequest = serde_json::from_value(json!({
            "scope": "user:u",
            "source": "chat",
            "content": "c",
            "idempotency_key": "k",
            "facts": [{
                "statement": "s",
                "subject": {"name": "n", "kind": "k"},
                "predicate": "p",
                "object": {"name": "o", "kind": "k"},
                "valid_from": "2026-01-01T00:00:00Z",
                "supersedes": id,
                "corroborates": null,
            }],
        }))
        .expect("full request");
        assert_eq!(parsed.scope, Scope::user("u"));
        assert_eq!(parsed.facts[0].supersedes, Some(id));

        let minimal: CommitRequest = serde_json::from_value(json!({
            "scope": "agent:a",
            "source": "s",
            "content": "c",
            "facts": [{"statement": "s", "subject": {"name": "n", "kind": "k"}, "predicate": "p"}],
        }))
        .expect("minimal request");
        assert!(minimal.idempotency_key.is_none());
        assert!(minimal.facts[0].object.is_none());

        let response = CommitResponse {
            episode_id: id,
            facts: vec![CommittedFact {
                id,
                status: FactStatus::Quarantined,
            }],
        };
        assert_eq!(
            serde_json::to_value(response).expect("serialize"),
            json!({"episode_id": id, "facts": [{"id": id, "status": "quarantined"}]})
        );
    }

    #[test]
    fn a_trust_field_in_a_request_is_refused() {
        let attempt = json!({
            "scope": "user:u",
            "source": "s",
            "content": "c",
            "trust": "system",
            "facts": [],
        });
        assert!(serde_json::from_value::<CommitRequest>(attempt).is_err());
    }

    #[test]
    fn trust_is_clamped_to_agent() {
        for (max_trust, expected) in [
            (Trust::External, Trust::External),
            (Trust::Agent, Trust::Agent),
            (Trust::User, Trust::Agent),
            (Trust::System, Trust::Agent),
        ] {
            assert_eq!(max_trust.min(Trust::Agent), expected);
        }
    }

    #[test]
    fn reused_key_with_a_different_request_is_refused() {
        let ep = episode();
        let same = request(Vec::new());
        assert!(require_same_request(&ep, &same, Trust::Agent).is_ok());
        assert!(require_same_request(&ep, &same, Trust::External).is_err());
        let mut other_scope = request(Vec::new());
        other_scope.scope = Scope::user("v");
        assert!(require_same_request(&ep, &other_scope, Trust::Agent).is_err());
        let mut other_content = request(Vec::new());
        other_content.content = "d".into();
        assert!(require_same_request(&ep, &other_content, Trust::Agent).is_err());
    }

    #[test]
    fn response_aligns_corroborations_with_inserted_ids() {
        let ep = episode();
        let (first, second, linked) = (Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7());
        let slots = vec![
            Slot::New(FactStatus::Active),
            Slot::Corroborated(linked),
            Slot::New(FactStatus::Quarantined),
        ];
        let response = committed(&ep, slots, vec![first, second]).expect("aligned");
        let got: Vec<_> = response.facts.iter().map(|f| (f.id, f.status)).collect();
        assert_eq!(
            got,
            [
                (first, FactStatus::Active),
                (linked, FactStatus::Active),
                (second, FactStatus::Quarantined),
            ]
        );
    }

    #[test]
    fn losing_a_concurrent_replay_returns_no_facts() {
        let ep = episode();
        let response =
            committed(&ep, vec![Slot::New(FactStatus::Active)], Vec::new()).expect("replay");
        assert_eq!(response.episode_id, ep.id);
        assert!(response.facts.is_empty());
    }
}
