//! Commits (SPEC section 3): the host decides the facts, and MemoryOS validates and gates them.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use futures::future::try_join_all;
use memoryos_core::{
    normalize_name, Embedder, Episode, Error, Fact, FactOp, FactStatus, MemoryStore, NewEpisode,
    RequestContext, Result, Scope, ScoredFact, Trust,
};
use memoryos_retrieve::Retriever;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entities::{add_earned_aliases, embed_batch, Merge};
use crate::gate::{build_op, status_for, FactDraft, Resolution, Source};
use crate::redact::redact;
use crate::validate::{is_blank_or_unstorable, MAX_CONTENT_CHARS};

const MAX_FACTS: usize = 50;
/// Names and keys are in unique indexes, whose rows PostgreSQL limits to about 2700 bytes. At four
/// bytes a character, a name at this limit and a kind at the next one still fit together.
const MAX_INDEXED_CHARS: usize = 512;
const MAX_KIND_CHARS: usize = 64;
const MAX_FUTURE_VALID_FROM_DAYS: i64 = 1;
/// Current facts compared with each statement that does not name `corroborates`.
const MATCH_CANDIDATES: i64 = 10;
/// At most this many `related` facts per created fact: enough to spot an older value, little to read.
const MAX_RELATED: usize = 3;

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
    /// When the fact stops being true with nothing replacing it. After `valid_from`.
    #[serde(default)]
    pub expires_at: Option<DateTime<Utc>>,
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

/// A current fact in the commit's scope that resembles a created fact without repeating it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelatedFact {
    pub id: Uuid,
    pub statement: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommittedFact {
    pub id: Uuid,
    pub status: FactStatus,
    /// No fact was created: `id` is an existing fact that gained this source, because the request
    /// named it in `corroborates` or because it holds the same statement after normalization.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub corroborated: bool,
    /// For a created fact, up to three current facts of the scope that were candidates but not
    /// repeats, best first. Empty on a replay.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related: Vec<RelatedFact>,
    /// Set with `corroborated` when the request fact named this id in `supersedes` and the
    /// statement was already a current fact: that fact, `id`, gained this source and became the
    /// successor of the superseded fact, instead of a duplicate being created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded: Option<Uuid>,
}

/// `facts` has one entry per request fact, in order, read back from the store. Each status is the
/// fact's status at the time of the response, so the store's last trust check is reflected, and a
/// replayed idempotency key returns the original ids with whatever has happened to them since.
///
/// `redacted` is true when a secret in a statement, an entity name or `content` was masked before
/// storage, so the caller knows the stored text differs from what it sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    pub episode_id: Uuid,
    pub facts: Vec<CommittedFact>,
    #[serde(default)]
    pub redacted: bool,
}

pub struct HostCommitter {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn Embedder>,
    /// The search behind `recall_candidates`, used to match statements before they are saved.
    retriever: Retriever,
}

/// What `build_ops` decided for one request.
struct Built {
    ops: Vec<FactOp>,
    merges: Vec<Merge>,
    /// One list per request fact, in request order.
    related: Vec<Vec<RelatedFact>>,
}

/// The result of comparing one statement with the current facts of its scope.
#[derive(Debug, Default, PartialEq)]
struct Matched {
    /// A current fact with the same normalized statement, which the statement then corroborates.
    existing: Option<Uuid>,
    /// Empty when `existing` is set, since nothing is created.
    related: Vec<RelatedFact>,
}

impl HostCommitter {
    pub fn new(store: Arc<dyn MemoryStore>, embedder: Arc<dyn Embedder>) -> Self {
        let retriever = Retriever::new(store.clone(), embedder.clone());
        Self {
            store,
            embedder,
            retriever,
        }
    }

    /// `max_trust` is the calling key's limit. Commit trust is `external` for an `external` key
    /// and `agent` for every other key, because the host model also reads untrusted content.
    pub async fn commit(
        &self,
        ctx: &RequestContext,
        req: CommitRequest,
        max_trust: Trust,
    ) -> Result<CommitResponse> {
        // Before anything is stored or compared, so a replay compares the masked text it stored.
        let mut req = req;
        let redacted = redact_request(&mut req);
        validate_request(&req)?;
        let trust = max_trust.min(Trust::Agent);

        // The key can only be looked up by inserting, and a replay must be recognized before the
        // references are checked: the original commit may have closed the fact it superseded.
        let (episode, created) = self
            .store
            .claim_episode(
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
            return self
                .response(ctx, &episode, &req, Vec::new(), redacted)
                .await;
        }

        let built = match self.build_ops(ctx, &req, &episode).await {
            Ok(built) => built,
            Err(err) => {
                // A replay does not own the episode: the request that created it may still be
                // running, or may already have been acknowledged.
                if created {
                    self.discard(ctx, episode.id).await;
                }
                return Err(err);
            }
        };
        // A no-op if a concurrent request with the same key won the episode lock. The response
        // below then reports that request's facts, the same as any other replay.
        let inserted = self.store.apply_commit(ctx, episode.id, built.ops).await?;
        add_earned_aliases(self.store.as_ref(), ctx, &episode, built.merges, &inserted).await;
        self.response(ctx, &episode, &req, built.related, redacted)
            .await
    }

    async fn response(
        &self,
        ctx: &RequestContext,
        episode: &Episode,
        req: &CommitRequest,
        related: Vec<Vec<RelatedFact>>,
        redacted: bool,
    ) -> Result<CommitResponse> {
        // The episode was claimed before it was processed, so a fresh commit reads the time again.
        let processed_at = match episode.processed_at {
            Some(at) => Some(at),
            None => self.store.get_episode(ctx, episode.id).await?.processed_at,
        };
        let facts = self.store.facts_for_episode(ctx, episode.id).await?;
        Ok(CommitResponse {
            episode_id: episode.id,
            facts: in_request_order(req, facts, processed_at, related),
            redacted,
        })
    }

    async fn build_ops(
        &self,
        ctx: &RequestContext,
        req: &CommitRequest,
        episode: &Episode,
    ) -> Result<Built> {
        // Checked before any provider call or entity write, so a rejected commit costs neither.
        let mut targets = Vec::with_capacity(req.facts.len());
        for fact in &req.facts {
            let target = match fact.supersedes.or(fact.corroborates) {
                Some(id) => Some(self.require_current(ctx, &req.scope, id).await?),
                None => None,
            };
            targets.push(target);
        }

        // A named corroboration only links a source, so it needs no embedding. The others are
        // embedded with every entity they mention in one provider call, before matching decides
        // which of them are repeats.
        let embedded: Vec<&CommitFact> = req
            .facts
            .iter()
            .filter(|fact| fact.corroborates.is_none())
            .collect();
        let entities: Vec<&EntityRef> = embedded.iter().flat_map(|fact| mentions(fact)).collect();
        let mut batch = embed_batch(
            self.embedder.as_ref(),
            embedded.iter().map(|fact| fact.statement.clone()).collect(),
            &entities,
        )
        .await?;
        let mut vectors = std::mem::take(&mut batch.statement_embeddings).into_iter();
        let embeddings: Vec<Option<Vec<f32>>> = req
            .facts
            .iter()
            .map(|fact| match fact.corroborates {
                Some(_) => None,
                None => vectors.next(),
            })
            .collect();

        // A fact this request supersedes is closing, so a repeat of it must not be linked to it.
        let superseded: HashSet<Uuid> = req.facts.iter().filter_map(|f| f.supersedes).collect();
        // Concurrent, and bounded by the store's connection pool.
        let searches: Vec<_> = req
            .facts
            .iter()
            .zip(&embeddings)
            .map(|(fact, embedding)| {
                self.match_existing(ctx, &req.scope, fact, embedding.as_deref(), &superseded)
            })
            .collect();
        let mut matches: Vec<Matched> = try_join_all(searches).await?;
        // A supersede reuses a matching fact only if its new fact would be active. A quarantined
        // one must be created, since it is what records the close for approval to perform.
        for ((fact, target), matched) in req.facts.iter().zip(&targets).zip(&mut matches) {
            if let (Some(_), Some(old)) = (fact.supersedes, target) {
                if status_for(&episode.scope, episode.trust, Some(old.trust)) != FactStatus::Active
                {
                    matched.existing = None;
                }
            }
        }

        // Only a fact that is created needs its entities resolved, which may create them.
        let used: Vec<&EntityRef> = req
            .facts
            .iter()
            .zip(&matches)
            .filter(|(fact, matched)| fact.corroborates.is_none() && matched.existing.is_none())
            .flat_map(|(fact, _)| mentions(fact))
            .collect();
        let resolved = batch
            .resolve(self.store.as_ref(), ctx, &req.scope, &used)
            .await?;

        let source = Source {
            scope: &episode.scope,
            trust: episode.trust,
        };
        let mut ops = Vec::with_capacity(req.facts.len());
        let mut related = Vec::with_capacity(req.facts.len());
        let decided = req.facts.iter().zip(targets).zip(embeddings).zip(matches);
        for (((fact, target), embedding), matched) in decided {
            let resolution = if let (Some(fact_id), Some(old)) = (matched.existing, &target) {
                // Only a supersede has both a target and a match.
                Resolution::SupersedeWithExisting {
                    old_fact_id: old.id,
                    fact_id,
                }
            } else if let Some(fact_id) = fact.corroborates.or(matched.existing) {
                Resolution::Corroborate { fact_id }
            } else {
                let embedding = embedding
                    .ok_or_else(|| Error::Internal("statement embedding is missing".into()))?;
                let draft = FactDraft {
                    subject_id: resolved.entity_id(&fact.subject)?,
                    predicate: fact.predicate.clone(),
                    object_id: fact
                        .object
                        .as_ref()
                        .map(|object| resolved.entity_id(object))
                        .transpose()?,
                    statement: fact.statement.clone(),
                    embedding,
                    valid_from: fact.valid_from.unwrap_or(episode.occurred_at),
                    expires_at: fact.expires_at,
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
            related.push(matched.related);
        }
        Ok(Built {
            ops,
            merges: resolved.merges,
            related,
        })
    }

    /// Compares a statement that does not name `corroborates` with the current facts of the
    /// commit's scope, through the same search as `recall_candidates`. The caller is already
    /// allowed that scope, and the search reads no other scope and no other tenant. A supersede
    /// only looks for a repeat, which then takes the place of its new fact, and lists no `related`.
    async fn match_existing(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        fact: &CommitFact,
        embedding: Option<&[f32]>,
        superseded: &HashSet<Uuid>,
    ) -> Result<Matched> {
        let (None, Some(embedding)) = (fact.corroborates, embedding) else {
            return Ok(Matched::default());
        };
        let found = self
            .retriever
            .candidates_with_embedding(
                ctx,
                std::slice::from_ref(scope),
                &fact.statement,
                embedding,
                MATCH_CANDIDATES,
            )
            .await?;
        let mut matched = pick(scope, &fact.statement, found, superseded);
        if fact.supersedes.is_some() {
            matched.related.clear();
        }
        Ok(matched)
    }

    /// One message for every failure, so a caller cannot probe which ids exist elsewhere. Row-level
    /// security hides another tenant's fact. "Current" is the store's own filter.
    async fn require_current(&self, ctx: &RequestContext, scope: &Scope, id: Uuid) -> Result<Fact> {
        match self.store.current_fact(ctx, scope, id).await? {
            Some(fact) if fact.superseded_by.is_none() => Ok(fact),
            _ => Err(Error::InvalidInput(format!(
                "fact {id} is not current in this scope"
            ))),
        }
    }

    /// A commit refused before `apply_commit` must leave no source behind. The episode holds no
    /// facts, since `apply_commit` is what links them, and if a replay has processed it since, the
    /// store leaves it alone. If this fails, what remains is an unprocessed episode that nothing reads.
    async fn discard(&self, ctx: &RequestContext, episode_id: Uuid) {
        if let Err(err) = self
            .store
            .discard_unprocessed_episode(ctx, episode_id)
            .await
        {
            tracing::warn!(%episode_id, error = %err, "could not discard a refused commit");
        }
    }
}

/// Masks secrets in everything a commit stores as text: `content`, statements and entity names.
/// Returns whether anything was masked.
fn redact_request(req: &mut CommitRequest) -> bool {
    let mut changed = redact(&mut req.content);
    for fact in &mut req.facts {
        changed |= redact(&mut fact.statement);
        changed |= redact(&mut fact.subject.name);
        if let Some(object) = &mut fact.object {
            changed |= redact(&mut object.name);
        }
    }
    changed
}

fn mentions(fact: &CommitFact) -> impl Iterator<Item = &EntityRef> {
    std::iter::once(&fact.subject).chain(&fact.object)
}

/// Statement identity for automatic corroboration: case-insensitive, whitespace runs collapsed,
/// trailing periods ignored.
fn normalize_statement(statement: &str) -> String {
    normalize_name(statement.trim_end().trim_end_matches('.'))
}

/// Splits ranked candidates into the first repeat of `statement` and up to `MAX_RELATED` others.
fn pick(
    scope: &Scope,
    statement: &str,
    found: Vec<ScoredFact>,
    superseded: &HashSet<Uuid>,
) -> Matched {
    let wanted = normalize_statement(statement);
    let mut matched = Matched::default();
    let now = Utc::now();
    for ScoredFact { fact, score } in found {
        // The search reads only current, active facts of `scope`. Checked again because a wrong
        // match would link this source to a fact the caller must not reach.
        let eligible = fact.scope == *scope
            && fact.status == FactStatus::Active
            && fact.superseded_by.is_none()
            && fact.invalidated_at.is_none()
            && !fact.expired_at(now)
            && !superseded.contains(&fact.id);
        if !eligible {
            continue;
        }
        if normalize_statement(&fact.statement) == wanted {
            matched.existing.get_or_insert(fact.id);
        } else if matched.related.len() < MAX_RELATED {
            matched.related.push(RelatedFact {
                id: fact.id,
                statement: fact.statement,
                score,
            });
        }
    }
    if matched.existing.is_some() {
        matched.related.clear();
    }
    matched
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
        at_most("idempotency_key", key, MAX_INDEXED_CHARS)?;
    }
    let now = Utc::now();
    let latest_valid_from = now + Duration::days(MAX_FUTURE_VALID_FROM_DAYS);
    for fact in &req.facts {
        if fact.valid_from.is_some_and(|from| from > latest_valid_from) {
            return Err(Error::InvalidInput(format!(
                "valid_from is more than {MAX_FUTURE_VALID_FROM_DAYS} day in the future"
            )));
        }
        // An omitted valid_from becomes the time the episode is stored, which is about now.
        if let Some(expires_at) = fact.expires_at {
            if expires_at <= fact.valid_from.unwrap_or(now) {
                return Err(Error::InvalidInput(
                    "expires_at must be after valid_from".into(),
                ));
            }
        }
        non_empty("statement", &fact.statement)?;
        non_empty("predicate", &fact.predicate)?;
        for entity in std::iter::once(&fact.subject).chain(&fact.object) {
            non_empty("entity name", &entity.name)?;
            non_empty("entity kind", &entity.kind)?;
            at_most("entity name", &entity.name, MAX_INDEXED_CHARS)?;
            at_most("entity kind", &entity.kind, MAX_KIND_CHARS)?;
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

fn at_most(field: &str, value: &str, max: usize) -> Result<()> {
    if value.chars().count() > max {
        return Err(Error::InvalidInput(format!(
            "{field} exceeds {max} characters"
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

/// Lines the episode's facts up with the request. A request fact that corroborates names its fact.
/// A fact the episode created was recorded in the transaction that processed it, at
/// `processed_at`, and one it only linked was recorded earlier. A request fact that names nothing,
/// or only `supersedes`, takes a linked fact with the same normalized statement if there is one,
/// since that is what it was matched to, and otherwise the next created fact in creation order. A
/// supersede matched that way reports the fact it closed as `superseded`. With no `processed_at`
/// every unnamed fact counts as created. The key check does not compare the fact lists, so a
/// replay may not match: whatever is left over is appended rather than hidden, and a request fact
/// with no match is skipped. `related` goes onto created facts by request position.
fn in_request_order(
    req: &CommitRequest,
    facts: Vec<Fact>,
    processed_at: Option<DateTime<Utc>>,
    related: Vec<Vec<RelatedFact>>,
) -> Vec<CommittedFact> {
    let named: HashSet<Uuid> = req.facts.iter().filter_map(|f| f.corroborates).collect();
    let as_committed = |fact: &Fact, corroborated: bool, related: Vec<RelatedFact>| CommittedFact {
        id: fact.id,
        status: fact.status,
        corroborated,
        related,
        superseded: None,
    };
    let (created, linked): (Vec<&Fact>, Vec<&Fact>) = facts
        .iter()
        .filter(|fact| !named.contains(&fact.id))
        .partition(|fact| processed_at.is_none_or(|at| fact.recorded_at >= at));
    let mut created = created.into_iter();
    let mut related = related.into_iter();
    let mut linked_reported = HashSet::new();
    let mut ordered = Vec::with_capacity(facts.len());
    for request_fact in &req.facts {
        let fact_related = related.next().unwrap_or_default();
        let wanted = normalize_statement(&request_fact.statement);
        let repeated = linked
            .iter()
            .find(|fact| normalize_statement(&fact.statement) == wanted);
        let found = match (request_fact.corroborates, request_fact.supersedes, repeated) {
            (Some(id), _, _) => facts
                .iter()
                .find(|fact| fact.id == id)
                .map(|fact| as_committed(fact, true, Vec::new())),
            (None, supersedes, Some(fact)) => {
                linked_reported.insert(fact.id);
                Some(CommittedFact {
                    superseded: supersedes,
                    ..as_committed(fact, true, Vec::new())
                })
            }
            (None, _, _) => created
                .next()
                .map(|fact| as_committed(fact, false, fact_related)),
        };
        ordered.extend(found);
    }
    ordered.extend(created.map(|fact| as_committed(fact, false, Vec::new())));
    ordered.extend(
        linked
            .iter()
            .filter(|fact| !linked_reported.contains(&fact.id))
            .map(|fact| as_committed(fact, true, Vec::new())),
    );
    ordered
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
            expires_at: None,
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
    fn refuses_names_and_keys_over_the_limit_counted_in_characters() {
        let long = "é".repeat(MAX_INDEXED_CHARS);
        let with_name = |name: String| {
            let mut one = fact();
            one.subject.name = name;
            request(vec![one])
        };
        assert!(validate_request(&with_name(long.clone())).is_ok());
        assert!(matches!(
            validate_request(&with_name(format!("{long}é"))),
            Err(Error::InvalidInput(_))
        ));

        let mut object = fact();
        object.object = Some(EntityRef {
            name: format!("{long}é"),
            kind: "k".into(),
        });
        assert!(validate_request(&request(vec![object])).is_err());

        let mut kind = fact();
        kind.subject.kind = "k".repeat(MAX_KIND_CHARS + 1);
        assert!(validate_request(&request(vec![kind])).is_err());

        let mut keyed = request(Vec::new());
        keyed.idempotency_key = Some(long.clone());
        assert!(validate_request(&keyed).is_ok());
        keyed.idempotency_key = Some(format!("{long}é"));
        assert!(matches!(
            validate_request(&keyed),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn refuses_valid_from_over_a_day_ahead() {
        let at = |offset: Duration| {
            let mut one = fact();
            one.valid_from = Some(Utc::now() + offset);
            request(vec![one])
        };
        assert!(validate_request(&at(Duration::days(-3650))).is_ok());
        assert!(validate_request(&at(Duration::hours(23))).is_ok());
        assert!(matches!(
            validate_request(&at(Duration::hours(25))),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn refuses_expiry_not_after_the_start() {
        let at = |valid_from: Option<DateTime<Utc>>, expires_at: DateTime<Utc>| {
            let mut one = fact();
            one.valid_from = valid_from;
            one.expires_at = Some(expires_at);
            request(vec![one])
        };
        let start = Utc::now() - Duration::days(10);
        assert!(validate_request(&at(Some(start), start + Duration::days(1))).is_ok());
        assert!(validate_request(&at(None, Utc::now() + Duration::hours(1))).is_ok());
        for bad in [
            at(Some(start), start),
            at(Some(start), start - Duration::days(1)),
            at(None, Utc::now() - Duration::hours(1)),
        ] {
            assert!(matches!(
                validate_request(&bad),
                Err(Error::InvalidInput(_))
            ));
        }
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
            redacted: false,
            facts: vec![
                CommittedFact {
                    id,
                    status: FactStatus::Quarantined,
                    corroborated: false,
                    related: Vec::new(),
                    superseded: None,
                },
                CommittedFact {
                    id,
                    status: FactStatus::Active,
                    corroborated: true,
                    related: Vec::new(),
                    superseded: Some(id),
                },
                CommittedFact {
                    id,
                    status: FactStatus::Active,
                    corroborated: false,
                    related: vec![RelatedFact {
                        id,
                        statement: "r".into(),
                        score: 0.5,
                    }],
                    superseded: None,
                },
            ],
        };
        // The optional fields are left out when they say nothing.
        assert_eq!(
            serde_json::to_value(response).expect("serialize"),
            json!({"episode_id": id, "facts": [
                {"id": id, "status": "quarantined"},
                {"id": id, "status": "active", "corroborated": true, "superseded": id},
                {"id": id, "status": "active", "related": [{"id": id, "statement": "r", "score": 0.5}]},
            ], "redacted": false})
        );
    }

    #[test]
    fn redaction_covers_content_statements_and_entity_names() {
        let mut one = fact();
        one.statement = "Ada set password=hunter2".into();
        one.subject.name = "sk-abcdefghijklmnopqrstuvwxyz".into();
        one.object = Some(EntityRef {
            name: "AKIAIOSFODNN7EXAMPLE".into(),
            kind: "key".into(),
        });
        one.predicate = "uses_key".into();
        let mut req = request(vec![one]);
        req.content = "token=abc123".into();
        assert!(redact_request(&mut req));
        assert_eq!(req.content, "token=[redacted]");
        assert_eq!(req.facts[0].statement, "Ada set password=[redacted]");
        assert_eq!(req.facts[0].subject.name, "[redacted]");
        assert_eq!(
            req.facts[0].object.as_ref().map(|o| o.name.as_str()),
            Some("[redacted]")
        );
        assert_eq!(req.facts[0].predicate, "uses_key");

        let mut clean = request(vec![fact()]);
        assert!(!redact_request(&mut clean));
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

    fn stored(id: Uuid, status: FactStatus) -> Fact {
        Fact {
            id,
            scope: Scope::user("u"),
            subject_id: Uuid::now_v7(),
            predicate: "p".into(),
            object_id: None,
            statement: "s".into(),
            trust: Trust::Agent,
            status,
            valid_from: Utc::now(),
            valid_to: None,
            expires_at: None,
            recorded_at: Utc::now(),
            invalidated_at: None,
            superseded_by: None,
            pending_supersede_of: None,
            source_episode_ids: Vec::new(),
        }
    }

    fn got(committed: &[CommittedFact]) -> Vec<(Uuid, FactStatus)> {
        committed.iter().map(|f| (f.id, f.status)).collect()
    }

    /// The corroborated fact is older than both new ones, so id order alone would misplace it.
    #[test]
    fn response_follows_the_request_and_not_id_order() {
        let linked = Uuid::now_v7();
        let (first, second) = (Uuid::now_v7(), Uuid::now_v7());
        let mut corroborating = fact();
        corroborating.corroborates = Some(linked);
        let req = request(vec![fact(), corroborating, fact()]);
        let facts = vec![
            stored(linked, FactStatus::Active),
            stored(first, FactStatus::Active),
            stored(second, FactStatus::Quarantined),
        ];
        assert_eq!(
            got(&in_request_order(&req, facts, None, Vec::new())),
            [
                (first, FactStatus::Active),
                (linked, FactStatus::Active),
                (second, FactStatus::Quarantined),
            ]
        );
    }

    #[test]
    fn a_replay_that_does_not_match_hides_nothing() {
        let (first, second) = (Uuid::now_v7(), Uuid::now_v7());
        let mut gone = fact();
        gone.corroborates = Some(Uuid::now_v7());
        let req = request(vec![gone, fact()]);
        let facts = vec![
            stored(first, FactStatus::Active),
            stored(second, FactStatus::Active),
        ];
        assert_eq!(
            got(&in_request_order(&req, facts, None, Vec::new())),
            [(first, FactStatus::Active), (second, FactStatus::Active)]
        );
    }

    fn said(statement: &str) -> CommitFact {
        let mut one = fact();
        one.statement = statement.into();
        one
    }

    fn scored(scope: &Scope, statement: &str, score: f64) -> ScoredFact {
        let mut found = stored(Uuid::now_v7(), FactStatus::Active);
        found.scope = scope.clone();
        found.statement = statement.into();
        ScoredFact { fact: found, score }
    }

    #[test]
    fn statements_match_ignoring_case_spacing_and_a_final_period() {
        let base = normalize_statement("Ada likes tea");
        for variant in ["ada LIKES tea", "  Ada \t likes\ntea. ", "Ada likes tea..."] {
            assert_eq!(normalize_statement(variant), base, "{variant:?}");
        }
        for different in ["Ada likes teas", "Ada likes tea, black", "Ada.likes tea"] {
            assert_ne!(normalize_statement(different), base, "{different:?}");
        }
    }

    #[test]
    fn pick_takes_the_first_repeat_and_reports_no_related_with_it() {
        let scope = Scope::user("u");
        let found = vec![
            scored(&scope, "Ada likes coffee", 0.9),
            scored(&scope, "ada likes tea.", 0.8),
        ];
        let repeat = found[1].fact.id;
        let matched = pick(&scope, "Ada likes tea", found, &HashSet::new());
        assert_eq!(matched.existing, Some(repeat));
        assert!(matched.related.is_empty());
    }

    #[test]
    fn pick_lists_at_most_three_related_in_rank_order() {
        let scope = Scope::user("u");
        let found: Vec<ScoredFact> = (0..5)
            .map(|i| {
                scored(
                    &scope,
                    &format!("Ada likes drink {i}"),
                    1.0 - f64::from(i) / 10.0,
                )
            })
            .collect();
        let expected: Vec<Uuid> = found.iter().take(3).map(|s| s.fact.id).collect();
        let matched = pick(&scope, "Ada likes tea", found, &HashSet::new());
        assert_eq!(matched.existing, None);
        let ids: Vec<Uuid> = matched.related.iter().map(|r| r.id).collect();
        assert_eq!(ids, expected);
        assert_eq!(matched.related[0].statement, "Ada likes drink 0");
    }

    #[test]
    fn pick_never_matches_another_scope_an_expired_a_quarantined_or_a_closing_fact() {
        let scope = Scope::user("u");
        let other_scope = scored(&Scope::user("v"), "Ada likes tea", 0.9);
        let mut expired = scored(&scope, "Ada likes tea", 0.85);
        expired.fact.expires_at = Some(Utc::now() - Duration::minutes(1));
        let mut quarantined = scored(&scope, "Ada likes tea", 0.8);
        quarantined.fact.status = FactStatus::Quarantined;
        let mut superseded = scored(&scope, "Ada likes tea", 0.7);
        superseded.fact.superseded_by = Some(Uuid::now_v7());
        let mut invalidated = scored(&scope, "Ada likes tea", 0.65);
        invalidated.fact.invalidated_at = Some(Utc::now());
        let closing = scored(&scope, "Ada likes tea", 0.6);
        let closing_id = closing.fact.id;
        let found = vec![
            other_scope,
            expired,
            quarantined,
            superseded,
            invalidated,
            closing,
        ];
        let matched = pick(&scope, "Ada likes tea", found, &HashSet::from([closing_id]));
        assert_eq!(matched, Matched::default());
    }

    /// A replay has no matching results, so a linked fact is told apart from a created one by
    /// when it was recorded.
    #[test]
    fn a_replay_reports_a_matched_repeat_where_the_request_had_it() {
        let processed_at = Utc::now();
        let mut repeat = stored(Uuid::now_v7(), FactStatus::Active);
        repeat.statement = "Ada likes tea".into();
        repeat.recorded_at = processed_at - Duration::days(1);
        let mut created = stored(Uuid::now_v7(), FactStatus::Quarantined);
        created.statement = "Ada likes chess".into();
        created.recorded_at = processed_at;
        let req = request(vec![said("Ada likes chess"), said("ADA likes tea.")]);
        let related = vec![RelatedFact {
            id: repeat.id,
            statement: repeat.statement.clone(),
            score: 0.4,
        }];
        let ordered = in_request_order(
            &req,
            vec![repeat.clone(), created.clone()],
            Some(processed_at),
            vec![related.clone(), Vec::new()],
        );
        assert_eq!(
            got(&ordered),
            [
                (created.id, FactStatus::Quarantined),
                (repeat.id, FactStatus::Active)
            ]
        );
        assert_eq!(
            (ordered[0].corroborated, ordered[1].corroborated),
            (false, true)
        );
        assert_eq!(ordered[0].related, related);
        assert!(ordered[1].related.is_empty());
    }
}
