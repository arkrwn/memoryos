//! Fact write logic shared by `apply_extraction`, `approve_fact` and the purge paths.
//! Every function runs inside the caller's tenant transaction.

use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use memoryos_core::{
    source_requires_quarantine, Error, FactOp, FactStatus, NewFact, PurgeReport, RequestContext,
    Result, Scope, Trust,
};
use serde_json::{json, Value};
use sqlx::{PgConnection, Row};
use uuid::Uuid;

use crate::{db, validate};

/// Guards the supersession walk against corrupted data. Real chains are far shorter.
const MAX_CHAIN_STEPS: usize = 10_000;

pub(crate) async fn audit(
    conn: &mut PgConnection,
    ctx: &RequestContext,
    action: &str,
    target_id: Option<Uuid>,
    detail: Value,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO audit_events (id, tenant, actor, action, target_id, detail) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(Uuid::now_v7())
    .bind(&ctx.tenant)
    .bind(&ctx.actor)
    .bind(action)
    .bind(target_id)
    .bind(detail)
    .execute(conn)
    .await
    .map_err(db)?;
    Ok(())
}

/// The episode a set of ops is applied for, already locked by the caller.
pub(crate) struct LockedEpisode {
    pub(crate) id: Uuid,
    pub(crate) scope: Scope,
    pub(crate) trust: Trust,
}

/// Applies every op for one unprocessed episode. Returns ids of inserted facts.
pub(crate) async fn apply_ops(
    conn: &mut PgConnection,
    ctx: &RequestContext,
    episode: &LockedEpisode,
    ops: &[FactOp],
) -> Result<Vec<Uuid>> {
    // Taking the row locks in id order keeps two extractions touching the same facts from
    // deadlocking. Missing rows are reported per op below.
    let referenced: Vec<Uuid> = ops
        .iter()
        .filter_map(|op| match op {
            FactOp::Add(_) => None,
            FactOp::Supersede { old_fact_id, .. } => Some(*old_fact_id),
            FactOp::Corroborate { fact_id } => Some(*fact_id),
        })
        .collect();
    if !referenced.is_empty() {
        sqlx::query("SELECT id FROM facts WHERE id = ANY($1) ORDER BY id FOR UPDATE")
            .bind(&referenced)
            .fetch_all(&mut *conn)
            .await
            .map_err(db)?;
    }

    let mut inserted = Vec::new();
    for op in ops {
        match op {
            FactOp::Add(new) => {
                inserted.push(insert_fact(conn, ctx, episode, new, None).await?);
            }
            FactOp::Supersede { old_fact_id, new } => {
                require_fact_in_scope(conn, *old_fact_id, &episode.scope).await?;
                let pending = (new.status == FactStatus::Quarantined).then_some(*old_fact_id);
                let id = insert_fact(conn, ctx, episode, new, pending).await?;
                if new.status == FactStatus::Active {
                    let settled =
                        settle_supersede(conn, id, new.trust, new.valid_from, *old_fact_id, None)
                            .await?;
                    // The gate compared against `old_fact_id`, which may no longer be the head.
                    if let Settled::Outranked { head } = settled {
                        sqlx::query(
                            "UPDATE facts SET status = 'quarantined', pending_supersede_of = $2 \
                             WHERE id = $1",
                        )
                        .bind(id)
                        .bind(head)
                        .execute(&mut *conn)
                        .await
                        .map_err(db)?;
                    }
                }
                inserted.push(id);
            }
            FactOp::Corroborate { fact_id } => {
                require_fact_in_scope(conn, *fact_id, &episode.scope).await?;
                link_source(conn, ctx, *fact_id, episode.id).await?;
            }
        }
    }
    Ok(inserted)
}

async fn require_fact_in_scope(conn: &mut PgConnection, id: Uuid, scope: &Scope) -> Result<()> {
    let row = sqlx::query("SELECT scope FROM facts WHERE id = $1")
        .bind(id)
        .fetch_optional(conn)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("fact {id}")))?;
    let found: String = row.try_get("scope").map_err(db)?;
    if found != scope.to_string() {
        return Err(Error::InvalidInput(format!(
            "fact {id} is outside the episode scope"
        )));
    }
    Ok(())
}

/// Checks the invariants of SPEC section 5, then inserts the fact and its source link. The store
/// refuses a fact that breaks the trust rules even if the caller computed it wrongly: this is the
/// last check before poisoned content could reach recall.
async fn insert_fact(
    conn: &mut PgConnection,
    ctx: &RequestContext,
    episode: &LockedEpisode,
    new: &NewFact,
    pending_supersede_of: Option<Uuid>,
) -> Result<Uuid> {
    validate::non_empty("predicate", &new.predicate)?;
    validate::non_empty("statement", &new.statement)?;
    let embedding = validate::embedding(&new.embedding)?;
    if new.scope != episode.scope {
        return Err(Error::InvalidInput(
            "fact scope differs from episode scope".into(),
        ));
    }
    if new.trust != episode.trust {
        return Err(Error::InvalidInput(
            "fact trust differs from episode trust".into(),
        ));
    }
    match new.status {
        FactStatus::Rejected => {
            return Err(Error::InvalidInput("a new fact cannot be rejected".into()))
        }
        FactStatus::Active if source_requires_quarantine(&episode.scope, episode.trust) => {
            return Err(Error::InvalidInput(format!(
                "{} facts in this scope must be quarantined",
                episode.trust.as_str()
            )))
        }
        _ => {}
    }
    if new.valid_to.is_some_and(|to| to < new.valid_from) {
        return Err(Error::InvalidInput("valid_to is before valid_from".into()));
    }
    require_entity_in_scope(conn, new.subject_id, &episode.scope).await?;
    if let Some(object_id) = new.object_id {
        require_entity_in_scope(conn, object_id, &episode.scope).await?;
    }

    let id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO facts (id, tenant, scope, subject_id, predicate, object_id, statement, \
         embedding, trust, status, valid_from, valid_to, pending_supersede_of) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
    )
    .bind(id)
    .bind(&ctx.tenant)
    .bind(new.scope.to_string())
    .bind(new.subject_id)
    .bind(&new.predicate)
    .bind(new.object_id)
    .bind(&new.statement)
    .bind(embedding)
    .bind(new.trust.as_str())
    .bind(new.status.as_str())
    .bind(new.valid_from)
    .bind(new.valid_to)
    .bind(pending_supersede_of)
    .execute(&mut *conn)
    .await
    .map_err(db)?;
    link_source(conn, ctx, id, episode.id).await?;
    Ok(id)
}

/// Foreign key checks bypass row-level security, so this read (which RLS does filter) is what
/// stops a fact from pointing at another tenant's entity.
async fn require_entity_in_scope(conn: &mut PgConnection, id: Uuid, scope: &Scope) -> Result<()> {
    let row = sqlx::query("SELECT scope FROM entities WHERE id = $1")
        .bind(id)
        .fetch_optional(conn)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("entity {id}")))?;
    let found: String = row.try_get("scope").map_err(db)?;
    if found != scope.to_string() {
        return Err(Error::InvalidInput(format!(
            "entity {id} is outside the episode scope"
        )));
    }
    Ok(())
}

async fn link_source(
    conn: &mut PgConnection,
    ctx: &RequestContext,
    fact_id: Uuid,
    episode_id: Uuid,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO fact_sources (fact_id, episode_id, tenant) VALUES ($1, $2, $3) \
         ON CONFLICT DO NOTHING",
    )
    .bind(fact_id)
    .bind(episode_id)
    .bind(&ctx.tenant)
    .execute(conn)
    .await
    .map_err(db)?;
    Ok(())
}

/// What `settle_supersede` did.
pub(crate) enum Settled {
    /// The chain is settled, or its head had nothing left to close.
    Done,
    /// The head has higher trust and nothing was written. The caller decides what happens to the
    /// new fact, which must not stay active next to a head it was meant to replace.
    Outranked { head: Uuid },
}

/// Applies SPEC section 4 between `new_id` and `old_id`.
///
/// If another extraction already superseded `old_id`, the walk moves to the current head of the
/// chain, so a fact never gets two successors. A retracted or inactive head has nothing left to
/// close and the new fact simply stays open. Trust is checked against the head, since the caller
/// only saw `old_id`. `reviewed` names a fact an admin already decided this one may replace, which
/// is exempt from that check.
pub(crate) async fn settle_supersede(
    conn: &mut PgConnection,
    new_id: Uuid,
    new_trust: Trust,
    new_valid_from: DateTime<Utc>,
    old_id: Uuid,
    reviewed: Option<Uuid>,
) -> Result<Settled> {
    let mut current = old_id;
    for _ in 0..MAX_CHAIN_STEPS {
        let row = sqlx::query(
            "SELECT status, trust, invalidated_at IS NOT NULL AS invalidated, valid_from, \
             superseded_by FROM facts WHERE id = $1 FOR UPDATE",
        )
        .bind(current)
        .fetch_one(&mut *conn)
        .await
        .map_err(db)?;
        if let Some(next) = row
            .try_get::<Option<Uuid>, _>("superseded_by")
            .map_err(db)?
        {
            current = next;
            continue;
        }
        let status: String = row.try_get("status").map_err(db)?;
        let invalidated: bool = row.try_get("invalidated").map_err(db)?;
        if status != FactStatus::Active.as_str() || invalidated {
            return Ok(Settled::Done);
        }
        let head_trust: Trust = row
            .try_get::<String, _>("trust")
            .map_err(db)?
            .parse()
            .map_err(|_| Error::Storage("fact has an invalid trust".into()))?;
        if head_trust > new_trust && reviewed != Some(current) {
            return Ok(Settled::Outranked { head: current });
        }
        let head_valid_from: DateTime<Utc> = row.try_get("valid_from").map_err(db)?;
        // Out of order: the new fact is history and the head stays current. Otherwise the head
        // closes. LEAST ignores NULL, and it keeps an earlier end date the fact already carried.
        let (closed, end, successor) = if new_valid_from < head_valid_from {
            (new_id, head_valid_from, current)
        } else {
            (current, new_valid_from, new_id)
        };
        sqlx::query(
            "UPDATE facts SET valid_to = LEAST(valid_to, $2), superseded_by = $3 WHERE id = $1",
        )
        .bind(closed)
        .bind(end)
        .bind(successor)
        .execute(&mut *conn)
        .await
        .map_err(db)?;
        return Ok(Settled::Done);
    }
    Err(Error::Internal(format!(
        "supersession chain from fact {old_id} exceeds {MAX_CHAIN_STEPS} steps"
    )))
}

/// A fact leaving its supersession chain, because it is deleted or goes back to quarantine.
struct ChainExit {
    valid_from: DateTime<Utc>,
    valid_to: Option<DateTime<Utc>>,
    superseded_by: Option<Uuid>,
}

/// Follows `start` past facts leaving the chain, to the first one staying in it.
fn surviving_successor(exits: &HashMap<Uuid, ChainExit>, start: Option<Uuid>) -> Option<Uuid> {
    let mut current = start;
    while let Some(exit) = current.and_then(|id| exits.get(&id)) {
        current = exit.superseded_by;
    }
    current
}

/// What a purge did, including the part `PurgeReport` has no field for.
pub(crate) struct PurgeOutcome {
    pub(crate) report: PurgeReport,
    pub(crate) facts_requarantined: u64,
}

/// `None` when no source remains.
fn highest_trust(trusts: Vec<String>) -> Result<Option<Trust>> {
    let mut highest = None;
    for text in trusts {
        let trust: Trust = text
            .parse()
            .map_err(|_| Error::Storage("episode has an invalid trust".into()))?;
        highest = highest.max(Some(trust));
    }
    Ok(highest)
}

/// Deletes one episode and what only it supports. Facts that keep other sources take the highest
/// trust among them, so a poisoned fact cannot stay active on the trust of the source that was
/// purged. The caller holds the episode row lock.
pub(crate) async fn purge_episode(conn: &mut PgConnection, id: Uuid) -> Result<PurgeOutcome> {
    // Lock before deciding: a concurrent corroboration holds these rows until it commits, and the
    // next statement then sees the source it added.
    sqlx::query(
        "SELECT f.id FROM facts f JOIN fact_sources s ON s.fact_id = f.id \
         WHERE s.episode_id = $1 ORDER BY f.id FOR UPDATE OF f",
    )
    .bind(id)
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;

    let rows = sqlx::query(
        "SELECT f.id, f.scope, f.status, f.valid_from, f.valid_to, f.superseded_by, f.subject_id, \
             f.object_id, ARRAY(SELECT e.trust FROM fact_sources o \
                 JOIN episodes e ON e.id = o.episode_id \
                 WHERE o.fact_id = f.id AND o.episode_id <> $1) AS remaining_trusts \
         FROM facts f JOIN fact_sources s ON s.fact_id = f.id \
         WHERE s.episode_id = $1",
    )
    .bind(id)
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;

    let mut exits: HashMap<Uuid, ChainExit> = HashMap::new();
    let mut doomed_ids = Vec::new();
    let mut requarantined = Vec::new();
    let mut retrusted = Vec::new();
    let mut touched_entities = BTreeSet::new();
    for row in &rows {
        let fact_id: Uuid = row.try_get("id").map_err(db)?;
        let exit = ChainExit {
            valid_from: row.try_get("valid_from").map_err(db)?,
            valid_to: row.try_get("valid_to").map_err(db)?,
            superseded_by: row.try_get("superseded_by").map_err(db)?,
        };
        let Some(trust) = highest_trust(row.try_get("remaining_trusts").map_err(db)?)? else {
            touched_entities.insert(row.try_get::<Uuid, _>("subject_id").map_err(db)?);
            if let Some(object_id) = row.try_get::<Option<Uuid>, _>("object_id").map_err(db)? {
                touched_entities.insert(object_id);
            }
            doomed_ids.push(fact_id);
            exits.insert(fact_id, exit);
            continue;
        };
        let scope: Scope = row
            .try_get::<String, _>("scope")
            .map_err(db)?
            .parse()
            .map_err(|_| Error::Storage("fact has an invalid scope".into()))?;
        let status: String = row.try_get("status").map_err(db)?;
        if status == FactStatus::Active.as_str() && source_requires_quarantine(&scope, trust) {
            requarantined.push((fact_id, trust));
            exits.insert(fact_id, exit);
        } else {
            retrusted.push((fact_id, trust));
        }
    }

    let reopened = restore_predecessors(conn, &exits).await?;

    for (fact_id, trust) in &retrusted {
        sqlx::query("UPDATE facts SET trust = $2 WHERE id = $1")
            .bind(fact_id)
            .bind(trust.as_str())
            .execute(&mut *conn)
            .await
            .map_err(db)?;
    }
    for (fact_id, trust) in &requarantined {
        // Approval puts it back where it was: behind a survivor that followed it, or else closing
        // the predecessor that was just reopened.
        let start = exits.get(fact_id).and_then(|exit| exit.superseded_by);
        let pending = surviving_successor(&exits, start).or_else(|| reopened.get(fact_id).copied());
        sqlx::query(
            "UPDATE facts SET status = 'quarantined', trust = $2, pending_supersede_of = $3, \
                 valid_to = CASE WHEN superseded_by IS NULL THEN valid_to END, \
                 superseded_by = NULL \
             WHERE id = $1",
        )
        .bind(fact_id)
        .bind(trust.as_str())
        .bind(pending)
        .execute(&mut *conn)
        .await
        .map_err(db)?;
    }

    let facts_deleted = sqlx::query("DELETE FROM facts WHERE id = ANY($1)")
        .bind(&doomed_ids)
        .execute(&mut *conn)
        .await
        .map_err(db)?
        .rows_affected();
    let episodes_deleted = sqlx::query("DELETE FROM episodes WHERE id = $1")
        .bind(id)
        .execute(&mut *conn)
        .await
        .map_err(db)?
        .rows_affected();
    let entities_deleted = delete_orphan_entities(conn, touched_entities).await?;

    Ok(PurgeOutcome {
        report: PurgeReport {
            facts_deleted,
            episodes_deleted,
            entities_deleted,
        },
        facts_requarantined: requarantined.len() as u64,
    })
}

/// A fact leaving the chain must not leave its effects behind: a fact it closed takes over its end
/// and successor. If it was current, its predecessor becomes current again. Returns, for each fact
/// leaving, the surviving predecessor with the latest start.
async fn restore_predecessors(
    conn: &mut PgConnection,
    exits: &HashMap<Uuid, ChainExit>,
) -> Result<HashMap<Uuid, Uuid>> {
    let exit_ids: Vec<Uuid> = exits.keys().copied().collect();
    let predecessors = sqlx::query(
        "SELECT id, superseded_by FROM (SELECT id, superseded_by, valid_from FROM facts \
             WHERE superseded_by = ANY($1) AND NOT id = ANY($1) ORDER BY id FOR UPDATE) locked \
         ORDER BY valid_from, id",
    )
    .bind(&exit_ids)
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;

    let mut reopened = HashMap::new();
    for row in &predecessors {
        let id: Uuid = row.try_get("id").map_err(db)?;
        let direct: Uuid = row.try_get("superseded_by").map_err(db)?;
        let Some(first) = exits.get(&direct) else {
            continue;
        };
        // Skip over runs of leaving facts to the first survivor, if there is one.
        reopened.insert(direct, id);
        let mut last = first;
        while let Some((next_id, next)) = last.superseded_by.and_then(|s| exits.get_key_value(&s)) {
            reopened.insert(*next_id, id);
            last = next;
        }
        // An end date earlier than the leaving fact's start was not set by that supersede: keep it.
        sqlx::query(
            "UPDATE facts SET \
                 valid_to = CASE WHEN valid_to < $2 THEN valid_to ELSE $3 END, \
                 superseded_by = $4 \
             WHERE id = $1",
        )
        .bind(id)
        .bind(first.valid_from)
        .bind(last.valid_to)
        .bind(last.superseded_by)
        .execute(&mut *conn)
        .await
        .map_err(db)?;
    }
    Ok(reopened)
}

/// Deletes the given entities if no fact references them. The row locks make the check safe: an
/// in-flight fact insert holds a key-share lock on its entity, so this waits for it to commit and
/// the following statement sees that fact.
async fn delete_orphan_entities(
    conn: &mut PgConnection,
    candidates: BTreeSet<Uuid>,
) -> Result<u64> {
    let ids: Vec<Uuid> = candidates.into_iter().collect();
    sqlx::query("SELECT id FROM entities WHERE id = ANY($1) ORDER BY id FOR UPDATE")
        .bind(&ids)
        .fetch_all(&mut *conn)
        .await
        .map_err(db)?;
    let deleted = sqlx::query(
        "DELETE FROM entities e WHERE e.id = ANY($1) AND NOT EXISTS ( \
             SELECT 1 FROM facts f WHERE f.subject_id = e.id OR f.object_id = e.id)",
    )
    .bind(&ids)
    .execute(conn)
    .await
    .map_err(db)?
    .rows_affected();
    Ok(deleted)
}

/// Hard-deletes a scope's facts, episodes, entities and sessions, plus an agent scope's blocks.
pub(crate) async fn purge_scope(conn: &mut PgConnection, scope: &Scope) -> Result<PurgeReport> {
    let scope_text = scope.to_string();
    let mut deleted = [0_u64; 3];
    let statements = [
        "DELETE FROM facts WHERE scope = $1",
        "DELETE FROM episodes WHERE scope = $1",
        "DELETE FROM entities WHERE scope = $1",
    ];
    for (count, sql) in deleted.iter_mut().zip(statements) {
        *count = sqlx::query(sql)
            .bind(&scope_text)
            .execute(&mut *conn)
            .await
            .map_err(db)?
            .rows_affected();
    }
    sqlx::query("DELETE FROM sessions WHERE scope = $1")
        .bind(&scope_text)
        .execute(&mut *conn)
        .await
        .map_err(db)?;
    if let Scope::Agent(agent_id) = scope {
        for sql in [
            "DELETE FROM sessions WHERE agent_id = $1",
            "DELETE FROM blocks WHERE agent_id = $1",
        ] {
            sqlx::query(sql)
                .bind(agent_id)
                .execute(&mut *conn)
                .await
                .map_err(db)?;
        }
    }
    let [facts_deleted, episodes_deleted, entities_deleted] = deleted;
    Ok(PurgeReport {
        facts_deleted,
        episodes_deleted,
        entities_deleted,
    })
}

/// Audit detail for a purge. Counts only, never content.
pub(crate) fn purge_detail(report: &PurgeReport) -> Value {
    json!({
        "facts_deleted": report.facts_deleted,
        "episodes_deleted": report.episodes_deleted,
        "entities_deleted": report.entities_deleted,
    })
}
