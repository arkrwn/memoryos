//! Fact write logic shared by commits, approval and purge paths. Every query is tenant-scoped;
//! the caller holds one IMMEDIATE transaction, which replaces PostgreSQL's row locks.

use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use memoryos_core::{
    source_requires_quarantine, Error, FactOp, FactStatus, NewFact, PurgeReport, RequestContext,
    Result, Scope, Trust,
};
use serde_json::{json, Value};
use sqlx::sqlite::SqliteRow;
use sqlx::{Row, SqliteConnection};
use uuid::Uuid;

use super::{db, embedding, json_array, rows, storable, uuid_strings};
use sqlx::AssertSqlSafe;

const MAX_CHAIN_STEPS: usize = 10_000;

pub(super) async fn audit(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    action: &str,
    target_id: Option<Uuid>,
    detail: Value,
    now: DateTime<Utc>,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO audit_events (id, tenant, actor, action, target_id, detail, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )
    .bind(Uuid::now_v7().to_string())
    .bind(&ctx.tenant)
    .bind(&ctx.actor)
    .bind(action)
    .bind(target_id.map(|id| id.to_string()))
    .bind(json_array(&detail)?)
    .bind(rows::timestamp(now))
    .execute(conn)
    .await
    .map_err(db)?;
    Ok(())
}

pub(super) struct LockedEpisode {
    pub(super) id: Uuid,
    pub(super) scope: Scope,
    pub(super) trust: Trust,
}

pub(super) async fn apply_ops(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    episode: &LockedEpisode,
    ops: &[FactOp],
    now: DateTime<Utc>,
) -> Result<Vec<Uuid>> {
    let mut inserted = Vec::new();
    for op in ops {
        match op {
            FactOp::Add(new) => {
                let id = insert_fact(conn, ctx, episode, new, None, now).await?;
                if new.status == FactStatus::Active
                    && outranked_by_current(conn, ctx, new, now).await?
                {
                    sqlx::query(
                        "UPDATE facts SET status = 'quarantined' WHERE tenant = ?1 AND id = ?2",
                    )
                    .bind(&ctx.tenant)
                    .bind(id.to_string())
                    .execute(&mut *conn)
                    .await
                    .map_err(db)?;
                }
                inserted.push(id);
            }
            FactOp::Supersede { old_fact_id, new } => {
                require_fact_in_scope(conn, ctx, *old_fact_id, &episode.scope).await?;
                let pending = (new.status == FactStatus::Quarantined).then_some(*old_fact_id);
                let id = insert_fact(conn, ctx, episode, new, pending, now).await?;
                if new.status == FactStatus::Active {
                    let settled = settle_supersede(
                        conn,
                        ctx,
                        id,
                        new.trust,
                        new.valid_from,
                        *old_fact_id,
                        None,
                        None,
                    )
                    .await?;
                    if let Settled::Outranked { head } = settled {
                        sqlx::query(
                            "UPDATE facts SET status = 'quarantined', pending_supersede_of = ?3 \
                             WHERE tenant = ?1 AND id = ?2",
                        )
                        .bind(&ctx.tenant)
                        .bind(id.to_string())
                        .bind(head.to_string())
                        .execute(&mut *conn)
                        .await
                        .map_err(db)?;
                    }
                }
                inserted.push(id);
            }
            FactOp::Corroborate { fact_id } => {
                require_fact_in_scope(conn, ctx, *fact_id, &episode.scope).await?;
                link_source(conn, ctx, *fact_id, episode.id).await?;
            }
            FactOp::SupersedeWithExisting {
                old_fact_id,
                fact_id,
            } => {
                supersede_with_existing(conn, ctx, episode, *old_fact_id, *fact_id, now).await?;
            }
        }
    }
    Ok(inserted)
}

async fn supersede_with_existing(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    episode: &LockedEpisode,
    old_id: Uuid,
    existing_id: Uuid,
    now: DateTime<Utc>,
) -> Result<()> {
    require_fact_in_scope(conn, ctx, old_id, &episode.scope).await?;
    require_fact_in_scope(conn, ctx, existing_id, &episode.scope).await?;
    if old_id == existing_id {
        return Err(Error::InvalidInput(format!(
            "fact {old_id} cannot supersede itself"
        )));
    }
    if source_requires_quarantine(&episode.scope, episode.trust) {
        return Err(Error::InvalidInput(format!(
            "{} facts in this scope must be quarantined",
            episode.trust.as_str()
        )));
    }
    let row = sqlx::query(
        "SELECT valid_from FROM facts WHERE tenant = ?1 AND id = ?2 AND status = 'active' \
         AND superseded_by IS NULL AND invalidated_at IS NULL \
         AND (expires_at IS NULL OR expires_at > ?3)",
    )
    .bind(&ctx.tenant)
    .bind(existing_id.to_string())
    .bind(rows::timestamp(now))
    .fetch_optional(&mut *conn)
    .await
    .map_err(db)?
    .ok_or_else(|| Error::InvalidInput(format!("fact {existing_id} is not current")))?;
    let valid_from = time(&row, "valid_from")?;
    link_source(conn, ctx, existing_id, episode.id).await?;
    let settled = settle_supersede(
        conn,
        ctx,
        existing_id,
        episode.trust,
        valid_from,
        old_id,
        None,
        Some(episode.id),
    )
    .await?;
    match settled {
        Settled::Outranked { head } if head == old_id => Err(Error::InvalidInput(format!(
            "fact {old_id} has higher trust than this source"
        ))),
        _ => Ok(()),
    }
}

async fn outranked_by_current(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    new: &NewFact,
    now: DateTime<Utc>,
) -> Result<bool> {
    let higher: Vec<String> = [Trust::External, Trust::Agent, Trust::User, Trust::System]
        .into_iter()
        .filter(|trust| *trust > new.trust)
        .map(|trust| trust.as_str().to_owned())
        .collect();
    let filter = rows::visible(5, 5);
    let sql = format!(
        "SELECT EXISTS (SELECT 1 FROM facts f WHERE f.tenant = ?1 AND f.scope = ?2 \
         AND f.subject_id = ?3 AND f.predicate = ?4 \
         AND f.trust IN (SELECT value FROM json_each(?6)) AND {filter})"
    );
    sqlx::query_scalar(AssertSqlSafe(sql))
        .bind(&ctx.tenant)
        .bind(new.scope.to_string())
        .bind(new.subject_id.to_string())
        .bind(&new.predicate)
        .bind(rows::timestamp(now))
        .bind(json_array(&higher)?)
        .fetch_one(conn)
        .await
        .map_err(db)
}

async fn require_fact_in_scope(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    id: Uuid,
    scope: &Scope,
) -> Result<()> {
    let row = sqlx::query("SELECT scope FROM facts WHERE tenant = ?1 AND id = ?2")
        .bind(&ctx.tenant)
        .bind(id.to_string())
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

async fn insert_fact(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    episode: &LockedEpisode,
    new: &NewFact,
    pending_supersede_of: Option<Uuid>,
    now: DateTime<Utc>,
) -> Result<Uuid> {
    crate::validate::non_empty("predicate", &new.predicate)?;
    crate::validate::non_empty("statement", &new.statement)?;
    storable(&new.predicate)?;
    storable(&new.statement)?;
    let embedding = embedding(&new.embedding)?;
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
    if new.expires_at.is_some_and(|at| at <= new.valid_from) {
        return Err(Error::InvalidInput(
            "expires_at must be after valid_from".into(),
        ));
    }
    require_entity_in_scope(conn, ctx, new.subject_id, &episode.scope).await?;
    if let Some(object_id) = new.object_id {
        require_entity_in_scope(conn, ctx, object_id, &episode.scope).await?;
    }

    let id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO facts (id, tenant, scope, subject_id, predicate, object_id, statement, \
         trust, status, valid_from, valid_to, pending_supersede_of, expires_at, recorded_at) \
         VALUES (?2, ?1, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
    )
    .bind(&ctx.tenant)
    .bind(id.to_string())
    .bind(new.scope.to_string())
    .bind(new.subject_id.to_string())
    .bind(&new.predicate)
    .bind(new.object_id.map(|id| id.to_string()))
    .bind(&new.statement)
    .bind(new.trust.as_str())
    .bind(new.status.as_str())
    .bind(rows::timestamp(new.valid_from))
    .bind(new.valid_to.map(rows::timestamp))
    .bind(pending_supersede_of.map(|id| id.to_string()))
    .bind(new.expires_at.map(rows::timestamp))
    .bind(rows::timestamp(now))
    .execute(&mut *conn)
    .await
    .map_err(db)?;
    sqlx::query(
        "INSERT INTO fact_embeddings (rowid, embedding) \
         SELECT rowid, ?3 FROM facts WHERE tenant = ?1 AND id = ?2",
    )
    .bind(&ctx.tenant)
    .bind(id.to_string())
    .bind(embedding)
    .execute(&mut *conn)
    .await
    .map_err(db)?;
    link_source(conn, ctx, id, episode.id).await?;
    Ok(id)
}

async fn require_entity_in_scope(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    id: Uuid,
    scope: &Scope,
) -> Result<()> {
    let row = sqlx::query("SELECT scope FROM entities WHERE tenant = ?1 AND id = ?2")
        .bind(&ctx.tenant)
        .bind(id.to_string())
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
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    fact_id: Uuid,
    episode_id: Uuid,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO fact_sources (fact_id, episode_id, tenant) VALUES (?2, ?3, ?1) \
         ON CONFLICT (fact_id, episode_id) DO NOTHING",
    )
    .bind(&ctx.tenant)
    .bind(fact_id.to_string())
    .bind(episode_id.to_string())
    .execute(conn)
    .await
    .map_err(db)?;
    Ok(())
}

pub(super) enum Settled {
    Done,
    Outranked { head: Uuid },
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn settle_supersede(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    new_id: Uuid,
    new_trust: Trust,
    new_valid_from: DateTime<Utc>,
    old_id: Uuid,
    reviewed: Option<Uuid>,
    closing_episode: Option<Uuid>,
) -> Result<Settled> {
    let mut current = old_id;
    for _ in 0..MAX_CHAIN_STEPS {
        let row = sqlx::query(
            "SELECT status, trust, invalidated_at, valid_from, superseded_by \
             FROM facts WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(current.to_string())
        .fetch_one(&mut *conn)
        .await
        .map_err(db)?;
        if let Some(next) = optional_uuid(&row, "superseded_by")? {
            current = next;
            continue;
        }
        if current == new_id {
            return Ok(Settled::Done);
        }
        let status: String = row.try_get("status").map_err(db)?;
        let invalidated = optional_time(&row, "invalidated_at")?.is_some();
        if status != FactStatus::Active.as_str() || invalidated {
            return Ok(Settled::Done);
        }
        let head_trust = trust(&row, "trust")?;
        if head_trust > new_trust && reviewed != Some(current) {
            return Ok(Settled::Outranked { head: current });
        }
        let head_valid_from = time(&row, "valid_from")?;
        let (closed, end, successor) = if new_valid_from < head_valid_from {
            let later = earliest_later_in_chain(conn, ctx, current, new_id, new_valid_from).await?;
            (new_id, later.1, later.0)
        } else {
            (current, new_valid_from, new_id)
        };
        sqlx::query(
            "UPDATE facts SET \
                 valid_to = CASE WHEN valid_to IS NULL OR ?3 < valid_to THEN ?3 ELSE valid_to END, \
                 superseded_by = ?4, closed_by_episode = ?5 WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(closed.to_string())
        .bind(rows::timestamp(end))
        .bind(successor.to_string())
        .bind(closing_episode.map(|id| id.to_string()))
        .execute(&mut *conn)
        .await
        .map_err(db)?;
        return Ok(Settled::Done);
    }
    Err(Error::Internal(format!(
        "supersession chain from fact {old_id} exceeds {MAX_CHAIN_STEPS} steps"
    )))
}

async fn earliest_later_in_chain(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    member: Uuid,
    new_id: Uuid,
    after: DateTime<Utc>,
) -> Result<(Uuid, DateTime<Utc>)> {
    let row = sqlx::query(
        "WITH RECURSIVE chain(id) AS ( \
             SELECT ?2 \
             UNION \
             SELECT n.id FROM chain c \
             JOIN facts cur ON cur.tenant = ?1 AND cur.id = c.id \
             JOIN facts n ON n.tenant = cur.tenant AND n.scope = cur.scope \
                 AND (n.superseded_by = cur.id OR n.id = cur.superseded_by)) \
         SELECT f.id, f.valid_from FROM facts f JOIN chain c ON c.id = f.id \
         WHERE f.tenant = ?1 AND f.id <> ?3 AND f.status = 'active' AND f.invalidated_at IS NULL \
             AND f.valid_from > ?4 ORDER BY f.valid_from, f.id LIMIT 1",
    )
    .bind(&ctx.tenant)
    .bind(member.to_string())
    .bind(new_id.to_string())
    .bind(rows::timestamp(after))
    .fetch_one(conn)
    .await
    .map_err(db)?;
    Ok((uuid(&row, "id")?, time(&row, "valid_from")?))
}

struct ChainExit {
    valid_from: DateTime<Utc>,
    valid_to: Option<DateTime<Utc>>,
    superseded_by: Option<Uuid>,
    closed_by_episode: Option<Uuid>,
}

fn surviving_successor(exits: &HashMap<Uuid, ChainExit>, start: Option<Uuid>) -> Option<Uuid> {
    let mut current = start;
    while let Some(exit) = current.and_then(|id| exits.get(&id)) {
        current = exit.superseded_by;
    }
    current
}

fn chain_target(
    exits: &HashMap<Uuid, ChainExit>,
    reopened: &HashMap<Uuid, Uuid>,
    id: Uuid,
) -> Option<Uuid> {
    let start = exits.get(&id).and_then(|exit| exit.superseded_by);
    surviving_successor(exits, start).or_else(|| reopened.get(&id).copied())
}

pub(super) struct PurgeOutcome {
    pub(super) report: PurgeReport,
    pub(super) facts_requarantined: u64,
}

pub(super) fn highest_trust(trusts: Vec<String>) -> Result<Option<Trust>> {
    let mut highest = None;
    for text in trusts {
        let trust: Trust = text
            .parse()
            .map_err(|_| Error::Storage("stored trust is invalid".into()))?;
        highest = highest.max(Some(trust));
    }
    Ok(highest)
}

pub(super) async fn purge_episode(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    id: Uuid,
) -> Result<PurgeOutcome> {
    let found = sqlx::query(
        "SELECT f.id, f.scope, f.status, f.trust, f.valid_from, f.valid_to, f.superseded_by, \
             f.closed_by_episode, f.subject_id, f.object_id, \
             (SELECT json_group_array(e.trust) FROM fact_sources o \
                 JOIN episodes e ON e.tenant = o.tenant AND e.id = o.episode_id \
                 WHERE o.tenant = f.tenant AND o.fact_id = f.id AND o.episode_id <> ?2\
             ) AS remaining_trusts \
         FROM facts f JOIN fact_sources s ON s.tenant = f.tenant AND s.fact_id = f.id \
         WHERE f.tenant = ?1 AND s.episode_id = ?2",
    )
    .bind(&ctx.tenant)
    .bind(id.to_string())
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;

    let mut exits: HashMap<Uuid, ChainExit> = HashMap::new();
    let mut doomed_ids = Vec::new();
    let mut requarantined = Vec::new();
    let mut retrusted = HashMap::new();
    let mut lowered = HashMap::new();
    let mut touched_entities = BTreeSet::new();
    for row in &found {
        let fact_id = uuid(row, "id")?;
        let exit = ChainExit {
            valid_from: time(row, "valid_from")?,
            valid_to: optional_time(row, "valid_to")?,
            superseded_by: optional_uuid(row, "superseded_by")?,
            closed_by_episode: optional_uuid(row, "closed_by_episode")?,
        };
        let remaining: String = row.try_get("remaining_trusts").map_err(db)?;
        let remaining: Vec<String> = serde_json::from_str(&remaining)
            .map_err(|e| Error::Storage(format!("invalid remaining trusts JSON: {e}")))?;
        let Some(highest) = highest_trust(remaining)? else {
            touched_entities.insert(uuid(row, "subject_id")?);
            if let Some(object_id) = optional_uuid(row, "object_id")? {
                touched_entities.insert(object_id);
            }
            doomed_ids.push(fact_id);
            exits.insert(fact_id, exit);
            continue;
        };
        let current = trust(row, "trust")?;
        let trust = highest.min(current);
        let scope: Scope = row
            .try_get::<String, _>("scope")
            .map_err(db)?
            .parse()
            .map_err(|_| Error::Storage("fact has an invalid scope".into()))?;
        let status: String = row.try_get("status").map_err(db)?;
        let active = status == FactStatus::Active.as_str();
        if active && source_requires_quarantine(&scope, trust) {
            requarantined.push((fact_id, trust));
            exits.insert(fact_id, exit);
        } else {
            if active && trust < current {
                lowered.insert(fact_id, exit);
            }
            retrusted.insert(fact_id, trust);
        }
    }

    let reclosed = closed_by(conn, ctx, id).await?;
    let mut closed = closed_facts(conn, ctx, &found).await?;
    for facts in closed.values_mut() {
        facts.retain(|fact| !reclosed.contains_key(&fact.id));
    }
    loop {
        let outranked: Vec<Uuid> = lowered
            .keys()
            .copied()
            .filter(|id| closes_higher_trust(&closed, &exits, &retrusted, *id))
            .collect();
        if outranked.is_empty() {
            break;
        }
        for fact_id in outranked {
            if let (Some(trust), Some(exit)) =
                (retrusted.remove(&fact_id), lowered.remove(&fact_id))
            {
                requarantined.push((fact_id, trust));
                exits.insert(fact_id, exit);
            }
        }
    }

    reopen_closed_by(conn, ctx, &reclosed, &mut exits).await?;
    let reopened = restore_predecessors(conn, ctx, &exits).await?;

    for exit_id in exits.keys() {
        if let Some(target) = chain_target(&exits, &reopened, *exit_id) {
            sqlx::query(
                "UPDATE facts SET pending_supersede_of = ?3 \
                 WHERE tenant = ?1 AND pending_supersede_of = ?2",
            )
            .bind(&ctx.tenant)
            .bind(exit_id.to_string())
            .bind(target.to_string())
            .execute(&mut *conn)
            .await
            .map_err(db)?;
        }
    }

    for (fact_id, trust) in &retrusted {
        sqlx::query("UPDATE facts SET trust = ?3 WHERE tenant = ?1 AND id = ?2")
            .bind(&ctx.tenant)
            .bind(fact_id.to_string())
            .bind(trust.as_str())
            .execute(&mut *conn)
            .await
            .map_err(db)?;
    }
    for (fact_id, trust) in &requarantined {
        let pending = chain_target(&exits, &reopened, *fact_id);
        sqlx::query(
            "UPDATE facts SET status = 'quarantined', trust = ?3, pending_supersede_of = ?4, \
                 valid_to = CASE WHEN superseded_by IS NULL THEN valid_to ELSE NULL END, \
                 superseded_by = NULL, closed_by_episode = NULL WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(fact_id.to_string())
        .bind(trust.as_str())
        .bind(pending.map(|id| id.to_string()))
        .execute(&mut *conn)
        .await
        .map_err(db)?;
    }

    remove_aliases_only_from(conn, ctx, id).await?;

    let doomed = json_array(&uuid_strings(doomed_ids))?;
    sqlx::query(
        "DELETE FROM fact_embeddings WHERE rowid IN (\
             SELECT rowid FROM facts WHERE tenant = ?1 AND id IN (SELECT value FROM json_each(?2)))",
    )
    .bind(&ctx.tenant)
    .bind(&doomed)
    .execute(&mut *conn)
    .await
    .map_err(db)?;
    let facts_deleted = sqlx::query(
        "DELETE FROM facts WHERE tenant = ?1 AND id IN (SELECT value FROM json_each(?2))",
    )
    .bind(&ctx.tenant)
    .bind(doomed)
    .execute(&mut *conn)
    .await
    .map_err(db)?
    .rows_affected();
    let episodes_deleted = sqlx::query("DELETE FROM episodes WHERE tenant = ?1 AND id = ?2")
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .execute(&mut *conn)
        .await
        .map_err(db)?
        .rows_affected();
    let entities_deleted = delete_orphan_entities(conn, ctx, touched_entities).await?;

    Ok(PurgeOutcome {
        report: PurgeReport {
            facts_deleted,
            episodes_deleted,
            entities_deleted,
        },
        facts_requarantined: requarantined.len() as u64,
    })
}

async fn closed_by(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    episode_id: Uuid,
) -> Result<HashMap<Uuid, Option<DateTime<Utc>>>> {
    let found = sqlx::query(
        "SELECT f.id, CASE WHEN f.valid_to < s.valid_from THEN f.valid_to ELSE NULL END AS valid_to \
         FROM facts f JOIN facts s ON s.tenant = f.tenant AND s.id = f.superseded_by \
         WHERE f.tenant = ?1 AND f.closed_by_episode = ?2 AND f.invalidated_at IS NULL \
         ORDER BY f.id",
    )
    .bind(&ctx.tenant)
    .bind(episode_id.to_string())
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;
    found
        .iter()
        .map(|row| Ok((uuid(row, "id")?, optional_time(row, "valid_to")?)))
        .collect()
}

async fn reopen_closed_by(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    reclosed: &HashMap<Uuid, Option<DateTime<Utc>>>,
    exits: &mut HashMap<Uuid, ChainExit>,
) -> Result<()> {
    for (fact_id, valid_to) in reclosed {
        sqlx::query(
            "UPDATE facts SET valid_to = ?3, superseded_by = NULL, closed_by_episode = NULL \
             WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(fact_id.to_string())
        .bind(valid_to.map(rows::timestamp))
        .execute(&mut *conn)
        .await
        .map_err(db)?;
        if let Some(exit) = exits.get_mut(fact_id) {
            exit.valid_to = *valid_to;
            exit.superseded_by = None;
            exit.closed_by_episode = None;
        }
    }
    Ok(())
}

pub(super) async fn remove_aliases_only_from(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    episode_id: Uuid,
) -> Result<()> {
    let earned = sqlx::query(
        "SELECT entity_id, alias_key, alias FROM entity_alias_sources \
         WHERE tenant = ?1 AND episode_id = ?2",
    )
    .bind(&ctx.tenant)
    .bind(episode_id.to_string())
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;
    sqlx::query("DELETE FROM entity_alias_sources WHERE tenant = ?1 AND episode_id = ?2")
        .bind(&ctx.tenant)
        .bind(episode_id.to_string())
        .execute(&mut *conn)
        .await
        .map_err(db)?;
    for row in &earned {
        let entity_id = uuid(row, "entity_id")?;
        let alias_key: String = row.try_get("alias_key").map_err(db)?;
        let alias: String = row.try_get("alias").map_err(db)?;
        let remains = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM entity_alias_sources s \
                 WHERE s.tenant = ?1 AND s.entity_id = ?2 AND s.alias = ?3) AS same_alias, \
                EXISTS(SELECT 1 FROM entity_alias_sources s \
                 WHERE s.tenant = ?1 AND s.entity_id = ?2 AND s.alias_key = ?4) AS same_key",
        )
        .bind(&ctx.tenant)
        .bind(entity_id.to_string())
        .bind(&alias)
        .bind(&alias_key)
        .fetch_one(&mut *conn)
        .await
        .map_err(db)?;
        let same_alias: bool = remains.try_get("same_alias").map_err(db)?;
        let same_key: bool = remains.try_get("same_key").map_err(db)?;
        let current =
            sqlx::query("SELECT aliases, alias_keys FROM entities WHERE tenant = ?1 AND id = ?2")
                .bind(&ctx.tenant)
                .bind(entity_id.to_string())
                .fetch_one(&mut *conn)
                .await
                .map_err(db)?;
        if !same_alias || !same_key {
            let mut aliases: Vec<String> = decode_json(&current, "aliases")?;
            let mut alias_keys: Vec<String> = decode_json(&current, "alias_keys")?;
            if !same_alias {
                aliases.retain(|value| value != &alias);
            }
            if !same_key {
                alias_keys.retain(|value| value != &alias_key);
            }
            sqlx::query(
                "UPDATE entities SET aliases = ?3, alias_keys = ?4 WHERE tenant = ?1 AND id = ?2",
            )
            .bind(&ctx.tenant)
            .bind(entity_id.to_string())
            .bind(json_array(&aliases)?)
            .bind(json_array(&alias_keys)?)
            .execute(&mut *conn)
            .await
            .map_err(db)?;
        }
    }
    Ok(())
}

struct ClosedFact {
    id: Uuid,
    trust: Trust,
    live: bool,
}

async fn closed_facts(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    rows_: &[SqliteRow],
) -> Result<HashMap<Uuid, Vec<ClosedFact>>> {
    let ids = rows_
        .iter()
        .map(|row| uuid(row, "id"))
        .collect::<Result<Vec<Uuid>>>()?;
    let found = sqlx::query(
        "SELECT id, trust, superseded_by, \
             (status = 'active' AND invalidated_at IS NULL) AS live FROM facts \
         WHERE tenant = ?1 AND superseded_by IN (SELECT value FROM json_each(?2))",
    )
    .bind(&ctx.tenant)
    .bind(json_array(&uuid_strings(ids))?)
    .fetch_all(conn)
    .await
    .map_err(db)?;
    let mut closed: HashMap<Uuid, Vec<ClosedFact>> = HashMap::new();
    for row in &found {
        closed
            .entry(
                optional_uuid(row, "superseded_by")?
                    .ok_or_else(|| Error::Storage("closed fact has no superseding fact".into()))?,
            )
            .or_default()
            .push(ClosedFact {
                id: uuid(row, "id")?,
                trust: trust(row, "trust")?,
                live: row.try_get("live").map_err(db)?,
            });
    }
    Ok(closed)
}

fn closes_higher_trust(
    closed: &HashMap<Uuid, Vec<ClosedFact>>,
    exits: &HashMap<Uuid, ChainExit>,
    retrusted: &HashMap<Uuid, Trust>,
    id: Uuid,
) -> bool {
    let Some(&trust) = retrusted.get(&id) else {
        return false;
    };
    let mut stack = vec![id];
    while let Some(closer) = stack.pop() {
        for fact in closed.get(&closer).into_iter().flatten() {
            if exits.contains_key(&fact.id) {
                stack.push(fact.id);
            } else if fact.live && retrusted.get(&fact.id).copied().unwrap_or(fact.trust) > trust {
                return true;
            }
        }
    }
    false
}

async fn restore_predecessors(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    exits: &HashMap<Uuid, ChainExit>,
) -> Result<HashMap<Uuid, Uuid>> {
    let exit_ids = json_array(&uuid_strings(exits.keys().copied()))?;
    let predecessors = sqlx::query(
        "SELECT id, superseded_by FROM facts WHERE tenant = ?1 \
             AND superseded_by IN (SELECT value FROM json_each(?2)) \
             AND id NOT IN (SELECT value FROM json_each(?2)) AND invalidated_at IS NULL \
         ORDER BY valid_from, id",
    )
    .bind(&ctx.tenant)
    .bind(exit_ids)
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;

    let mut reopened = HashMap::new();
    for row in &predecessors {
        let id = uuid(row, "id")?;
        let direct = optional_uuid(row, "superseded_by")?
            .ok_or_else(|| Error::Storage("predecessor has no successor".into()))?;
        let Some(first) = exits.get(&direct) else {
            continue;
        };
        reopened.insert(direct, id);
        let mut last = first;
        while let Some((next_id, next)) = last.superseded_by.and_then(|s| exits.get_key_value(&s)) {
            reopened.insert(*next_id, id);
            last = next;
        }
        sqlx::query(
            "UPDATE facts SET \
                 valid_to = CASE WHEN valid_to < ?3 THEN valid_to ELSE ?4 END, \
                 superseded_by = ?5, closed_by_episode = ?6 WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .bind(rows::timestamp(first.valid_from))
        .bind(last.valid_to.map(rows::timestamp))
        .bind(last.superseded_by.map(|id| id.to_string()))
        .bind(last.closed_by_episode.map(|id| id.to_string()))
        .execute(&mut *conn)
        .await
        .map_err(db)?;
    }
    Ok(reopened)
}

pub(super) async fn leave_chain_on_retraction(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    id: Uuid,
) -> Result<()> {
    let row = sqlx::query(
        "SELECT valid_from, valid_to, superseded_by, closed_by_episode FROM facts \
         WHERE tenant = ?1 AND id = ?2",
    )
    .bind(&ctx.tenant)
    .bind(id.to_string())
    .fetch_one(&mut *conn)
    .await
    .map_err(db)?;
    let exit = ChainExit {
        valid_from: time(&row, "valid_from")?,
        valid_to: optional_time(&row, "valid_to")?,
        superseded_by: optional_uuid(&row, "superseded_by")?,
        closed_by_episode: optional_uuid(&row, "closed_by_episode")?,
    };
    let exits = HashMap::from([(id, exit)]);
    let reopened = restore_predecessors(conn, ctx, &exits).await?;
    if let Some(target) = chain_target(&exits, &reopened, id) {
        sqlx::query(
            "UPDATE facts SET pending_supersede_of = ?3 \
             WHERE tenant = ?1 AND pending_supersede_of = ?2",
        )
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .bind(target.to_string())
        .execute(conn)
        .await
        .map_err(db)?;
    }
    Ok(())
}

async fn delete_orphan_entities(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    candidates: BTreeSet<Uuid>,
) -> Result<u64> {
    let candidates = json_array(&uuid_strings(candidates))?;
    let doomed: Vec<String> = sqlx::query_scalar(
        "SELECT e.id FROM entities e WHERE e.tenant = ?1 \
             AND e.id IN (SELECT value FROM json_each(?2)) AND NOT EXISTS (\
             SELECT 1 FROM facts f WHERE f.tenant = e.tenant \
                 AND (f.subject_id = e.id OR f.object_id = e.id))",
    )
    .bind(&ctx.tenant)
    .bind(&candidates)
    .fetch_all(&mut *conn)
    .await
    .map_err(db)?;
    if doomed.is_empty() {
        return Ok(0);
    }
    let doomed = json_array(&doomed)?;
    sqlx::query(
        "DELETE FROM entity_embeddings WHERE rowid IN (\
             SELECT rowid FROM entities WHERE tenant = ?1 \
             AND id IN (SELECT value FROM json_each(?2)))",
    )
    .bind(&ctx.tenant)
    .bind(&doomed)
    .execute(&mut *conn)
    .await
    .map_err(db)?;
    let deleted = sqlx::query(
        "DELETE FROM entities WHERE tenant = ?1 AND id IN (SELECT value FROM json_each(?2))",
    )
    .bind(&ctx.tenant)
    .bind(doomed)
    .execute(conn)
    .await
    .map_err(db)?
    .rows_affected();
    Ok(deleted)
}

pub(super) async fn purge_scope(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    scope: &Scope,
) -> Result<PurgeReport> {
    let scope_text = scope.to_string();
    let episode_ids: Vec<String> =
        sqlx::query_scalar("SELECT id FROM episodes WHERE tenant = ?1 AND scope = ?2 ORDER BY id")
            .bind(&ctx.tenant)
            .bind(&scope_text)
            .fetch_all(&mut *conn)
            .await
            .map_err(db)?;
    for episode_id in episode_ids {
        let episode_id = Uuid::parse_str(&episode_id)
            .map_err(|e| Error::Storage(format!("invalid episode UUID: {e}")))?;
        remove_aliases_only_from(conn, ctx, episode_id).await?;
    }
    sqlx::query(
        "DELETE FROM fact_embeddings WHERE rowid IN (\
             SELECT rowid FROM facts WHERE tenant = ?1 AND scope = ?2)",
    )
    .bind(&ctx.tenant)
    .bind(&scope_text)
    .execute(&mut *conn)
    .await
    .map_err(db)?;
    let facts_deleted = sqlx::query("DELETE FROM facts WHERE tenant = ?1 AND scope = ?2")
        .bind(&ctx.tenant)
        .bind(&scope_text)
        .execute(&mut *conn)
        .await
        .map_err(db)?
        .rows_affected();
    let episodes_deleted = sqlx::query("DELETE FROM episodes WHERE tenant = ?1 AND scope = ?2")
        .bind(&ctx.tenant)
        .bind(&scope_text)
        .execute(&mut *conn)
        .await
        .map_err(db)?
        .rows_affected();
    sqlx::query(
        "DELETE FROM entity_embeddings WHERE rowid IN (\
             SELECT rowid FROM entities WHERE tenant = ?1 AND scope = ?2)",
    )
    .bind(&ctx.tenant)
    .bind(&scope_text)
    .execute(&mut *conn)
    .await
    .map_err(db)?;
    let entities_deleted = sqlx::query("DELETE FROM entities WHERE tenant = ?1 AND scope = ?2")
        .bind(&ctx.tenant)
        .bind(&scope_text)
        .execute(&mut *conn)
        .await
        .map_err(db)?
        .rows_affected();
    if let Scope::Agent(agent_id) = scope {
        sqlx::query("DELETE FROM blocks WHERE tenant = ?1 AND agent_id = ?2")
            .bind(&ctx.tenant)
            .bind(agent_id)
            .execute(&mut *conn)
            .await
            .map_err(db)?;
    }
    Ok(PurgeReport {
        facts_deleted,
        episodes_deleted,
        entities_deleted,
    })
}

pub(super) fn purge_detail(report: &PurgeReport) -> Value {
    json!({
        "facts_deleted": report.facts_deleted,
        "episodes_deleted": report.episodes_deleted,
        "entities_deleted": report.entities_deleted,
    })
}

fn uuid(row: &SqliteRow, column: &str) -> Result<Uuid> {
    let value: String = row.try_get(column).map_err(db)?;
    Uuid::parse_str(&value)
        .map_err(|_| Error::Storage(format!("column {column} holds an invalid UUID")))
}

fn optional_uuid(row: &SqliteRow, column: &str) -> Result<Option<Uuid>> {
    let value: Option<String> = row.try_get(column).map_err(db)?;
    value
        .map(|value| {
            Uuid::parse_str(&value)
                .map_err(|_| Error::Storage(format!("column {column} holds an invalid UUID")))
        })
        .transpose()
}

fn time(row: &SqliteRow, column: &str) -> Result<DateTime<Utc>> {
    let value: i64 = row.try_get(column).map_err(db)?;
    DateTime::from_timestamp_micros(value)
        .ok_or_else(|| Error::Storage(format!("column {column} holds an invalid timestamp")))
}

fn optional_time(row: &SqliteRow, column: &str) -> Result<Option<DateTime<Utc>>> {
    let value: Option<i64> = row.try_get(column).map_err(db)?;
    value
        .map(|value| {
            DateTime::from_timestamp_micros(value).ok_or_else(|| {
                Error::Storage(format!("column {column} holds an invalid timestamp"))
            })
        })
        .transpose()
}

fn trust(row: &SqliteRow, column: &str) -> Result<Trust> {
    row.try_get::<String, _>(column)
        .map_err(db)?
        .parse()
        .map_err(|_| Error::Storage("stored trust is invalid".into()))
}

fn decode_json<T: serde::de::DeserializeOwned>(row: &SqliteRow, column: &str) -> Result<T> {
    let value: String = row.try_get(column).map_err(db)?;
    serde_json::from_str(&value)
        .map_err(|e| Error::Storage(format!("column {column} holds invalid JSON: {e}")))
}
