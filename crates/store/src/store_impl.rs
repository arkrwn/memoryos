//! `impl MemoryStore for PgStore`. SQL text is constant or built from constants; every value,
//! including all user text, is a bind parameter.

use std::collections::BTreeSet;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use memoryos_core::{
    normalize_name, ApiKey, Block, BlockVersion, Entity, Episode, Error, Fact, FactFilter, FactOp,
    FactStatus, GraphNeighborhood, Job, MemoryStore, NewEntity, NewEpisode, PurgeReport,
    RequestContext, Result, Scope, ScoredEpisode, ScoredFact, Session, SessionMessage, Trust,
};
use pgvector::Vector;
use serde_json::{json, Value};
use sqlx::{PgConnection, Row};
use uuid::Uuid;

use crate::rows::{
    self, visible, API_KEY_COLS, BLOCK_COLS, ENTITY_COLS, EPISODE_COLS, FACT_COLS, MESSAGE_COLS,
    SESSION_COLS,
};
use crate::{db, facts, validate, PgStore};

/// Hard cap on `neighbors`, so a depth-5 walk through a hub cannot load an unbounded graph, whatever
/// limit the caller asks for.
const MAX_GRAPH_EDGES: i64 = 5000;

/// Applies pgvector's iterative scan for this transaction, so rows removed by scope and time
/// filters do not shrink the result below `limit`.
async fn tune_vector_scan(conn: &mut PgConnection, limit: i64) -> Result<()> {
    let ef_search = (limit * 4).clamp(200, 1000).to_string();
    sqlx::query(
        "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true), \
                set_config('hnsw.ef_search', $1, true), \
                set_config('hnsw.max_scan_tuples', '100000', true)",
    )
    .bind(ef_search)
    .execute(conn)
    .await
    .map_err(db)?;
    Ok(())
}

async fn load_fact(conn: &mut PgConnection, id: Uuid) -> Result<Fact> {
    let sql = format!("SELECT {FACT_COLS} FROM facts f WHERE f.id = $1");
    let row = sqlx::query(&sql)
        .bind(id)
        .fetch_optional(conn)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("fact {id}")))?;
    rows::fact(&row)
}

/// Locks a fact and returns its state for the quarantine transitions.
struct QuarantineState {
    status: String,
    trust: Trust,
    invalidated: bool,
    valid_from: DateTime<Utc>,
    pending_supersede_of: Option<Uuid>,
}

async fn lock_quarantined(conn: &mut PgConnection, id: Uuid) -> Result<QuarantineState> {
    let row = sqlx::query(
        "SELECT status, trust, invalidated_at IS NOT NULL AS invalidated, valid_from, \
         pending_supersede_of FROM facts WHERE id = $1 FOR UPDATE",
    )
    .bind(id)
    .fetch_optional(conn)
    .await
    .map_err(db)?
    .ok_or_else(|| Error::NotFound(format!("fact {id}")))?;
    let state = QuarantineState {
        status: row.try_get("status").map_err(db)?,
        trust: row
            .try_get::<String, _>("trust")
            .map_err(db)?
            .parse()
            .map_err(|_| Error::Storage("fact has an invalid trust".into()))?,
        invalidated: row.try_get("invalidated").map_err(db)?,
        valid_from: row.try_get("valid_from").map_err(db)?,
        pending_supersede_of: row.try_get("pending_supersede_of").map_err(db)?,
    };
    if state.status != FactStatus::Quarantined.as_str() {
        return Err(Error::InvalidInput(format!(
            "fact {id} is {}, not quarantined",
            state.status
        )));
    }
    Ok(state)
}

async fn require_session(conn: &mut PgConnection, id: Uuid) -> Result<i64> {
    let row = sqlx::query("SELECT compacted_through FROM sessions WHERE id = $1")
        .bind(id)
        .fetch_optional(conn)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("session {id}")))?;
    row.try_get("compacted_through").map_err(db)
}

impl PgStore {
    async fn insert_episode_inner(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
        enqueue: bool,
    ) -> Result<Episode> {
        validate::non_empty("source", &episode.source)?;
        validate::non_empty("content", &episode.content)?;
        let metadata = match episode.metadata {
            Value::Null => json!({}),
            other => other,
        };
        let mut tx = self.tenant_tx(ctx).await?;
        let id = Uuid::now_v7();
        let sql = format!(
            "INSERT INTO episodes (id, tenant, scope, source, trust, content, metadata, \
             occurred_at, idempotency_key) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8::timestamptz, now()), $9) \
             ON CONFLICT (tenant, idempotency_key) DO NOTHING RETURNING {EPISODE_COLS}"
        );
        let inserted = sqlx::query(&sql)
            .bind(id)
            .bind(&ctx.tenant)
            .bind(episode.scope.to_string())
            .bind(&episode.source)
            .bind(episode.trust.as_str())
            .bind(&episode.content)
            .bind(metadata)
            .bind(episode.occurred_at)
            .bind(&episode.idempotency_key)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?;

        let row = match inserted {
            Some(row) => {
                if enqueue {
                    sqlx::query("SELECT enqueue_job($1, $2, 'extract', $3::jsonb)")
                        .bind(Uuid::now_v7())
                        .bind(&ctx.tenant)
                        .bind(json!({ "episode_id": id.to_string() }))
                        .execute(&mut *tx)
                        .await
                        .map_err(db)?;
                }
                row
            }
            // The conflicting insert has committed by now, so this statement's snapshot sees it.
            None => {
                let sql = format!(
                    "SELECT {EPISODE_COLS} FROM episodes WHERE tenant = $1 AND idempotency_key = $2"
                );
                sqlx::query(&sql)
                    .bind(&ctx.tenant)
                    .bind(&episode.idempotency_key)
                    .fetch_one(&mut *tx)
                    .await
                    .map_err(db)?
            }
        };
        let result = rows::episode(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }
}

#[async_trait]
impl MemoryStore for PgStore {
    async fn insert_episode(&self, ctx: &RequestContext, episode: NewEpisode) -> Result<Episode> {
        self.insert_episode_inner(ctx, episode, true).await
    }

    async fn insert_unqueued_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> Result<Episode> {
        self.insert_episode_inner(ctx, episode, false).await
    }

    async fn get_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<Episode> {
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!("SELECT {EPISODE_COLS} FROM episodes WHERE id = $1");
        let row = sqlx::query(&sql)
            .bind(id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("episode {id}")))?;
        let result = rows::episode(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn purge_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<PurgeReport> {
        let mut tx = self.tenant_tx(ctx).await?;
        sqlx::query("SELECT id FROM episodes WHERE id = $1 FOR UPDATE")
            .bind(id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("episode {id}")))?;
        let outcome = facts::purge_episode(&mut tx, id).await?;
        let mut detail = facts::purge_detail(&outcome.report);
        detail["facts_requarantined"] = json!(outcome.facts_requarantined);
        facts::audit(&mut tx, ctx, "episode.purge", Some(id), detail).await?;
        tx.commit().await.map_err(db)?;
        Ok(outcome.report)
    }

    async fn purge_scope(&self, ctx: &RequestContext, scope: &Scope) -> Result<PurgeReport> {
        let mut tx = self.tenant_tx(ctx).await?;
        let report = facts::purge_scope(&mut tx, scope).await?;
        let mut detail = facts::purge_detail(&report);
        detail["scope"] = json!(scope.to_string());
        facts::audit(&mut tx, ctx, "scope.purge", None, detail).await?;
        tx.commit().await.map_err(db)?;
        Ok(report)
    }

    async fn apply_extraction(
        &self,
        ctx: &RequestContext,
        episode_id: Uuid,
        ops: Vec<FactOp>,
    ) -> Result<Vec<Uuid>> {
        let mut tx = self.tenant_tx(ctx).await?;
        // The row lock serializes concurrent or retried jobs for one episode. The loser waits
        // here, then reads the winner's processed_at and does nothing.
        let row = sqlx::query(
            "SELECT scope, trust, processed_at IS NOT NULL AS processed \
             FROM episodes WHERE id = $1 FOR UPDATE",
        )
        .bind(episode_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("episode {episode_id}")))?;
        if row.try_get::<bool, _>("processed").map_err(db)? {
            return Ok(Vec::new());
        }
        let scope: Scope = row
            .try_get::<String, _>("scope")
            .map_err(db)?
            .parse()
            .map_err(|_| Error::Storage("episode has an invalid scope".into()))?;
        let trust: Trust = row
            .try_get::<String, _>("trust")
            .map_err(db)?
            .parse()
            .map_err(|_| Error::Storage("episode has an invalid trust".into()))?;

        let episode = facts::LockedEpisode {
            id: episode_id,
            scope,
            trust,
        };
        let inserted = facts::apply_ops(&mut tx, ctx, &episode, &ops).await?;

        sqlx::query("UPDATE episodes SET processed_at = now() WHERE id = $1")
            .bind(episode_id)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        tx.commit().await.map_err(db)?;
        Ok(inserted)
    }

    async fn upsert_entity(&self, ctx: &RequestContext, entity: NewEntity) -> Result<Entity> {
        validate::non_empty("kind", &entity.kind)?;
        validate::non_empty("name", &entity.name)?;
        let embedding = validate::embedding(&entity.embedding)?;
        let normalized = normalize_name(&entity.name);
        let alias_keys: Vec<String> = entity.aliases.iter().map(|a| normalize_name(a)).collect();
        let scope = entity.scope.to_string();
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!(
            "INSERT INTO entities (id, tenant, scope, kind, name, normalized_name, aliases, \
             embedding, alias_keys) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
             ON CONFLICT (tenant, scope, kind, normalized_name) DO NOTHING \
             RETURNING {ENTITY_COLS}"
        );
        let inserted = sqlx::query(&sql)
            .bind(Uuid::now_v7())
            .bind(&ctx.tenant)
            .bind(&scope)
            .bind(&entity.kind)
            .bind(&entity.name)
            .bind(&normalized)
            .bind(&entity.aliases)
            .bind(embedding)
            .bind(&alias_keys)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?;
        let row = match inserted {
            Some(row) => row,
            None => {
                let sql = format!(
                    "SELECT {ENTITY_COLS} FROM entities \
                     WHERE scope = $1 AND kind = $2 AND normalized_name = $3"
                );
                sqlx::query(&sql)
                    .bind(&scope)
                    .bind(&entity.kind)
                    .bind(&normalized)
                    .fetch_one(&mut *tx)
                    .await
                    .map_err(db)?
            }
        };
        let result = rows::entity(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn get_entity(&self, ctx: &RequestContext, id: Uuid) -> Result<Entity> {
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!("SELECT {ENTITY_COLS} FROM entities WHERE id = $1");
        let row = sqlx::query(&sql)
            .bind(id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("entity {id}")))?;
        let result = rows::entity(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn similar_entities(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        embedding: &[f32],
        limit: i64,
    ) -> Result<Vec<(Entity, f64)>> {
        let limit = validate::limit(limit)?;
        let query: Vector = validate::embedding(embedding)?;
        let mut tx = self.tenant_tx(ctx).await?;
        tune_vector_scan(&mut tx, limit).await?;
        let sql = format!(
            "WITH r AS MATERIALIZED ( \
                 SELECT {ENTITY_COLS}, embedding <=> $1 AS distance FROM entities \
                 WHERE scope = $2 ORDER BY distance LIMIT $3) \
             SELECT * FROM r ORDER BY distance + 0, id"
        );
        let found = sqlx::query(&sql)
            .bind(query)
            .bind(scope.to_string())
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let mut result = Vec::with_capacity(found.len());
        for row in &found {
            let distance: f64 = row.try_get("distance").map_err(db)?;
            // Cosine similarity spans -1..=1 and rounding can pass 1; the trait promises 0..=1.
            result.push((rows::entity(row)?, (1.0 - distance).clamp(0.0, 1.0)));
        }
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn add_entity_alias(&self, ctx: &RequestContext, id: Uuid, alias: &str) -> Result<()> {
        validate::non_empty("alias", alias)?;
        let mut tx = self.tenant_tx(ctx).await?;
        let updated = sqlx::query(
            "UPDATE entities SET \
                 aliases = CASE WHEN $2 = ANY(aliases) THEN aliases \
                     ELSE array_append(aliases, $2) END, \
                 alias_keys = CASE WHEN $3 = ANY(alias_keys) THEN alias_keys \
                     ELSE array_append(alias_keys, $3) END \
             WHERE id = $1",
        )
        .bind(id)
        .bind(alias)
        .bind(normalize_name(alias))
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        if updated.rows_affected() == 0 {
            return Err(Error::NotFound(format!("entity {id}")));
        }
        tx.commit().await.map_err(db)
    }

    async fn find_entity(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        kind: &str,
        name: &str,
    ) -> Result<Option<Entity>> {
        validate::non_empty("kind", kind)?;
        validate::non_empty("name", name)?;
        let normalized = normalize_name(name);
        let scope = scope.to_string();
        let mut tx = self.tenant_tx(ctx).await?;
        // Two statements so that a name hit stays a unique-index lookup. Oldest wins if two entities
        // ever claim one alias, which keeps the answer stable.
        let by_name = format!(
            "SELECT {ENTITY_COLS} FROM entities \
             WHERE scope = $1 AND kind = $2 AND normalized_name = $3"
        );
        let by_alias = format!(
            "SELECT {ENTITY_COLS} FROM entities \
             WHERE scope = $1 AND kind = $2 AND $3 = ANY(alias_keys) ORDER BY id LIMIT 1"
        );
        let mut found = None;
        for sql in [&by_name, &by_alias] {
            found = sqlx::query(sql)
                .bind(&scope)
                .bind(kind)
                .bind(&normalized)
                .fetch_optional(&mut *tx)
                .await
                .map_err(db)?;
            if found.is_some() {
                break;
            }
        }
        let result = found.as_ref().map(rows::entity).transpose()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn get_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        let mut tx = self.tenant_tx(ctx).await?;
        let result = load_fact(&mut tx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn facts_for_episode(&self, ctx: &RequestContext, episode_id: Uuid) -> Result<Vec<Fact>> {
        let mut tx = self.tenant_tx(ctx).await?;
        sqlx::query("SELECT id FROM episodes WHERE id = $1")
            .bind(episode_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("episode {episode_id}")))?;
        // Ids are UUIDv7 from one generator per process, so id order is creation order.
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.id IN (SELECT l.fact_id FROM fact_sources l WHERE l.episode_id = $1) \
             ORDER BY f.id"
        );
        let found = sqlx::query(&sql)
            .bind(episode_id)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn fact_history(&self, ctx: &RequestContext, id: Uuid) -> Result<Vec<Fact>> {
        let mut tx = self.tenant_tx(ctx).await?;
        // UNION drops rows already seen, so the walk ends even on cyclic data. Backdated facts
        // join a chain from the side, hence the walk in both directions.
        let sql = format!(
            "WITH RECURSIVE chain(id) AS ( \
                 SELECT id FROM facts WHERE id = $1 \
                 UNION \
                 SELECT n.id FROM chain c \
                 JOIN facts cur ON cur.id = c.id \
                 JOIN facts n ON n.scope = cur.scope \
                     AND (n.superseded_by = cur.id OR n.id = cur.superseded_by)) \
             SELECT {FACT_COLS} FROM facts f JOIN chain c ON c.id = f.id \
             ORDER BY f.valid_from, f.recorded_at, f.id"
        );
        let found = sqlx::query(&sql)
            .bind(id)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        if found.is_empty() {
            return Err(Error::NotFound(format!("fact {id}")));
        }
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn invalidate_fact(&self, ctx: &RequestContext, id: Uuid, reason: &str) -> Result<Fact> {
        let mut tx = self.tenant_tx(ctx).await?;
        // COALESCE keeps the first retraction time, which known_at reads depend on.
        let updated = sqlx::query(
            "UPDATE facts SET invalidated_at = COALESCE(invalidated_at, now()) WHERE id = $1",
        )
        .bind(id)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        if updated.rows_affected() == 0 {
            return Err(Error::NotFound(format!("fact {id}")));
        }
        facts::audit(
            &mut tx,
            ctx,
            "fact.invalidate",
            Some(id),
            json!({ "reason": reason }),
        )
        .await?;
        let result = load_fact(&mut tx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn current_facts_for_subject(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        subject_id: Uuid,
    ) -> Result<Vec<Fact>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let filter = visible(3, 4);
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.scope = $1 AND f.subject_id = $2 AND {filter} \
             ORDER BY f.valid_from, f.id"
        );
        let found = sqlx::query(&sql)
            .bind(scope.to_string())
            .bind(subject_id)
            .bind(None::<DateTime<Utc>>)
            .bind(None::<DateTime<Utc>>)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn search_facts_vector(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        embedding: &[f32],
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        let limit = validate::limit(limit)?;
        let query: Vector = validate::embedding(embedding)?;
        if scopes.is_empty() {
            return Ok(Vec::new());
        }
        let mut tx = self.tenant_tx(ctx).await?;
        tune_vector_scan(&mut tx, limit).await?;
        // relaxed_order can return rows slightly out of distance order, so pgvector's documented
        // pattern re-sorts the materialized result. `+ 0` keeps the planner from merging the sorts.
        let filter_sql = visible(3, 4);
        let sql = format!(
            "WITH r AS MATERIALIZED ( \
                 SELECT f.id AS rid, f.embedding <=> $1 AS distance FROM facts f \
                 WHERE f.scope = ANY($2) AND {filter_sql} \
                 ORDER BY distance LIMIT $5) \
             SELECT {FACT_COLS}, 1 - r.distance AS similarity \
             FROM r JOIN facts f ON f.id = r.rid \
             ORDER BY r.distance + 0, f.id"
        );
        let found = sqlx::query(&sql)
            .bind(query)
            .bind(validate::scope_strings(scopes))
            .bind(filter.as_of)
            .bind(filter.known_at)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = scored_facts(&found, "similarity")?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn search_facts_text(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        let limit = validate::limit(limit)?;
        if scopes.is_empty() {
            return Ok(Vec::new());
        }
        let mut tx = self.tenant_tx(ctx).await?;
        // GIN is unusable under RLS, so the DEFINER function finds ids and rank for app.tenant.
        // The join loads the rows under RLS, in one statement and so on one snapshot. The
        // function treats NULL times as no match, which is why the defaults are applied here.
        let sql = format!(
            "SELECT {FACT_COLS}, s.hit_rank::float8 AS rank \
             FROM search_fact_ids_text($2, $1, COALESCE($3::timestamptz, now()), \
                 COALESCE($4::timestamptz, now()), $5) AS s(hit_id, hit_rank) \
             JOIN facts f ON f.id = s.hit_id \
             ORDER BY s.hit_rank DESC, f.id"
        );
        let found = sqlx::query(&sql)
            .bind(query)
            .bind(validate::scope_strings(scopes))
            .bind(filter.as_of)
            .bind(filter.known_at)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = scored_facts(&found, "rank")?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn search_pending_episodes(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        query: &str,
        limit: i64,
    ) -> Result<Vec<ScoredEpisode>> {
        let limit = validate::limit(limit)?;
        if scopes.is_empty() {
            return Ok(Vec::new());
        }
        let mut tx = self.tenant_tx(ctx).await?;
        // Same approach as search_facts_text. The function limits trust to user and system.
        let sql = format!(
            "SELECT {EPISODE_COLS}, s.hit_rank::float8 AS rank \
             FROM search_pending_episode_ids($2, $1, $3) AS s(hit_id, hit_rank) \
             JOIN episodes ON episodes.id = s.hit_id \
             ORDER BY s.hit_rank DESC, episodes.id"
        );
        let found = sqlx::query(&sql)
            .bind(query)
            .bind(validate::scope_strings(scopes))
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let mut result = Vec::with_capacity(found.len());
        for row in &found {
            result.push(ScoredEpisode {
                episode: rows::episode(row)?,
                score: row.try_get("rank").map_err(db)?,
            });
        }
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn list_quarantined(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        limit: i64,
    ) -> Result<Vec<Fact>> {
        let limit = validate::limit(limit)?;
        if scopes.is_empty() {
            return Ok(Vec::new());
        }
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.scope = ANY($1) AND f.status = 'quarantined' \
             ORDER BY f.recorded_at, f.id LIMIT $2"
        );
        let found = sqlx::query(&sql)
            .bind(validate::scope_strings(scopes))
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn approve_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        let mut tx = self.tenant_tx(ctx).await?;
        let state = lock_quarantined(&mut tx, id).await?;
        if state.invalidated {
            return Err(Error::InvalidInput(format!(
                "fact {id} was retracted and cannot be approved"
            )));
        }
        if let Some(old_id) = state.pending_supersede_of {
            let settled = facts::settle_supersede(
                &mut tx,
                id,
                state.trust,
                state.valid_from,
                old_id,
                Some(old_id),
            )
            .await?;
            // The review covered `old_id`, not this head, so it does not authorize closing it.
            if let facts::Settled::Outranked { head } = settled {
                sqlx::query("UPDATE facts SET pending_supersede_of = $2 WHERE id = $1")
                    .bind(id)
                    .bind(head)
                    .execute(&mut *tx)
                    .await
                    .map_err(db)?;
                facts::audit(
                    &mut tx,
                    ctx,
                    "fact.approve_held",
                    Some(id),
                    json!({ "reviewed": old_id, "head": head }),
                )
                .await?;
                tx.commit().await.map_err(db)?;
                return Err(Error::InvalidInput(format!(
                    "fact {id} now contradicts fact {head}, which has higher trust: review it again"
                )));
            }
        }
        sqlx::query(
            "UPDATE facts SET status = 'active', pending_supersede_of = NULL WHERE id = $1",
        )
        .bind(id)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        facts::audit(
            &mut tx,
            ctx,
            "fact.approve",
            Some(id),
            json!({ "from_status": state.status, "superseded": state.pending_supersede_of }),
        )
        .await?;
        let result = load_fact(&mut tx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn reject_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        let mut tx = self.tenant_tx(ctx).await?;
        let state = lock_quarantined(&mut tx, id).await?;
        sqlx::query(
            "UPDATE facts SET status = 'rejected', pending_supersede_of = NULL WHERE id = $1",
        )
        .bind(id)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        facts::audit(
            &mut tx,
            ctx,
            "fact.reject",
            Some(id),
            json!({ "from_status": state.status }),
        )
        .await?;
        let result = load_fact(&mut tx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
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
        if !(1..=5).contains(&depth) {
            return Err(Error::InvalidInput(format!(
                "depth must be in 1..=5, got {depth}"
            )));
        }
        if edge_limit < 1 {
            return Err(Error::InvalidInput(format!(
                "edge_limit must be positive, got {edge_limit}"
            )));
        }
        let edge_limit = edge_limit.min(MAX_GRAPH_EDGES);
        let scope_texts = validate::scope_strings(scopes);
        let mut tx = self.tenant_tx(ctx).await?;
        let root_row = sqlx::query("SELECT id FROM entities WHERE id = $1 AND scope = ANY($2)")
            .bind(root)
            .bind(&scope_texts)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?;
        if root_row.is_none() {
            return Err(Error::NotFound(format!("entity {root}")));
        }

        // Breadth-first. `seen` holds every visited entity, so a cycle adds nothing to the next
        // frontier and the walk stops. Facts without an object are included: they describe the
        // entity and retrieval uses them as graph-expanded candidates.
        //
        // Each frontier entity walks its own index newest first and stops at the limit, so a hub
        // costs the limit and not its edge count. Rows are loaded only for the ids that win.
        let filter_sql = visible(3, 4);
        let sql = format!(
            "WITH hits AS MATERIALIZED ( \
                 SELECT DISTINCT h.id FROM unnest($2::uuid[]) AS frontier(entity_id) \
                 CROSS JOIN LATERAL ( \
                     (SELECT f.id FROM facts f \
                      WHERE f.subject_id = frontier.entity_id AND f.scope = ANY($1) \
                          AND NOT f.id = ANY($6) AND {filter_sql} \
                      ORDER BY f.id DESC LIMIT $5) \
                     UNION ALL \
                     (SELECT f.id FROM facts f \
                      WHERE f.object_id = frontier.entity_id AND f.scope = ANY($1) \
                          AND NOT f.id = ANY($6) AND {filter_sql} \
                      ORDER BY f.id DESC LIMIT $5)) h \
                 ORDER BY h.id DESC LIMIT $5) \
             SELECT {FACT_COLS} FROM hits JOIN facts f ON f.id = hits.id ORDER BY f.id DESC"
        );
        let mut seen = BTreeSet::from([root]);
        let mut seen_edges: Vec<Uuid> = Vec::new();
        let mut edges = Vec::new();
        let mut frontier = vec![root];
        for _ in 0..depth {
            if frontier.is_empty() {
                break;
            }
            let remaining = edge_limit - edges.len() as i64;
            if remaining <= 0 {
                tracing::debug!(%root, depth, edge_limit, "graph walk stopped at the edge limit");
                break;
            }
            let found = sqlx::query(&sql)
                .bind(&scope_texts)
                .bind(&frontier)
                .bind(filter.as_of)
                .bind(filter.known_at)
                .bind(remaining)
                .bind(&seen_edges)
                .fetch_all(&mut *tx)
                .await
                .map_err(db)?;
            let mut next = Vec::new();
            for row in &found {
                let fact = rows::fact(row)?;
                seen_edges.push(fact.id);
                for entity in std::iter::once(fact.subject_id).chain(fact.object_id) {
                    if seen.insert(entity) {
                        next.push(entity);
                    }
                }
                edges.push(fact);
            }
            frontier = next;
        }

        let ids: Vec<Uuid> = seen.into_iter().collect();
        let sql = format!("SELECT {ENTITY_COLS} FROM entities WHERE id = ANY($1) ORDER BY id");
        let found = sqlx::query(&sql)
            .bind(&ids)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let entities = found.iter().map(rows::entity).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(GraphNeighborhood {
            root,
            entities,
            edges,
        })
    }

    async fn list_blocks(&self, ctx: &RequestContext, agent_id: &str) -> Result<Vec<Block>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!("SELECT {BLOCK_COLS} FROM blocks WHERE agent_id = $1 ORDER BY label");
        let found = sqlx::query(&sql)
            .bind(agent_id)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::block).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn get_block(&self, ctx: &RequestContext, agent_id: &str, label: &str) -> Result<Block> {
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!("SELECT {BLOCK_COLS} FROM blocks WHERE agent_id = $1 AND label = $2");
        let row = sqlx::query(&sql)
            .bind(agent_id)
            .bind(label)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("block {agent_id}/{label}")))?;
        let result = rows::block(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
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
        validate::non_empty("agent_id", agent_id)?;
        validate::non_empty("label", label)?;
        if max_chars <= 0 {
            return Err(Error::InvalidInput("max_chars must be positive".into()));
        }
        // Characters, not bytes: the limit must not penalize non-ASCII text.
        let len = content.chars().count();
        let max = max_chars as usize;
        if len > max {
            return Err(Error::BlockTooLarge { len, max });
        }

        let mut tx = self.tenant_tx(ctx).await?;
        let written = match expected_version {
            None => {
                let sql = format!(
                    "INSERT INTO blocks (id, tenant, agent_id, label, content, max_chars, version) \
                     VALUES ($1, $2, $3, $4, $5, $6, 1) \
                     ON CONFLICT (tenant, agent_id, label) DO NOTHING RETURNING {BLOCK_COLS}"
                );
                sqlx::query(&sql)
                    .bind(Uuid::now_v7())
                    .bind(&ctx.tenant)
                    .bind(agent_id)
                    .bind(label)
                    .bind(content)
                    .bind(max_chars)
                    .fetch_optional(&mut *tx)
                    .await
                    .map_err(db)?
            }
            // One statement: the version check is re-evaluated after any row lock wait, so of two
            // concurrent writers exactly one matches.
            Some(version) => {
                let sql = format!(
                    "UPDATE blocks SET content = $1, max_chars = $2, version = version + 1, \
                     updated_at = now() \
                     WHERE agent_id = $3 AND label = $4 AND version = $5 RETURNING {BLOCK_COLS}"
                );
                sqlx::query(&sql)
                    .bind(content)
                    .bind(max_chars)
                    .bind(agent_id)
                    .bind(label)
                    .bind(version)
                    .fetch_optional(&mut *tx)
                    .await
                    .map_err(db)?
            }
        };

        let Some(row) = written else {
            let current =
                sqlx::query("SELECT version FROM blocks WHERE agent_id = $1 AND label = $2")
                    .bind(agent_id)
                    .bind(label)
                    .fetch_optional(&mut *tx)
                    .await
                    .map_err(db)?;
            return Err(match current {
                Some(row) => Error::VersionConflict {
                    expected: expected_version.unwrap_or(0),
                    current: row.try_get("version").map_err(db)?,
                },
                None => Error::NotFound(format!("block {agent_id}/{label}")),
            });
        };
        let block = rows::block(&row)?;
        sqlx::query(
            "INSERT INTO block_versions (block_id, tenant, version, content, edited_by) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(block.id)
        .bind(&ctx.tenant)
        .bind(block.version)
        .bind(content)
        .bind(&ctx.actor)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        tx.commit().await.map_err(db)?;
        Ok(block)
    }

    async fn block_versions(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        label: &str,
    ) -> Result<Vec<BlockVersion>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(
            "SELECT v.version, v.content, v.edited_by, v.edited_at FROM block_versions v \
             JOIN blocks b ON b.id = v.block_id \
             WHERE b.agent_id = $1 AND b.label = $2 ORDER BY v.version",
        )
        .bind(agent_id)
        .bind(label)
        .fetch_all(&mut *tx)
        .await
        .map_err(db)?;
        // Every block has at least its first version, so no rows means no block.
        if found.is_empty() {
            return Err(Error::NotFound(format!("block {agent_id}/{label}")));
        }
        let result = found
            .iter()
            .map(rows::block_version)
            .collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn create_session(
        &self,
        ctx: &RequestContext,
        agent_id: &str,
        scope: &Scope,
    ) -> Result<Session> {
        validate::non_empty("agent_id", agent_id)?;
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!(
            "INSERT INTO sessions (id, tenant, agent_id, scope) VALUES ($1, $2, $3, $4) \
             RETURNING {SESSION_COLS}"
        );
        let row = sqlx::query(&sql)
            .bind(Uuid::now_v7())
            .bind(&ctx.tenant)
            .bind(agent_id)
            .bind(scope.to_string())
            .fetch_one(&mut *tx)
            .await
            .map_err(db)?;
        let result = rows::session(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn get_session(&self, ctx: &RequestContext, id: Uuid) -> Result<Session> {
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!("SELECT {SESSION_COLS} FROM sessions WHERE id = $1");
        let row = sqlx::query(&sql)
            .bind(id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("session {id}")))?;
        let result = rows::session(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn append_message(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
        role: &str,
        content: Value,
    ) -> Result<SessionMessage> {
        validate::non_empty("role", role)?;
        let mut tx = self.tenant_tx(ctx).await?;
        // The row lock from this UPDATE serializes appenders until commit, which also makes `seq`
        // gapless: a failed insert rolls the counter back.
        let row = sqlx::query(
            "UPDATE sessions SET next_seq = next_seq + 1 WHERE id = $1 \
             RETURNING next_seq - 1 AS seq",
        )
        .bind(session_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("session {session_id}")))?;
        let seq: i64 = row.try_get("seq").map_err(db)?;
        let sql = format!(
            "INSERT INTO session_messages (session_id, tenant, seq, role, content) \
             VALUES ($1, $2, $3, $4, $5) RETURNING {MESSAGE_COLS}"
        );
        let row = sqlx::query(&sql)
            .bind(session_id)
            .bind(&ctx.tenant)
            .bind(seq)
            .bind(role)
            .bind(content)
            .fetch_one(&mut *tx)
            .await
            .map_err(db)?;
        let result = rows::message(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn load_messages(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
    ) -> Result<Vec<SessionMessage>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let compacted_through = require_session(&mut tx, session_id).await?;
        let sql = format!(
            "SELECT {MESSAGE_COLS} FROM session_messages \
             WHERE session_id = $1 AND seq > $2 ORDER BY seq"
        );
        let found = sqlx::query(&sql)
            .bind(session_id)
            .bind(compacted_through)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::message).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn compact_session(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
        through_seq: i64,
        summary: &str,
    ) -> Result<()> {
        let mut tx = self.tenant_tx(ctx).await?;
        let row = sqlx::query(
            "SELECT compacted_through, next_seq FROM sessions WHERE id = $1 FOR UPDATE",
        )
        .bind(session_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("session {session_id}")))?;
        let compacted_through: i64 = row.try_get("compacted_through").map_err(db)?;
        let next_seq: i64 = row.try_get("next_seq").map_err(db)?;
        if through_seq >= next_seq {
            return Err(Error::InvalidInput(format!(
                "through_seq {through_seq} is past the last message"
            )));
        }
        // An older compaction arriving late must not overwrite a newer summary.
        if through_seq < compacted_through {
            return Err(Error::InvalidInput(format!(
                "session is already compacted through {compacted_through}"
            )));
        }
        sqlx::query("UPDATE sessions SET summary = $2, compacted_through = $3 WHERE id = $1")
            .bind(session_id)
            .bind(summary)
            .bind(through_seq)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        sqlx::query("DELETE FROM session_messages WHERE session_id = $1 AND seq <= $2")
            .bind(session_id)
            .bind(through_seq)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        tx.commit().await.map_err(db)
    }

    async fn session_summary(
        &self,
        ctx: &RequestContext,
        session_id: Uuid,
    ) -> Result<Option<String>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let row = sqlx::query("SELECT summary FROM sessions WHERE id = $1")
            .bind(session_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("session {session_id}")))?;
        let result = row.try_get("summary").map_err(db)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn claim_job(&self, worker_id: &str, lease_secs: i64) -> Result<Option<Job>> {
        validate::non_empty("worker_id", worker_id)?;
        if lease_secs <= 0 {
            return Err(Error::InvalidInput("lease_secs must be positive".into()));
        }
        let row = sqlx::query(
            "WITH due AS ( \
                 SELECT id FROM jobs \
                 WHERE (status = 'pending' AND run_after <= now()) \
                    OR (status = 'running' AND locked_until < now()) \
                 ORDER BY run_after, id \
                 FOR UPDATE SKIP LOCKED LIMIT 1) \
             UPDATE jobs j SET status = 'running', attempts = j.attempts + 1, locked_by = $1, \
                 locked_until = now() + make_interval(secs => $2) \
             FROM due WHERE j.id = due.id \
             RETURNING j.id, j.tenant, j.kind, j.payload, j.attempts",
        )
        .bind(worker_id)
        .bind(lease_secs as f64)
        .fetch_optional(&self.pool)
        .await
        .map_err(db)?;
        row.as_ref().map(rows::job).transpose()
    }

    async fn renew_job(&self, job: &Job, worker_id: &str, lease_secs: i64) -> Result<bool> {
        if lease_secs <= 0 {
            return Err(Error::InvalidInput("lease_secs must be positive".into()));
        }
        let updated = sqlx::query(
            "UPDATE jobs SET locked_until = now() + make_interval(secs => $4) \
             WHERE id = $1 AND status = 'running' AND locked_by = $2 AND attempts = $3",
        )
        .bind(job.id)
        .bind(worker_id)
        .bind(job.attempts)
        .bind(lease_secs as f64)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(updated.rows_affected() == 1)
    }

    async fn complete_job(&self, job: &Job, worker_id: &str) -> Result<bool> {
        let updated = sqlx::query(
            "UPDATE jobs SET status = 'done', locked_by = NULL, locked_until = NULL \
             WHERE id = $1 AND status = 'running' AND locked_by = $2 AND attempts = $3",
        )
        .bind(job.id)
        .bind(worker_id)
        .bind(job.attempts)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(updated.rows_affected() == 1)
    }

    async fn fail_job(
        &self,
        job: &Job,
        worker_id: &str,
        error: &str,
        run_after: DateTime<Utc>,
        max_attempts: i32,
    ) -> Result<bool> {
        // PostgreSQL rejects NUL in text, and failing here would lose the retry.
        let error = error.replace('\0', "");
        let updated = sqlx::query(
            "UPDATE jobs SET \
                 status = CASE WHEN attempts >= $6 THEN 'dead' ELSE 'pending' END, \
                 last_error = $4, run_after = $5, locked_by = NULL, locked_until = NULL \
             WHERE id = $1 AND status = 'running' AND locked_by = $2 AND attempts = $3",
        )
        .bind(job.id)
        .bind(worker_id)
        .bind(job.attempts)
        .bind(error)
        .bind(run_after)
        .bind(max_attempts)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(updated.rows_affected() == 1)
    }

    async fn release_job(
        &self,
        job: &Job,
        worker_id: &str,
        run_after: DateTime<Utc>,
    ) -> Result<bool> {
        let updated = sqlx::query(
            "UPDATE jobs SET status = 'pending', attempts = attempts - 1, run_after = $4, \
                 locked_by = NULL, locked_until = NULL \
             WHERE id = $1 AND status = 'running' AND locked_by = $2 AND attempts = $3",
        )
        .bind(job.id)
        .bind(worker_id)
        .bind(job.attempts)
        .bind(run_after)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(updated.rows_affected() == 1)
    }

    async fn find_api_key(&self, key_hash: &str) -> Result<Option<ApiKey>> {
        let sql = format!(
            "SELECT {API_KEY_COLS} FROM api_keys WHERE key_hash = $1 AND revoked_at IS NULL"
        );
        let row = sqlx::query(&sql)
            .bind(key_hash)
            .fetch_optional(&self.pool)
            .await
            .map_err(db)?;
        row.as_ref().map(rows::api_key).transpose()
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
        validate::non_empty("tenant", tenant)?;
        validate::non_empty("name", name)?;
        validate::key_hash(key_hash)?;
        let sql = format!(
            "INSERT INTO api_keys (id, tenant, name, key_hash, scopes, max_trust, is_admin) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING {API_KEY_COLS}"
        );
        let row = sqlx::query(&sql)
            .bind(Uuid::now_v7())
            .bind(tenant)
            .bind(name)
            .bind(key_hash)
            .bind(scopes)
            .bind(max_trust.as_str())
            .bind(is_admin)
            .fetch_one(&self.pool)
            .await
            .map_err(db)?;
        rows::api_key(&row)
    }

    async fn revoke_api_key(&self, id: Uuid) -> Result<()> {
        let updated = sqlx::query(
            "UPDATE api_keys SET revoked_at = COALESCE(revoked_at, now()) WHERE id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        if updated.rows_affected() == 0 {
            return Err(Error::NotFound(format!("api key {id}")));
        }
        Ok(())
    }

    async fn ping(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(db)?;
        Ok(())
    }
}

fn scored_facts(found: &[sqlx::postgres::PgRow], score_column: &str) -> Result<Vec<ScoredFact>> {
    let mut result = Vec::with_capacity(found.len());
    for row in found {
        result.push(ScoredFact {
            fact: rows::fact(row)?,
            score: row.try_get(score_column).map_err(db)?,
        });
    }
    Ok(result)
}
