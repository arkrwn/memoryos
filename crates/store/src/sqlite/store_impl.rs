//! `impl MemoryStore for SqliteStore`. SQL fragments are fixed; values, including FTS expressions
//! constructed from quoted user tokens, are always bound parameters.

use std::collections::BTreeSet;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use memoryos_core::{
    normalize_name, ApiKey, Block, BlockVersion, Entity, Episode, Error, Fact, FactFilter, FactOp,
    FactStatus, GraphNeighborhood, MemoryStore, NewEntity, NewEpisode, PurgeReport, RequestContext,
    Result, Scope, ScoredFact, Trust,
};
use serde_json::{json, Value};
use sqlx::sqlite::SqliteRow;
use sqlx::{AssertSqlSafe, Row, SqliteConnection};
use uuid::Uuid;

use super::rows::{self, API_KEY_COLS, BLOCK_COLS, ENTITY_COLS, EPISODE_COLS, FACT_COLS};
use super::{db, embedding, facts, json_array, storable, uuid_strings, SqliteStore};

const MAX_GRAPH_EDGES: i64 = 5000;
const MAX_SUBJECT_FACTS: i64 = 200;
const MAX_BLOCK_VERSIONS: i64 = 100;

async fn load_fact(conn: &mut SqliteConnection, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
    let sql = format!("SELECT {FACT_COLS} FROM facts f WHERE f.tenant = ?1 AND f.id = ?2");
    let row = sqlx::query(AssertSqlSafe(sql))
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .fetch_optional(conn)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("fact {id}")))?;
    rows::fact(&row)
}

struct QuarantineState {
    status: String,
    trust: Trust,
    invalidated: bool,
    valid_from: DateTime<Utc>,
    pending_supersede_of: Option<Uuid>,
}

async fn read_quarantined(
    conn: &mut SqliteConnection,
    ctx: &RequestContext,
    id: Uuid,
) -> Result<QuarantineState> {
    let row = sqlx::query(
        "SELECT status, trust, invalidated_at, valid_from, pending_supersede_of \
         FROM facts WHERE tenant = ?1 AND id = ?2",
    )
    .bind(&ctx.tenant)
    .bind(id.to_string())
    .fetch_optional(conn)
    .await
    .map_err(db)?
    .ok_or_else(|| Error::NotFound(format!("fact {id}")))?;
    let invalidated_at: Option<i64> = row.try_get("invalidated_at").map_err(db)?;
    let valid_from: i64 = row.try_get("valid_from").map_err(db)?;
    let pending: Option<String> = row.try_get("pending_supersede_of").map_err(db)?;
    let state = QuarantineState {
        status: row.try_get("status").map_err(db)?,
        trust: row
            .try_get::<String, _>("trust")
            .map_err(db)?
            .parse()
            .map_err(|_| Error::Storage("fact has an invalid trust".into()))?,
        invalidated: invalidated_at.is_some(),
        valid_from: DateTime::from_timestamp_micros(valid_from)
            .ok_or_else(|| Error::Storage("fact has an invalid valid_from".into()))?,
        pending_supersede_of: pending
            .map(|value| {
                Uuid::parse_str(&value)
                    .map_err(|_| Error::Storage("fact has an invalid pending target".into()))
            })
            .transpose()?,
    };
    if state.status != FactStatus::Quarantined.as_str() {
        return Err(Error::InvalidInput(format!(
            "fact {id} is {}, not quarantined",
            state.status
        )));
    }
    Ok(state)
}

#[async_trait]
impl MemoryStore for SqliteStore {
    async fn claim_episode(
        &self,
        ctx: &RequestContext,
        episode: NewEpisode,
    ) -> Result<(Episode, bool)> {
        crate::validate::non_empty("source", &episode.source)?;
        crate::validate::non_empty("content", &episode.content)?;
        storable(&episode.source)?;
        storable(&episode.content)?;
        let metadata = match episode.metadata {
            Value::Null => json!({}),
            other => other,
        };
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let id = Uuid::now_v7();
        let sql = format!(
            "INSERT INTO episodes (id, tenant, scope, source, trust, content, metadata, \
             occurred_at, created_at, idempotency_key) \
             VALUES (?2, ?1, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) \
             ON CONFLICT (tenant, idempotency_key) DO NOTHING RETURNING {EPISODE_COLS}"
        );
        let inserted = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(id.to_string())
            .bind(episode.scope.to_string())
            .bind(&episode.source)
            .bind(episode.trust.as_str())
            .bind(&episode.content)
            .bind(json_array(&metadata)?)
            .bind(rows::timestamp(episode.occurred_at.unwrap_or(now)))
            .bind(rows::timestamp(now))
            .bind(&episode.idempotency_key)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?;
        let created = inserted.is_some();
        let row = match inserted {
            Some(row) => row,
            None => {
                let sql = format!(
                    "SELECT {EPISODE_COLS} FROM episodes \
                     WHERE tenant = ?1 AND idempotency_key = ?2"
                );
                sqlx::query(AssertSqlSafe(sql))
                    .bind(&ctx.tenant)
                    .bind(&episode.idempotency_key)
                    .fetch_one(&mut *tx)
                    .await
                    .map_err(db)?
            }
        };
        let result = rows::episode(&row)?;
        tx.commit().await.map_err(db)?;
        Ok((result, created))
    }

    async fn discard_unprocessed_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<bool> {
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let pending = sqlx::query(
            "SELECT 1 FROM episodes WHERE tenant = ?1 AND id = ?2 AND processed_at IS NULL",
        )
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .fetch_optional(&mut *tx)
        .await
        .map_err(db)?
        .is_some();
        if !pending {
            tx.commit().await.map_err(db)?;
            return Ok(false);
        }
        sqlx::query("DELETE FROM episodes WHERE tenant = ?1 AND id = ?2")
            .bind(&ctx.tenant)
            .bind(id.to_string())
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        facts::audit(&mut tx, ctx, "episode.discard", Some(id), json!({}), now).await?;
        tx.commit().await.map_err(db)?;
        Ok(true)
    }

    async fn get_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<Episode> {
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!("SELECT {EPISODE_COLS} FROM episodes WHERE tenant = ?1 AND id = ?2");
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(id.to_string())
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("episode {id}")))?;
        let result = rows::episode(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn purge_episode(&self, ctx: &RequestContext, id: Uuid) -> Result<PurgeReport> {
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        sqlx::query("SELECT id FROM episodes WHERE tenant = ?1 AND id = ?2")
            .bind(&ctx.tenant)
            .bind(id.to_string())
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("episode {id}")))?;
        let outcome = facts::purge_episode(&mut tx, ctx, id).await?;
        let mut detail = facts::purge_detail(&outcome.report);
        detail["facts_requarantined"] = json!(outcome.facts_requarantined);
        facts::audit(&mut tx, ctx, "episode.purge", Some(id), detail, now).await?;
        tx.commit().await.map_err(db)?;
        Ok(outcome.report)
    }

    async fn purge_scope(&self, ctx: &RequestContext, scope: &Scope) -> Result<PurgeReport> {
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let report = facts::purge_scope(&mut tx, ctx, scope).await?;
        let mut detail = facts::purge_detail(&report);
        detail["scope"] = json!(scope.to_string());
        facts::audit(&mut tx, ctx, "scope.purge", None, detail, now).await?;
        tx.commit().await.map_err(db)?;
        Ok(report)
    }

    async fn apply_commit(
        &self,
        ctx: &RequestContext,
        episode_id: Uuid,
        ops: Vec<FactOp>,
    ) -> Result<Vec<Uuid>> {
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let row = sqlx::query(
            "SELECT scope, trust, processed_at FROM episodes WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(episode_id.to_string())
        .fetch_optional(&mut *tx)
        .await
        .map_err(db)?
        .ok_or_else(|| Error::NotFound(format!("episode {episode_id}")))?;
        let processed: Option<i64> = row.try_get("processed_at").map_err(db)?;
        if processed.is_some() {
            tx.commit().await.map_err(db)?;
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
        let inserted = facts::apply_ops(&mut tx, ctx, &episode, &ops, now).await?;
        sqlx::query("UPDATE episodes SET processed_at = ?3 WHERE tenant = ?1 AND id = ?2")
            .bind(&ctx.tenant)
            .bind(episode_id.to_string())
            .bind(rows::timestamp(now))
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        tx.commit().await.map_err(db)?;
        Ok(inserted)
    }

    async fn upsert_entity(&self, ctx: &RequestContext, entity: NewEntity) -> Result<Entity> {
        crate::validate::non_empty("kind", &entity.kind)?;
        crate::validate::non_empty("name", &entity.name)?;
        storable(&entity.kind)?;
        storable(&entity.name)?;
        for alias in &entity.aliases {
            storable(alias)?;
        }
        let embedding = embedding(&entity.embedding)?;
        let normalized = normalize_name(&entity.name);
        let alias_keys: Vec<String> = entity.aliases.iter().map(|a| normalize_name(a)).collect();
        let scope = entity.scope.to_string();
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let id = Uuid::now_v7();
        let sql = format!(
            "INSERT INTO entities (id, tenant, scope, kind, name, normalized_name, aliases, \
             alias_keys, created_at) VALUES (?2, ?1, ?3, ?4, ?5, ?6, ?7, ?8, ?9) \
             ON CONFLICT (tenant, scope, kind, normalized_name) DO NOTHING RETURNING {ENTITY_COLS}"
        );
        let inserted = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(id.to_string())
            .bind(&scope)
            .bind(&entity.kind)
            .bind(&entity.name)
            .bind(&normalized)
            .bind(json_array(&entity.aliases)?)
            .bind(json_array(&alias_keys)?)
            .bind(rows::timestamp(now))
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?;
        let row = match inserted {
            Some(row) => {
                sqlx::query(
                    "INSERT INTO entity_embeddings (rowid, embedding) \
                     SELECT rowid, ?3 FROM entities WHERE tenant = ?1 AND id = ?2",
                )
                .bind(&ctx.tenant)
                .bind(id.to_string())
                .bind(embedding)
                .execute(&mut *tx)
                .await
                .map_err(db)?;
                row
            }
            None => {
                let sql = format!(
                    "SELECT {ENTITY_COLS} FROM entities \
                     WHERE tenant = ?1 AND scope = ?2 AND kind = ?3 AND normalized_name = ?4"
                );
                sqlx::query(AssertSqlSafe(sql))
                    .bind(&ctx.tenant)
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
        let sql = format!("SELECT {ENTITY_COLS} FROM entities WHERE tenant = ?1 AND id = ?2");
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(id.to_string())
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
        embedding_: &[f32],
        limit: i64,
    ) -> Result<Vec<(Entity, f64)>> {
        let limit = crate::validate::limit(limit)?;
        let query = embedding(embedding_)?;
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!(
            "SELECT {ENTITY_COLS}, vec_distance_cosine(v.embedding, ?3) AS distance \
             FROM entity_embeddings v JOIN entities e ON e.rowid = v.rowid AND e.tenant = ?1 \
             WHERE e.scope = ?2 ORDER BY distance, e.id LIMIT ?4"
        );
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(scope.to_string())
            .bind(query)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let mut result = Vec::with_capacity(found.len());
        for row in &found {
            let distance: f64 = row.try_get("distance").map_err(db)?;
            result.push((rows::entity(row)?, (1.0 - distance).clamp(0.0, 1.0)));
        }
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn add_entity_alias(
        &self,
        ctx: &RequestContext,
        id: Uuid,
        alias: &str,
        source_episode: Option<Uuid>,
    ) -> Result<()> {
        crate::validate::non_empty("alias", alias)?;
        storable(alias)?;
        let alias_key = normalize_name(alias);
        let mut tx = self.write_tx(ctx).await?;
        let row =
            sqlx::query("SELECT aliases, alias_keys FROM entities WHERE tenant = ?1 AND id = ?2")
                .bind(&ctx.tenant)
                .bind(id.to_string())
                .fetch_optional(&mut *tx)
                .await
                .map_err(db)?
                .ok_or_else(|| Error::NotFound(format!("entity {id}")))?;
        let mut aliases: Vec<String> = decode_json(&row, "aliases")?;
        let mut alias_keys: Vec<String> = decode_json(&row, "alias_keys")?;
        if !aliases.iter().any(|value| value == alias) {
            aliases.push(alias.to_owned());
        }
        if !alias_keys.iter().any(|value| value == &alias_key) {
            alias_keys.push(alias_key.clone());
        }
        sqlx::query(
            "UPDATE entities SET aliases = ?3, alias_keys = ?4 WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .bind(json_array(&aliases)?)
        .bind(json_array(&alias_keys)?)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        if let Some(episode_id) = source_episode {
            sqlx::query(
                "INSERT INTO entity_alias_sources (entity_id, alias_key, episode_id, tenant, alias) \
                 VALUES (?2, ?3, ?4, ?1, ?5) \
                 ON CONFLICT (entity_id, alias_key, episode_id) DO NOTHING",
            )
            .bind(&ctx.tenant)
            .bind(id.to_string())
            .bind(alias_key)
            .bind(episode_id.to_string())
            .bind(alias)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
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
        crate::validate::non_empty("kind", kind)?;
        crate::validate::non_empty("name", name)?;
        let normalized = normalize_name(name);
        let scope = scope.to_string();
        let mut tx = self.tenant_tx(ctx).await?;
        let by_name = format!(
            "SELECT {ENTITY_COLS} FROM entities e \
             WHERE e.tenant = ?1 AND e.scope = ?2 AND e.kind = ?3 AND e.normalized_name = ?4"
        );
        let by_alias = format!(
            "SELECT {ENTITY_COLS} FROM entities e \
             WHERE e.tenant = ?1 AND e.scope = ?2 AND e.kind = ?3 \
               AND EXISTS (SELECT 1 FROM json_each(e.alias_keys) a WHERE a.value = ?4) \
             ORDER BY e.id LIMIT 1"
        );
        let mut found = None;
        for sql in [&by_name, &by_alias] {
            found = sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(&ctx.tenant)
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
        let result = load_fact(&mut tx, ctx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn facts_for_episode(&self, ctx: &RequestContext, episode_id: Uuid) -> Result<Vec<Fact>> {
        let mut tx = self.tenant_tx(ctx).await?;
        sqlx::query("SELECT id FROM episodes WHERE tenant = ?1 AND id = ?2")
            .bind(&ctx.tenant)
            .bind(episode_id.to_string())
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?
            .ok_or_else(|| Error::NotFound(format!("episode {episode_id}")))?;
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f JOIN fact_sources l \
                 ON l.tenant = f.tenant AND l.fact_id = f.id \
             WHERE f.tenant = ?1 AND l.episode_id = ?2 ORDER BY f.id"
        );
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(episode_id.to_string())
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn fact_history(&self, ctx: &RequestContext, id: Uuid) -> Result<Vec<Fact>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!(
            "WITH RECURSIVE chain(id) AS ( \
                 SELECT id FROM facts WHERE tenant = ?1 AND id = ?2 \
                 UNION \
                 SELECT n.id FROM chain c \
                 JOIN facts cur ON cur.tenant = ?1 AND cur.id = c.id \
                 JOIN facts n ON n.tenant = cur.tenant AND n.scope = cur.scope \
                     AND (n.superseded_by = cur.id OR n.id = cur.superseded_by)) \
             SELECT {FACT_COLS} FROM facts f JOIN chain c ON c.id = f.id \
             WHERE f.tenant = ?1 ORDER BY f.valid_from, f.recorded_at, f.id"
        );
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(id.to_string())
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
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let updated = sqlx::query(
            "UPDATE facts SET invalidated_at = COALESCE(invalidated_at, ?3) \
             WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .bind(rows::timestamp(now))
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        if updated.rows_affected() == 0 {
            return Err(Error::NotFound(format!("fact {id}")));
        }
        facts::leave_chain_on_retraction(&mut tx, ctx, id).await?;
        facts::audit(
            &mut tx,
            ctx,
            "fact.invalidate",
            Some(id),
            json!({ "reason": reason }),
            now,
        )
        .await?;
        let result = load_fact(&mut tx, ctx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn current_fact(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        id: Uuid,
    ) -> Result<Option<Fact>> {
        let now = rows::timestamp(Utc::now());
        let filter = rows::visible(4, 4);
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.tenant = ?1 AND f.id = ?2 AND f.scope = ?3 AND {filter}"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(id.to_string())
            .bind(scope.to_string())
            .bind(now)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?;
        let result = row.as_ref().map(rows::fact).transpose()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn current_facts_for_subject(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        subject_id: Uuid,
    ) -> Result<Vec<Fact>> {
        let now = rows::timestamp(Utc::now());
        let filter = rows::visible(4, 4);
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.tenant = ?1 AND f.scope = ?2 AND f.subject_id = ?3 AND {filter} \
             ORDER BY f.valid_from DESC, f.id DESC LIMIT ?5"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(scope.to_string())
            .bind(subject_id.to_string())
            .bind(now)
            .bind(MAX_SUBJECT_FACTS)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let mut result = found.iter().map(rows::fact).collect::<Result<Vec<_>>>()?;
        result.reverse();
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn search_facts_vector(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        embedding_: &[f32],
        filter: &FactFilter,
        limit: i64,
    ) -> Result<Vec<ScoredFact>> {
        let limit = crate::validate::limit(limit)?;
        let query = embedding(embedding_)?;
        if scopes.is_empty() {
            return Ok(Vec::new());
        }
        let now = Utc::now();
        let as_of = rows::timestamp(filter.as_of.unwrap_or(now));
        let known_at = rows::timestamp(filter.known_at.unwrap_or(now));
        let scopes = json_array(&crate::validate::scope_strings(scopes))?;
        let visibility = rows::visible(4, 5);
        let sql = format!(
            "SELECT {FACT_COLS}, 1.0 - vec_distance_cosine(v.embedding, ?2) AS similarity \
             FROM fact_embeddings v JOIN facts f ON f.rowid = v.rowid AND f.tenant = ?1 \
             WHERE f.scope IN (SELECT value FROM json_each(?3)) AND {visibility} \
             ORDER BY vec_distance_cosine(v.embedding, ?2), f.id LIMIT ?6"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(query)
            .bind(scopes)
            .bind(as_of)
            .bind(known_at)
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
        let limit = crate::validate::limit(limit)?;
        if scopes.is_empty() {
            return Ok(Vec::new());
        }
        storable(query)?;
        let Some(match_query) = safe_match_query(query) else {
            return Ok(Vec::new());
        };
        let now = Utc::now();
        let as_of = rows::timestamp(filter.as_of.unwrap_or(now));
        let known_at = rows::timestamp(filter.known_at.unwrap_or(now));
        let scopes = json_array(&crate::validate::scope_strings(scopes))?;
        let visibility = rows::visible(4, 5);
        // Ranking parity with PostgreSQL is not promised. Filters are applied before the newest
        // candidate cap. Per-document match counts avoid bm25's cross-tenant corpus statistics.
        let sql = format!(
            "WITH candidates AS MATERIALIZED (\
                 SELECT f.rowid AS rid FROM facts f \
                 JOIN facts_fts ON facts_fts.rowid = f.rowid \
                 WHERE f.tenant = ?1 AND facts_fts MATCH ?2 \
                   AND f.scope IN (SELECT value FROM json_each(?3)) AND {visibility} \
                 ORDER BY f.id DESC LIMIT MAX(1000, ?6)),\
             ranked AS (\
                 SELECT c.rid, highlight(facts_fts, 0, char(1), char(2)) AS highlighted \
                 FROM candidates c JOIN facts_fts ON facts_fts.rowid = c.rid \
                 WHERE facts_fts MATCH ?2)\
             SELECT {FACT_COLS}, r.highlighted FROM ranked r \
             JOIN facts f ON f.rowid = r.rid AND f.tenant = ?1"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(match_query)
            .bind(scopes)
            .bind(as_of)
            .bind(known_at)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = text_scored_facts(&found, limit)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn list_quarantined(
        &self,
        ctx: &RequestContext,
        scopes: &[Scope],
        limit: i64,
    ) -> Result<Vec<Fact>> {
        let limit = crate::validate::limit(limit)?;
        if scopes.is_empty() {
            return Ok(Vec::new());
        }
        let scopes = json_array(&crate::validate::scope_strings(scopes))?;
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.tenant = ?1 AND f.scope IN (SELECT value FROM json_each(?2)) \
               AND f.status = 'quarantined' ORDER BY f.recorded_at, f.id LIMIT ?3"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(scopes)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn approve_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let state = read_quarantined(&mut tx, ctx, id).await?;
        if state.invalidated {
            return Err(Error::InvalidInput(format!(
                "fact {id} was retracted and cannot be approved"
            )));
        }
        if let Some(old_id) = state.pending_supersede_of {
            let settled = facts::settle_supersede(
                &mut tx,
                ctx,
                id,
                state.trust,
                state.valid_from,
                old_id,
                Some(old_id),
                None,
            )
            .await?;
            if let facts::Settled::Outranked { head } = settled {
                sqlx::query(
                    "UPDATE facts SET pending_supersede_of = ?3 WHERE tenant = ?1 AND id = ?2",
                )
                .bind(&ctx.tenant)
                .bind(id.to_string())
                .bind(head.to_string())
                .execute(&mut *tx)
                .await
                .map_err(db)?;
                facts::audit(
                    &mut tx,
                    ctx,
                    "fact.approve_held",
                    Some(id),
                    json!({ "reviewed": old_id, "head": head }),
                    now,
                )
                .await?;
                tx.commit().await.map_err(db)?;
                return Err(Error::InvalidInput(format!(
                    "fact {id} now contradicts fact {head}, which has higher trust: review it again"
                )));
            }
        }
        sqlx::query(
            "UPDATE facts SET status = 'active', pending_supersede_of = NULL \
             WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        facts::audit(
            &mut tx,
            ctx,
            "fact.approve",
            Some(id),
            json!({ "from_status": state.status, "superseded": state.pending_supersede_of }),
            now,
        )
        .await?;
        let result = load_fact(&mut tx, ctx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn reject_fact(&self, ctx: &RequestContext, id: Uuid) -> Result<Fact> {
        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let state = read_quarantined(&mut tx, ctx, id).await?;
        sqlx::query(
            "UPDATE facts SET status = 'rejected', pending_supersede_of = NULL \
             WHERE tenant = ?1 AND id = ?2",
        )
        .bind(&ctx.tenant)
        .bind(id.to_string())
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        facts::audit(
            &mut tx,
            ctx,
            "fact.reject",
            Some(id),
            json!({ "from_status": state.status }),
            now,
        )
        .await?;
        let result = load_fact(&mut tx, ctx, id).await?;
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
        let scope_texts = json_array(&crate::validate::scope_strings(scopes))?;
        let now = Utc::now();
        let as_of = rows::timestamp(filter.as_of.unwrap_or(now));
        let known_at = rows::timestamp(filter.known_at.unwrap_or(now));
        let mut tx = self.tenant_tx(ctx).await?;
        let root_row = sqlx::query(
            "SELECT id FROM entities WHERE tenant = ?1 AND id = ?2 \
             AND scope IN (SELECT value FROM json_each(?3))",
        )
        .bind(&ctx.tenant)
        .bind(root.to_string())
        .bind(&scope_texts)
        .fetch_optional(&mut *tx)
        .await
        .map_err(db)?;
        if root_row.is_none() {
            return Err(Error::NotFound(format!("entity {root}")));
        }

        let visibility = rows::visible(4, 5);
        let sql = format!(
            "WITH hits AS MATERIALIZED (\
                 SELECT DISTINCT f.id FROM facts f JOIN json_each(?2) frontier \
                   ON f.subject_id = frontier.value OR f.object_id = frontier.value \
                 WHERE f.tenant = ?1 AND f.scope IN (SELECT value FROM json_each(?3)) \
                   AND f.id NOT IN (SELECT value FROM json_each(?7)) AND {visibility} \
                 ORDER BY f.id DESC LIMIT ?6)\
             SELECT {FACT_COLS} FROM hits JOIN facts f ON f.tenant = ?1 AND f.id = hits.id \
             ORDER BY f.id DESC"
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
            let found = sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(&ctx.tenant)
                .bind(json_array(&uuid_strings(frontier))?)
                .bind(&scope_texts)
                .bind(as_of)
                .bind(known_at)
                .bind(remaining)
                .bind(json_array(&uuid_strings(seen_edges.clone()))?)
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

        let ids = json_array(&uuid_strings(seen))?;
        let sql = format!(
            "SELECT {ENTITY_COLS} FROM entities e WHERE e.tenant = ?1 \
             AND e.id IN (SELECT value FROM json_each(?2)) ORDER BY e.id"
        );
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(ids)
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
        let sql = format!(
            "SELECT {BLOCK_COLS} FROM blocks WHERE tenant = ?1 AND agent_id = ?2 ORDER BY label"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(agent_id)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::block).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn get_block(&self, ctx: &RequestContext, agent_id: &str, label: &str) -> Result<Block> {
        let sql = format!(
            "SELECT {BLOCK_COLS} FROM blocks WHERE tenant = ?1 AND agent_id = ?2 AND label = ?3"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
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
        crate::validate::non_empty("agent_id", agent_id)?;
        crate::validate::non_empty("label", label)?;
        storable(agent_id)?;
        storable(label)?;
        storable(content)?;
        if max_chars <= 0 {
            return Err(Error::InvalidInput("max_chars must be positive".into()));
        }
        let len = content.chars().count();
        let max = max_chars as usize;
        if len > max {
            return Err(Error::BlockTooLarge { len, max });
        }

        let mut tx = self.write_tx(ctx).await?;
        let now = Utc::now();
        let written = match expected_version {
            None => {
                let sql = format!(
                    "INSERT INTO blocks (id, tenant, agent_id, label, content, max_chars, version, \
                     updated_at) VALUES (?2, ?1, ?3, ?4, ?5, ?6, 1, ?7) \
                     ON CONFLICT (tenant, agent_id, label) DO NOTHING RETURNING {BLOCK_COLS}"
                );
                sqlx::query(AssertSqlSafe(sql))
                    .bind(&ctx.tenant)
                    .bind(Uuid::now_v7().to_string())
                    .bind(agent_id)
                    .bind(label)
                    .bind(content)
                    .bind(max_chars)
                    .bind(rows::timestamp(now))
                    .fetch_optional(&mut *tx)
                    .await
                    .map_err(db)?
            }
            Some(version) => {
                let sql = format!(
                    "UPDATE blocks SET content = ?4, max_chars = ?5, version = version + 1, \
                     updated_at = ?6 WHERE tenant = ?1 AND agent_id = ?2 AND label = ?3 \
                       AND version = ?7 RETURNING {BLOCK_COLS}"
                );
                sqlx::query(AssertSqlSafe(sql))
                    .bind(&ctx.tenant)
                    .bind(agent_id)
                    .bind(label)
                    .bind(content)
                    .bind(max_chars)
                    .bind(rows::timestamp(now))
                    .bind(version)
                    .fetch_optional(&mut *tx)
                    .await
                    .map_err(db)?
            }
        };

        let Some(row) = written else {
            let current = sqlx::query(
                "SELECT version FROM blocks WHERE tenant = ?1 AND agent_id = ?2 AND label = ?3",
            )
            .bind(&ctx.tenant)
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
            "INSERT INTO block_versions (block_id, tenant, version, content, edited_by, edited_at) \
             VALUES (?2, ?1, ?3, ?4, ?5, ?6)",
        )
        .bind(&ctx.tenant)
        .bind(block.id.to_string())
        .bind(block.version)
        .bind(content)
        .bind(&ctx.actor)
        .bind(rows::timestamp(now))
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        sqlx::query(
            "DELETE FROM block_versions WHERE tenant = ?1 AND block_id = ?2 AND version <= ?3",
        )
        .bind(&ctx.tenant)
        .bind(block.id.to_string())
        .bind(block.version - MAX_BLOCK_VERSIONS)
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
             JOIN blocks b ON b.tenant = v.tenant AND b.id = v.block_id \
             WHERE v.tenant = ?1 AND b.agent_id = ?2 AND b.label = ?3 \
             ORDER BY v.version DESC LIMIT ?4",
        )
        .bind(&ctx.tenant)
        .bind(agent_id)
        .bind(label)
        .bind(MAX_BLOCK_VERSIONS)
        .fetch_all(&mut *tx)
        .await
        .map_err(db)?;
        if found.is_empty() {
            return Err(Error::NotFound(format!("block {agent_id}/{label}")));
        }
        let mut result = found
            .iter()
            .map(rows::block_version)
            .collect::<Result<Vec<_>>>()?;
        result.reverse();
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn find_api_key(&self, key_hash: &str) -> Result<Option<ApiKey>> {
        let sql = format!(
            "SELECT {API_KEY_COLS} FROM api_keys WHERE key_hash = ?1 AND revoked_at IS NULL"
        );
        let mut tx = self.pool.begin().await.map_err(db)?;
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(key_hash)
            .fetch_optional(&mut *tx)
            .await
            .map_err(db)?;
        let result = row.as_ref().map(rows::api_key).transpose()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
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
        crate::validate::non_empty("tenant", tenant)?;
        crate::validate::non_empty("name", name)?;
        crate::validate::key_hash(key_hash)?;
        storable(tenant)?;
        storable(name)?;
        for scope in scopes {
            storable(scope)?;
        }
        let sql = format!(
            "INSERT INTO api_keys (id, tenant, name, key_hash, scopes, max_trust, is_admin, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) RETURNING {API_KEY_COLS}"
        );
        let mut tx = self.any_write_tx().await?;
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(Uuid::now_v7().to_string())
            .bind(tenant)
            .bind(name)
            .bind(key_hash)
            .bind(json_array(&scopes)?)
            .bind(max_trust.as_str())
            .bind(is_admin)
            .bind(rows::timestamp(Utc::now()))
            .fetch_one(&mut *tx)
            .await
            .map_err(db)?;
        let result = rows::api_key(&row)?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    async fn revoke_api_key(&self, id: Uuid) -> Result<()> {
        let mut tx = self.any_write_tx().await?;
        let updated =
            sqlx::query("UPDATE api_keys SET revoked_at = COALESCE(revoked_at, ?2) WHERE id = ?1")
                .bind(id.to_string())
                .bind(rows::timestamp(Utc::now()))
                .execute(&mut *tx)
                .await
                .map_err(db)?;
        if updated.rows_affected() == 0 {
            return Err(Error::NotFound(format!("api key {id}")));
        }
        tx.commit().await.map_err(db)
    }

    async fn ping(&self) -> Result<()> {
        let mut tx = self.pool.begin().await.map_err(db)?;
        sqlx::query("SELECT 1")
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        tx.commit().await.map_err(db)?;
        Ok(())
    }
}

fn text_scored_facts(found: &[SqliteRow], limit: i64) -> Result<Vec<ScoredFact>> {
    let mut result = Vec::with_capacity(found.len());
    for row in found {
        let fact = rows::fact(row)?;
        let highlighted: String = row.try_get("highlighted").map_err(db)?;
        let inserted = highlighted.matches('\u{1}').count();
        let existing = fact.statement.matches('\u{1}').count();
        result.push(ScoredFact {
            fact,
            score: inserted.saturating_sub(existing) as f64,
        });
    }
    result.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.fact.id.cmp(&b.fact.id))
    });
    result.truncate(limit as usize);
    Ok(result)
}

fn scored_facts(found: &[SqliteRow], score_column: &str) -> Result<Vec<ScoredFact>> {
    let mut result = Vec::with_capacity(found.len());
    for row in found {
        result.push(ScoredFact {
            fact: rows::fact(row)?,
            score: row.try_get(score_column).map_err(db)?,
        });
    }
    Ok(result)
}

fn decode_json<T: serde::de::DeserializeOwned>(row: &SqliteRow, column: &str) -> Result<T> {
    let value: String = row.try_get(column).map_err(db)?;
    serde_json::from_str(&value)
        .map_err(|e| Error::Storage(format!("column {column} holds invalid JSON: {e}")))
}

/// Builds an FTS5 expression from literal quoted terms, following PostgreSQL
/// `websearch_to_tsquery`: quoted phrases, a leading `-` for negation, and `or` between two terms
/// as OR, with AND binding tighter. Every piece of user text is escaped and quoted, so FTS
/// operators are never injected. Terms with no letter or digit are dropped, as PostgreSQL drops
/// them, because FTS5 would read them as a phrase that matches no row. FTS5 has no standalone NOT,
/// so an OR group made only of negated terms is dropped; a query of only such groups matches
/// nothing.
fn safe_match_query(input: &str) -> Option<String> {
    struct Term {
        negated: bool,
        quoted: bool,
        text: String,
    }

    let mut chars = input.chars().peekable();
    let mut terms: Vec<Term> = Vec::new();
    while chars.peek().is_some() {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        let negated = if chars.peek() == Some(&'-') {
            let mut lookahead = chars.clone();
            lookahead.next();
            if lookahead.next().is_some_and(|c| c != '-') {
                chars.next();
                true
            } else {
                false
            }
        } else {
            false
        };
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        let quoted = chars.next_if_eq(&'"').is_some();
        let mut text = String::new();
        if quoted {
            let mut closed = false;
            while let Some(c) = chars.next() {
                if c == '"' {
                    if chars.next_if_eq(&'"').is_some() {
                        text.push('"');
                    } else {
                        closed = true;
                        break;
                    }
                } else {
                    text.push(c);
                }
            }
            if !closed {
                // Treat an unmatched quote as literal content rather than FTS syntax.
                text.insert(0, '"');
            }
        } else {
            for c in chars.by_ref() {
                if c.is_whitespace() {
                    break;
                }
                text.push(c);
            }
        }
        if text.chars().any(char::is_alphanumeric) {
            terms.push(Term {
                negated,
                quoted,
                text,
            });
        }
    }

    // A bare `or` is an operator only between two terms. PostgreSQL reads a leading, trailing or
    // repeated `or` as a word.
    let mut groups: Vec<Vec<Term>> = vec![Vec::new()];
    let mut previous_was_or = false;
    let count = terms.len();
    for (index, term) in terms.into_iter().enumerate() {
        let is_or = !term.negated && !term.quoted && term.text.eq_ignore_ascii_case("or");
        let current_has_term = groups.last().is_some_and(|group| !group.is_empty());
        if is_or && !previous_was_or && current_has_term && index + 1 < count {
            groups.push(Vec::new());
            previous_was_or = true;
            continue;
        }
        previous_was_or = false;
        if let Some(group) = groups.last_mut() {
            group.push(term);
        }
    }

    let quote = |text: &str| format!("\"{}\"", text.replace('"', "\"\""));
    let groups: Vec<String> = groups
        .into_iter()
        .filter(|group| group.iter().any(|term| !term.negated))
        .map(|mut group| {
            // Positive terms must precede NOT in FTS5. Reordering conjunctions does not change
            // the match.
            group.sort_by_key(|term| term.negated);
            let mut expression = String::new();
            for (index, term) in group.iter().enumerate() {
                if index > 0 {
                    expression.push_str(if term.negated { " NOT " } else { " AND " });
                }
                expression.push_str(&quote(&term.text));
            }
            expression
        })
        .collect();
    match groups.len() {
        0 => None,
        1 => groups.into_iter().next(),
        _ => Some(
            groups
                .iter()
                .map(|group| format!("({group})"))
                .collect::<Vec<_>>()
                .join(" OR "),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::safe_match_query;

    #[test]
    fn match_queries_quote_operators_and_phrases() {
        assert_eq!(safe_match_query("rust"), Some("\"rust\"".into()));
        assert_eq!(
            safe_match_query("\"rust and\" -nothingness"),
            Some("\"rust and\" NOT \"nothingness\"".into())
        );
        assert_eq!(
            safe_match_query("'; DROP TABLE facts; --"),
            Some("\"DROP\" AND \"TABLE\" AND \"facts;\"".into())
        );
        assert_eq!(safe_match_query(""), None);
        assert_eq!(safe_match_query("-secret"), None);
    }

    #[test]
    fn terms_without_letters_or_digits_are_dropped() {
        assert_eq!(
            safe_match_query("rust & go"),
            Some("\"rust\" AND \"go\"".into())
        );
        assert_eq!(
            safe_match_query("deploy -- staging / \"--\""),
            Some("\"deploy\" AND \"staging\"".into())
        );
        assert_eq!(safe_match_query("& -- /"), None);
    }

    #[test]
    fn or_between_terms_is_an_operator() {
        assert_eq!(
            safe_match_query("rust or go"),
            Some("(\"rust\") OR (\"go\")".into())
        );
        assert_eq!(
            safe_match_query("rust OR go"),
            Some("(\"rust\") OR (\"go\")".into())
        );
        assert_eq!(
            safe_match_query("a -b or c d"),
            Some("(\"a\" NOT \"b\") OR (\"c\" AND \"d\")".into())
        );
        // Leading, trailing, repeated and quoted `or` stay words, as in PostgreSQL.
        assert_eq!(
            safe_match_query("or rust"),
            Some("\"or\" AND \"rust\"".into())
        );
        assert_eq!(
            safe_match_query("rust or"),
            Some("\"rust\" AND \"or\"".into())
        );
        assert_eq!(
            safe_match_query("rust or or go"),
            Some("(\"rust\") OR (\"or\" AND \"go\")".into())
        );
        assert_eq!(
            safe_match_query("\"or\" go"),
            Some("\"or\" AND \"go\"".into())
        );
        // A group of only negated terms has no FTS5 form and is dropped.
        assert_eq!(safe_match_query("a or -b"), Some("\"a\"".into()));
    }
}
