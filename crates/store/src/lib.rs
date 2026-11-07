//! PostgreSQL implementation of `MemoryStore`. See docs/SPEC.md section 2 for the public API.

use std::collections::BTreeMap;
use std::time::Duration;

use memoryos_core::{
    Error, Fact, FactListState, FactView, RequestContext, Result, Scope, ScopeSummary, Trust,
};
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Postgres, Row, Transaction};
use uuid::Uuid;

mod facts;
mod rows;
mod store_impl;
mod validate;

use rows::FACT_COLS;

/// Postgres-backed store. Cheap to clone: it only holds the connection pool.
#[derive(Debug, Clone)]
pub struct PgStore {
    pool: PgPool,
}

impl PgStore {
    pub async fn connect(database_url: &str, max_connections: u32) -> Result<PgStore> {
        if max_connections == 0 {
            return Err(Error::InvalidInput(
                "max_connections must be at least 1".into(),
            ));
        }
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .acquire_timeout(Duration::from_secs(10))
            .connect(database_url)
            .await
            .map_err(db)?;
        Ok(PgStore { pool })
    }

    /// `list_quarantined` for exact scopes plus scope prefixes. An empty prefix matches every scope
    /// of the tenant, so the caller must already have reduced prefixes to what its key may read.
    /// `starts_with` and not `LIKE`, because `_` is legal in a scope and would be a wildcard.
    pub async fn list_quarantined_matching(
        &self,
        ctx: &RequestContext,
        exact: &[Scope],
        prefixes: &[String],
        limit: i64,
    ) -> Result<Vec<Fact>> {
        let limit = validate::limit(limit)?;
        if exact.is_empty() && prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.status = 'quarantined' AND (f.scope = ANY($1) OR EXISTS ( \
                 SELECT 1 FROM unnest($2::text[]) AS p(prefix) WHERE starts_with(f.scope, p.prefix))) \
             ORDER BY f.recorded_at, f.id LIMIT $3"
        );
        let found = sqlx::query(&sql)
            .bind(validate::scope_strings(exact))
            .bind(prefixes)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    /// Every scope of the tenant with its counts. Scans the tenant's facts, so it is for occasional
    /// reads and not a hot path.
    pub async fn scope_summaries(&self, ctx: &RequestContext) -> Result<Vec<ScopeSummary>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let filter = rows::visible_now();
        let sql = format!(
            "SELECT f.scope, \
                 count(*) FILTER (WHERE {filter}) AS current, \
                 count(*) FILTER (WHERE f.status = 'active') AS active, \
                 count(*) FILTER (WHERE f.status = 'quarantined') AS quarantined, \
                 max(f.recorded_at) AS last_recorded_at \
             FROM facts f GROUP BY f.scope"
        );
        let facts = sqlx::query(&sql).fetch_all(&mut *tx).await.map_err(db)?;
        let entities: Vec<(String, i64)> =
            sqlx::query_as("SELECT scope, count(*) FROM entities GROUP BY scope")
                .fetch_all(&mut *tx)
                .await
                .map_err(db)?;
        let blocks: Vec<(String, i64)> =
            sqlx::query_as("SELECT 'agent:' || agent_id, count(*) FROM blocks GROUP BY agent_id")
                .fetch_all(&mut *tx)
                .await
                .map_err(db)?;
        tx.commit().await.map_err(db)?;

        fn entry<'a>(
            all: &'a mut BTreeMap<String, ScopeSummary>,
            scope: &str,
        ) -> Result<&'a mut ScopeSummary> {
            let parsed: Scope = scope.parse()?;
            Ok(all.entry(scope.to_owned()).or_insert(ScopeSummary {
                scope: parsed,
                current: 0,
                past: 0,
                quarantined: 0,
                entities: 0,
                blocks: 0,
                last_recorded_at: None,
            }))
        }
        let mut all = BTreeMap::new();
        for row in &facts {
            let scope: String = row.try_get("scope").map_err(db)?;
            let current: i64 = row.try_get("current").map_err(db)?;
            let active: i64 = row.try_get("active").map_err(db)?;
            let summary = entry(&mut all, &scope)?;
            summary.current = current;
            summary.past = active - current;
            summary.quarantined = row.try_get("quarantined").map_err(db)?;
            summary.last_recorded_at = row.try_get("last_recorded_at").map_err(db)?;
        }
        for (scope, count) in entities {
            entry(&mut all, &scope)?.entities = count;
        }
        for (scope, count) in blocks {
            entry(&mut all, &scope)?.blocks = count;
        }
        Ok(all.into_values().collect())
    }

    /// Active facts of one scope, newest recorded first. `before` is the last id of the previous
    /// page: ids are UUIDv7, so they sort by when the fact was recorded.
    pub async fn list_facts(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        state: FactListState,
        limit: i64,
        before: Option<Uuid>,
    ) -> Result<Vec<FactView>> {
        let limit = validate::limit(limit)?;
        let filter = match state {
            FactListState::Current => rows::visible_now(),
            FactListState::Past => {
                format!("f.status = 'active' AND NOT ({})", rows::visible_now())
            }
            FactListState::All => "f.status = 'active'".to_owned(),
        };
        let sql = format!(
            "SELECT {FACT_COLS}, s.name AS subject_name, s.kind AS subject_kind, \
                 o.name AS object_name, o.kind AS object_kind \
             FROM facts f \
             JOIN entities s ON s.id = f.subject_id \
             LEFT JOIN entities o ON o.id = f.object_id \
             WHERE f.scope = $1 AND ($2::uuid IS NULL OR f.id < $2) AND {filter} \
             ORDER BY f.id DESC LIMIT $3"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(&sql)
            .bind(scope.to_string())
            .bind(before)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let views = found
            .iter()
            .map(|row| {
                Ok(FactView {
                    fact: rows::fact(row)?,
                    subject_name: row.try_get("subject_name").map_err(db)?,
                    subject_kind: row.try_get("subject_kind").map_err(db)?,
                    object_name: row.try_get("object_name").map_err(db)?,
                    object_kind: row.try_get("object_kind").map_err(db)?,
                })
            })
            .collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(views)
    }

    /// The highest trust among the scope's facts and episodes, `None` if it holds neither.
    pub async fn highest_trust_in_scope(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
    ) -> Result<Option<Trust>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let trusts: Vec<String> = sqlx::query_scalar(
            "SELECT trust FROM facts WHERE scope = $1 \
             UNION SELECT trust FROM episodes WHERE scope = $1",
        )
        .bind(scope.to_string())
        .fetch_all(&mut *tx)
        .await
        .map_err(db)?;
        tx.commit().await.map_err(db)?;
        facts::highest_trust(trusts)
    }

    pub async fn migrate(&self) -> Result<()> {
        sqlx::migrate!("../../migrations")
            .run(&self.pool)
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Opens a transaction in which row-level security applies for `ctx.tenant` (SPEC section 6).
    /// Both settings are transaction-local, so the pooled connection is clean on commit, rollback
    /// or drop.
    async fn tenant_tx(&self, ctx: &RequestContext) -> Result<Transaction<'_, Postgres>> {
        if ctx.tenant.is_empty() {
            return Err(Error::InvalidInput("tenant must not be empty".into()));
        }
        let mut tx = self.pool.begin().await.map_err(db)?;
        sqlx::query("SET LOCAL ROLE memoryos_app")
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        sqlx::query("SELECT set_config('app.tenant', $1, true)")
            .bind(&ctx.tenant)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        Ok(tx)
    }
}

/// Maps a database error to a domain error. Message text only: PostgreSQL keeps row contents in
/// the DETAIL field, which this does not read.
pub(crate) fn db(e: sqlx::Error) -> Error {
    if let sqlx::Error::RowNotFound = e {
        return Error::NotFound("row".into());
    }
    if let sqlx::Error::Database(d) = &e {
        match d.code().as_deref() {
            // NUL or otherwise unstorable characters in text or jsonb.
            Some("22021" | "22P05") => {
                return Error::InvalidInput("value contains an unstorable character".into())
            }
            Some("23514") => return Error::InvalidInput(format!("check failed: {}", d.message())),
            Some("23505") => return Error::InvalidInput("value already exists".into()),
            // A referenced row was deleted concurrently.
            Some("23503") => return Error::NotFound("referenced row".into()),
            _ => {}
        }
    }
    Error::Storage(e.to_string())
}
