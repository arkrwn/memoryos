//! PostgreSQL implementation of `MemoryStore`. See docs/SPEC.md section 2 for the public API.

use std::time::Duration;

use memoryos_core::{Error, RequestContext, Result};
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Postgres, Transaction};

mod facts;
mod rows;
mod store_impl;
mod validate;

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
