//! SQLite implementation of `MemoryStore`, with explicit tenant predicates in place of PostgreSQL
//! row-level security. sqlite-vec is statically registered; no dynamic extension loading is used.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::OnceLock;
use std::time::Duration;

use chrono::{DateTime, Utc};
use memoryos_core::{
    Error, Fact, FactListState, FactView, RequestContext, Result, Scope, ScopeSummary, Trust,
    EMBEDDING_DIMS,
};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{AssertSqlSafe, Row, Sqlite, SqlitePool, Transaction};
use uuid::Uuid;

mod facts;
mod rows;
mod store_impl;

use rows::FACT_COLS;

/// SQLite-backed store. A single pooled connection serializes writers within this process.
#[derive(Debug, Clone)]
pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    /// Opens a file database. In-memory databases are refused: `migrate` and `serve` run as
    /// separate processes, so `serve` would start on an empty database with no tables.
    pub async fn connect(database_url: &str) -> Result<Self> {
        if is_memory_url(database_url) {
            return Err(Error::InvalidInput(
                "SQLite needs a database file; in-memory databases are not supported".into(),
            ));
        }
        register_sqlite_vec()?;
        let options = SqliteConnectOptions::from_str(database_url)
            .map_err(db)?
            .create_if_missing(true)
            .foreign_keys(true)
            .busy_timeout(Duration::from_secs(10))
            .journal_mode(SqliteJournalMode::Wal);
        prepare_database_file(&options)?;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(10))
            .connect_with(options.clone())
            .await
            .map_err(db)?;
        // Fail at connect, not on the first vector write or search, if registration was ineffective.
        let _: String = sqlx::query_scalar("SELECT vec_version()")
            .fetch_one(&pool)
            .await
            .map_err(db)?;
        restrict_database_files(&options)?;
        Ok(Self { pool })
    }

    pub async fn migrate(&self) -> Result<()> {
        sqlx::migrate!("../../migrations/sqlite")
            .run(&self.pool)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    /// `list_quarantined` for exact scopes plus literal scope prefixes. An empty prefix matches all
    /// scopes in this tenant; authorization has already reduced the prefixes before this call.
    pub async fn list_quarantined_matching(
        &self,
        ctx: &RequestContext,
        exact: &[Scope],
        prefixes: &[String],
        limit: i64,
    ) -> Result<Vec<Fact>> {
        let limit = crate::validate::limit(limit)?;
        if exact.is_empty() && prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let exact = json_array(&crate::validate::scope_strings(exact))?;
        let prefixes = json_array(&prefixes)?;
        let mut tx = self.tenant_tx(ctx).await?;
        let sql = format!(
            "SELECT {FACT_COLS} FROM facts f \
             WHERE f.tenant = ?1 AND f.status = 'quarantined' \
               AND (f.scope IN (SELECT value FROM json_each(?2)) OR EXISTS (\
                   SELECT 1 FROM json_each(?3) p \
                   WHERE substr(f.scope, 1, length(p.value)) = p.value)) \
             ORDER BY f.recorded_at, f.id LIMIT ?4"
        );
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(exact)
            .bind(prefixes)
            .bind(limit)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let result = found.iter().map(rows::fact).collect::<Result<_>>()?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    /// Every scope of the tenant with its counts. This intentionally scans tenant-local rows and is
    /// not a hot-path query.
    pub async fn scope_summaries(&self, ctx: &RequestContext) -> Result<Vec<ScopeSummary>> {
        let now = rows::timestamp(Utc::now());
        let filter = rows::visible(2, 2);
        let sql = format!(
            "SELECT f.scope, \
                 SUM(CASE WHEN {filter} THEN 1 ELSE 0 END) AS current, \
                 SUM(CASE WHEN f.status = 'active' THEN 1 ELSE 0 END) AS active, \
                 SUM(CASE WHEN f.status = 'quarantined' THEN 1 ELSE 0 END) AS quarantined, \
                 MAX(f.recorded_at) AS last_recorded_at \
             FROM facts f WHERE f.tenant = ?1 GROUP BY f.scope"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let fact_rows = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(now)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?;
        let entities: Vec<(String, i64)> =
            sqlx::query_as("SELECT scope, count(*) FROM entities WHERE tenant = ?1 GROUP BY scope")
                .bind(&ctx.tenant)
                .fetch_all(&mut *tx)
                .await
                .map_err(db)?;
        let blocks: Vec<(String, i64)> = sqlx::query_as(
            "SELECT 'agent:' || agent_id, count(*) FROM blocks WHERE tenant = ?1 GROUP BY agent_id",
        )
        .bind(&ctx.tenant)
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
        for row in &fact_rows {
            let scope: String = row.try_get("scope").map_err(db)?;
            let current: i64 = row.try_get("current").map_err(db)?;
            let active: i64 = row.try_get("active").map_err(db)?;
            let summary = entry(&mut all, &scope)?;
            summary.current = current;
            summary.past = active - current;
            summary.quarantined = row.try_get("quarantined").map_err(db)?;
            let last: Option<i64> = row.try_get("last_recorded_at").map_err(db)?;
            summary.last_recorded_at = last
                .map(|value| {
                    DateTime::from_timestamp_micros(value)
                        .ok_or_else(|| Error::Storage("invalid last_recorded_at timestamp".into()))
                })
                .transpose()?;
        }
        for (scope, count) in entities {
            entry(&mut all, &scope)?.entities = count;
        }
        for (scope, count) in blocks {
            entry(&mut all, &scope)?.blocks = count;
        }
        Ok(all.into_values().collect())
    }

    /// Active facts in one scope, newest UUIDv7 first. `before` is the previous page's last id.
    pub async fn list_facts(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
        state: FactListState,
        limit: i64,
        before: Option<Uuid>,
    ) -> Result<Vec<FactView>> {
        let limit = crate::validate::limit(limit)?;
        let now = rows::timestamp(Utc::now());
        let filter = match state {
            FactListState::Current => rows::visible(5, 5),
            FactListState::Past => format!("f.status = 'active' AND NOT ({})", rows::visible(5, 5)),
            FactListState::All => "f.status = 'active'".to_owned(),
        };
        let sql = format!(
            "SELECT {FACT_COLS}, s.name AS subject_name, s.kind AS subject_kind, \
                 o.name AS object_name, o.kind AS object_kind \
             FROM facts f \
             JOIN entities s ON s.tenant = f.tenant AND s.id = f.subject_id \
             LEFT JOIN entities o ON o.tenant = f.tenant AND o.id = f.object_id \
             WHERE f.tenant = ?1 AND f.scope = ?2 AND (?3 IS NULL OR f.id < ?3) AND {filter} \
             ORDER BY f.id DESC LIMIT ?4"
        );
        let mut tx = self.tenant_tx(ctx).await?;
        let found = sqlx::query(AssertSqlSafe(sql))
            .bind(&ctx.tenant)
            .bind(scope.to_string())
            .bind(before.map(|id| id.to_string()))
            .bind(limit)
            .bind(now)
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

    /// Highest trust among the scope's facts and episodes, or `None` when it holds neither.
    pub async fn highest_trust_in_scope(
        &self,
        ctx: &RequestContext,
        scope: &Scope,
    ) -> Result<Option<Trust>> {
        let mut tx = self.tenant_tx(ctx).await?;
        let trusts: Vec<String> = sqlx::query_scalar(
            "SELECT trust FROM facts WHERE tenant = ?1 AND scope = ?2 \
             UNION SELECT trust FROM episodes WHERE tenant = ?1 AND scope = ?2",
        )
        .bind(&ctx.tenant)
        .bind(scope.to_string())
        .fetch_all(&mut *tx)
        .await
        .map_err(db)?;
        tx.commit().await.map_err(db)?;
        facts::highest_trust(trusts)
    }

    async fn tenant_tx(&self, ctx: &RequestContext) -> Result<Transaction<'_, Sqlite>> {
        if ctx.tenant.is_empty() {
            return Err(Error::InvalidInput("tenant must not be empty".into()));
        }
        self.pool.begin().await.map_err(db)
    }

    /// IMMEDIATE takes SQLite's write lock before a read can turn into a write. Together with the
    /// busy timeout this avoids deferred-transaction upgrade failures and replaces row-lock waits.
    async fn write_tx(&self, ctx: &RequestContext) -> Result<Transaction<'_, Sqlite>> {
        if ctx.tenant.is_empty() {
            return Err(Error::InvalidInput("tenant must not be empty".into()));
        }
        self.pool.begin_with("BEGIN IMMEDIATE").await.map_err(db)
    }

    async fn any_write_tx(&self) -> Result<Transaction<'_, Sqlite>> {
        self.pool.begin_with("BEGIN IMMEDIATE").await.map_err(db)
    }
}

#[cfg(unix)]
fn prepare_database_file(options: &SqliteConnectOptions) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};

    let path = options.get_filename();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        if !parent.exists() {
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true).mode(0o700);
            builder
                .create(parent)
                .map_err(|e| Error::Storage(format!("could not create SQLite directory: {e}")))?;
        }
    }
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
        .map_err(|e| Error::Storage(format!("could not create SQLite database: {e}")))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| Error::Storage(format!("could not restrict SQLite database: {e}")))
}

#[cfg(not(unix))]
fn prepare_database_file(_options: &SqliteConnectOptions) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn restrict_database_files(options: &SqliteConnectOptions) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let path = options.get_filename();
    for suffix in ["", "-wal", "-shm"] {
        let mut name = path.as_os_str().to_os_string();
        name.push(suffix);
        let candidate = std::path::PathBuf::from(name);
        match std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o600)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::Storage(format!(
                    "could not restrict SQLite database files: {e}"
                )))
            }
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn restrict_database_files(_options: &SqliteConnectOptions) -> Result<()> {
    Ok(())
}

/// `sqlite::memory:` and its spellings, and any URL opened with `mode=memory`.
fn is_memory_url(database_url: &str) -> bool {
    let lower = database_url.trim().to_ascii_lowercase();
    lower.contains("mode=memory")
        || matches!(
            lower.as_str(),
            ":memory:" | "sqlite::memory:" | "sqlite://:memory:"
        )
}

pub(super) fn json_array<T: serde::Serialize + ?Sized>(values: &T) -> Result<String> {
    serde_json::to_string(values).map_err(|e| Error::Internal(e.to_string()))
}

pub(super) fn uuid_strings(ids: impl IntoIterator<Item = Uuid>) -> Vec<String> {
    ids.into_iter().map(|id| id.to_string()).collect()
}

/// Local replacement for pgvector validation and encoding. sqlite-vec float32 BLOBs are
/// little-endian IEEE-754 values; zero and non-finite vectors have no valid cosine distance.
pub(super) fn embedding(values: &[f32]) -> Result<Vec<u8>> {
    if values.len() != EMBEDDING_DIMS {
        return Err(Error::InvalidInput(format!(
            "embedding has {} dimensions, expected {EMBEDDING_DIMS}",
            values.len()
        )));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "embedding has a non-finite value".into(),
        ));
    }
    if values.iter().all(|v| *v == 0.0) {
        return Err(Error::InvalidInput("embedding is the zero vector".into()));
    }
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub(super) fn storable(value: &str) -> Result<()> {
    if value.contains('\0') {
        Err(Error::InvalidInput(
            "value contains an unstorable character".into(),
        ))
    } else {
        Ok(())
    }
}

/// Maps SQLite errors without exposing bound row contents. Constraint categories are stable across
/// SQLite builds, unlike message text and extended result codes.
pub(super) fn db(e: sqlx::Error) -> Error {
    if matches!(e, sqlx::Error::RowNotFound) {
        return Error::NotFound("row".into());
    }
    if let sqlx::Error::Database(d) = &e {
        match d.kind() {
            sqlx::error::ErrorKind::UniqueViolation => {
                return Error::InvalidInput("value already exists".into())
            }
            sqlx::error::ErrorKind::ForeignKeyViolation => {
                return Error::NotFound("referenced row".into())
            }
            sqlx::error::ErrorKind::CheckViolation | sqlx::error::ErrorKind::NotNullViolation => {
                return Error::InvalidInput(format!("check failed: {}", d.message()))
            }
            _ => {}
        }
    }
    Error::Storage(e.to_string())
}

static VEC_REGISTRATION: OnceLock<std::result::Result<(), String>> = OnceLock::new();

#[allow(unsafe_code)]
fn register_sqlite_vec() -> Result<()> {
    let result = VEC_REGISTRATION.get_or_init(|| {
        // SAFETY: `sqlite3_vec_init` is the extension entry point compiled and linked by sqlite-vec.
        // SQLite requires it cast to its full initialization ABI for process-wide auto-registration.
        // The function pointer remains valid for the process lifetime, registration occurs once, and
        // SQLx opens connections only after this call returns. No database handle or borrowed memory
        // crosses the FFI boundary here.
        let rc = unsafe {
            libsqlite3_sys::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut libsqlite3_sys::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const libsqlite3_sys::sqlite3_api_routines,
                ) -> i32,
            >(
                sqlite_vec::sqlite3_vec_init as *const (),
            )))
        };
        if rc == libsqlite3_sys::SQLITE_OK {
            Ok(())
        } else {
            Err(format!(
                "sqlite-vec auto-extension registration failed: {rc}"
            ))
        }
    });
    result.clone().map_err(Error::Storage)
}
