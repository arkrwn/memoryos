//! SQLite row decoding and constant SQL fragments. Stored timestamps are Unix microseconds and
//! UUIDs are canonical strings; malformed persisted data is a storage error, never silently fixed.

use std::str::FromStr;

use chrono::{DateTime, Utc};
use memoryos_core::{ApiKey, Block, BlockVersion, Entity, Episode, Error, Fact, Result};
use sqlx::sqlite::SqliteRow;
use sqlx::{Decode, Row, Sqlite, Type};
use uuid::Uuid;

pub(super) const EPISODE_COLS: &str =
    "id, scope, source, trust, content, metadata, occurred_at, created_at, processed_at";
pub(super) const ENTITY_COLS: &str = "id, scope, kind, name, aliases";
pub(super) const BLOCK_COLS: &str = "id, agent_id, label, content, max_chars, version, updated_at";
pub(super) const API_KEY_COLS: &str = "id, tenant, name, scopes, max_trust, is_admin";
/// Requires the table alias `f`. Embeddings remain in their separate vec0 table.
pub(super) const FACT_COLS: &str = "f.id, f.scope, f.subject_id, f.predicate, f.object_id, \
    f.statement, f.trust, f.status, f.valid_from, f.valid_to, f.expires_at, f.recorded_at, \
    f.invalidated_at, f.superseded_by, f.pending_supersede_of, \
    (SELECT json_group_array(episode_id) FROM (\
        SELECT s.episode_id FROM fact_sources s \
        WHERE s.tenant = f.tenant AND s.fact_id = f.id ORDER BY s.episode_id\
    )) AS source_episode_ids";

/// SPEC section 4 visibility for `facts f`. Callers bind concrete transaction-stable Unix
/// microsecond timestamps at positions `t` and `k` (defaults are resolved in Rust).
pub(super) fn visible(t: u8, k: u8) -> String {
    format!(
        "f.status = 'active' \
         AND f.valid_from <= ?{t} \
         AND (f.valid_to IS NULL OR f.valid_to > ?{t}) \
         AND (f.expires_at IS NULL OR f.expires_at > ?{t}) \
         AND f.recorded_at <= ?{k} \
         AND (f.invalidated_at IS NULL OR f.invalidated_at > ?{k})"
    )
}

fn get<'r, T>(row: &'r SqliteRow, column: &str) -> Result<T>
where
    T: Decode<'r, Sqlite> + Type<Sqlite>,
{
    row.try_get(column)
        .map_err(|e| Error::Storage(e.to_string()))
}

fn parsed<T: FromStr>(row: &SqliteRow, column: &str) -> Result<T> {
    let text: String = get(row, column)?;
    text.parse()
        .map_err(|_| Error::Storage(format!("column {column} holds an invalid value")))
}

fn uuid(row: &SqliteRow, column: &str) -> Result<Uuid> {
    let text: String = get(row, column)?;
    Uuid::parse_str(&text)
        .map_err(|_| Error::Storage(format!("column {column} holds an invalid UUID")))
}

fn optional_uuid(row: &SqliteRow, column: &str) -> Result<Option<Uuid>> {
    let text: Option<String> = get(row, column)?;
    text.map(|value| {
        Uuid::parse_str(&value)
            .map_err(|_| Error::Storage(format!("column {column} holds an invalid UUID")))
    })
    .transpose()
}

pub(super) fn timestamp(value: DateTime<Utc>) -> i64 {
    value.timestamp_micros()
}

fn time(row: &SqliteRow, column: &str) -> Result<DateTime<Utc>> {
    let micros: i64 = get(row, column)?;
    DateTime::from_timestamp_micros(micros)
        .ok_or_else(|| Error::Storage(format!("column {column} holds an invalid timestamp")))
}

fn optional_time(row: &SqliteRow, column: &str) -> Result<Option<DateTime<Utc>>> {
    let micros: Option<i64> = get(row, column)?;
    micros
        .map(|value| {
            DateTime::from_timestamp_micros(value).ok_or_else(|| {
                Error::Storage(format!("column {column} holds an invalid timestamp"))
            })
        })
        .transpose()
}

fn json<T: serde::de::DeserializeOwned>(row: &SqliteRow, column: &str) -> Result<T> {
    let text: String = get(row, column)?;
    serde_json::from_str(&text)
        .map_err(|e| Error::Storage(format!("column {column} holds invalid JSON: {e}")))
}

pub(super) fn episode(row: &SqliteRow) -> Result<Episode> {
    Ok(Episode {
        id: uuid(row, "id")?,
        scope: parsed(row, "scope")?,
        source: get(row, "source")?,
        trust: parsed(row, "trust")?,
        content: get(row, "content")?,
        metadata: json(row, "metadata")?,
        occurred_at: time(row, "occurred_at")?,
        created_at: time(row, "created_at")?,
        processed_at: optional_time(row, "processed_at")?,
    })
}

pub(super) fn entity(row: &SqliteRow) -> Result<Entity> {
    Ok(Entity {
        id: uuid(row, "id")?,
        scope: parsed(row, "scope")?,
        kind: get(row, "kind")?,
        name: get(row, "name")?,
        aliases: json(row, "aliases")?,
    })
}

pub(super) fn fact(row: &SqliteRow) -> Result<Fact> {
    Ok(Fact {
        id: uuid(row, "id")?,
        scope: parsed(row, "scope")?,
        subject_id: uuid(row, "subject_id")?,
        predicate: get(row, "predicate")?,
        object_id: optional_uuid(row, "object_id")?,
        statement: get(row, "statement")?,
        trust: parsed(row, "trust")?,
        status: parsed(row, "status")?,
        valid_from: time(row, "valid_from")?,
        valid_to: optional_time(row, "valid_to")?,
        expires_at: optional_time(row, "expires_at")?,
        recorded_at: time(row, "recorded_at")?,
        invalidated_at: optional_time(row, "invalidated_at")?,
        superseded_by: optional_uuid(row, "superseded_by")?,
        pending_supersede_of: optional_uuid(row, "pending_supersede_of")?,
        source_episode_ids: json(row, "source_episode_ids")?,
    })
}

pub(super) fn block(row: &SqliteRow) -> Result<Block> {
    Ok(Block {
        id: uuid(row, "id")?,
        agent_id: get(row, "agent_id")?,
        label: get(row, "label")?,
        content: get(row, "content")?,
        max_chars: get(row, "max_chars")?,
        version: get(row, "version")?,
        updated_at: time(row, "updated_at")?,
    })
}

pub(super) fn block_version(row: &SqliteRow) -> Result<BlockVersion> {
    Ok(BlockVersion {
        version: get(row, "version")?,
        content: get(row, "content")?,
        edited_by: get(row, "edited_by")?,
        edited_at: time(row, "edited_at")?,
    })
}

pub(super) fn api_key(row: &SqliteRow) -> Result<ApiKey> {
    Ok(ApiKey {
        id: uuid(row, "id")?,
        tenant: get(row, "tenant")?,
        name: get(row, "name")?,
        scopes: json(row, "scopes")?,
        max_trust: parsed(row, "max_trust")?,
        is_admin: get(row, "is_admin")?,
    })
}
