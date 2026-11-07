//! Row decoding and shared column lists. Every SQL fragment here is a constant.

use std::str::FromStr;

use memoryos_core::{ApiKey, Block, BlockVersion, Entity, Episode, Error, Fact, Result};
use sqlx::postgres::PgRow;
use sqlx::{Decode, Postgres, Row, Type};

pub(crate) const EPISODE_COLS: &str =
    "id, scope, source, trust, content, metadata, occurred_at, created_at, processed_at";
pub(crate) const ENTITY_COLS: &str = "id, scope, kind, name, aliases";
pub(crate) const BLOCK_COLS: &str = "id, agent_id, label, content, max_chars, version, updated_at";
pub(crate) const API_KEY_COLS: &str = "id, tenant, name, scopes, max_trust, is_admin";
/// Requires the table alias `f`. The embedding is left out on purpose: it is never returned.
pub(crate) const FACT_COLS: &str = "f.id, f.scope, f.subject_id, f.predicate, f.object_id, \
    f.statement, f.trust, f.status, f.valid_from, f.valid_to, f.recorded_at, f.invalidated_at, \
    f.superseded_by, f.pending_supersede_of, ARRAY(SELECT s.episode_id FROM fact_sources s WHERE s.fact_id = f.id \
    ORDER BY s.episode_id) AS source_episode_ids";

/// SPEC section 4 visibility for `facts f`, with `$t` the valid time and `$k` the knowledge time.
/// Both parameters are nullable and default to the transaction's `now()`.
pub(crate) fn visible(t: u8, k: u8) -> String {
    visible_at(
        &format!("COALESCE(${t}::timestamptz, now())"),
        &format!("COALESCE(${k}::timestamptz, now())"),
    )
}

/// The same filter at the transaction's `now()`, with no parameters.
pub(crate) fn visible_now() -> String {
    visible_at("now()", "now()")
}

fn visible_at(t: &str, k: &str) -> String {
    format!(
        "f.status = 'active' \
         AND f.valid_from <= {t} \
         AND (f.valid_to IS NULL OR f.valid_to > {t}) \
         AND f.recorded_at <= {k} \
         AND (f.invalidated_at IS NULL OR f.invalidated_at > {k})"
    )
}

fn get<'r, T>(row: &'r PgRow, column: &str) -> Result<T>
where
    T: Decode<'r, Postgres> + Type<Postgres>,
{
    row.try_get(column)
        .map_err(|e| Error::Storage(e.to_string()))
}

fn parsed<T: FromStr>(row: &PgRow, column: &str) -> Result<T> {
    let text: String = get(row, column)?;
    text.parse()
        .map_err(|_| Error::Storage(format!("column {column} holds an invalid value")))
}

pub(crate) fn episode(row: &PgRow) -> Result<Episode> {
    Ok(Episode {
        id: get(row, "id")?,
        scope: parsed(row, "scope")?,
        source: get(row, "source")?,
        trust: parsed(row, "trust")?,
        content: get(row, "content")?,
        metadata: get(row, "metadata")?,
        occurred_at: get(row, "occurred_at")?,
        created_at: get(row, "created_at")?,
        processed_at: get(row, "processed_at")?,
    })
}

pub(crate) fn entity(row: &PgRow) -> Result<Entity> {
    Ok(Entity {
        id: get(row, "id")?,
        scope: parsed(row, "scope")?,
        kind: get(row, "kind")?,
        name: get(row, "name")?,
        aliases: get(row, "aliases")?,
    })
}

pub(crate) fn fact(row: &PgRow) -> Result<Fact> {
    Ok(Fact {
        id: get(row, "id")?,
        scope: parsed(row, "scope")?,
        subject_id: get(row, "subject_id")?,
        predicate: get(row, "predicate")?,
        object_id: get(row, "object_id")?,
        statement: get(row, "statement")?,
        trust: parsed(row, "trust")?,
        status: parsed(row, "status")?,
        valid_from: get(row, "valid_from")?,
        valid_to: get(row, "valid_to")?,
        recorded_at: get(row, "recorded_at")?,
        invalidated_at: get(row, "invalidated_at")?,
        superseded_by: get(row, "superseded_by")?,
        pending_supersede_of: get(row, "pending_supersede_of")?,
        source_episode_ids: get(row, "source_episode_ids")?,
    })
}

pub(crate) fn block(row: &PgRow) -> Result<Block> {
    Ok(Block {
        id: get(row, "id")?,
        agent_id: get(row, "agent_id")?,
        label: get(row, "label")?,
        content: get(row, "content")?,
        max_chars: get(row, "max_chars")?,
        version: get(row, "version")?,
        updated_at: get(row, "updated_at")?,
    })
}

pub(crate) fn block_version(row: &PgRow) -> Result<BlockVersion> {
    Ok(BlockVersion {
        version: get(row, "version")?,
        content: get(row, "content")?,
        edited_by: get(row, "edited_by")?,
        edited_at: get(row, "edited_at")?,
    })
}

pub(crate) fn api_key(row: &PgRow) -> Result<ApiKey> {
    Ok(ApiKey {
        id: get(row, "id")?,
        tenant: get(row, "tenant")?,
        name: get(row, "name")?,
        scopes: get(row, "scopes")?,
        max_trust: parsed(row, "max_trust")?,
        is_admin: get(row, "is_admin")?,
    })
}
