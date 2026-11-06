use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};

/// Authenticated caller. Every store call runs under `tenant`.
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub tenant: String,
    pub actor: String,
}

/// Who a memory belongs to. Serialized as `user:<id>`, `agent:<id>` or `shared:<id>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Scope {
    User(String),
    Agent(String),
    Shared(String),
}

impl Scope {
    pub fn user(id: impl Into<String>) -> Self {
        Self::User(id.into())
    }
    pub fn agent(id: impl Into<String>) -> Self {
        Self::Agent(id.into())
    }
    pub fn shared(id: impl Into<String>) -> Self {
        Self::Shared(id.into())
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(id) => write!(f, "user:{id}"),
            Self::Agent(id) => write!(f, "agent:{id}"),
            Self::Shared(id) => write!(f, "shared:{id}"),
        }
    }
}

impl FromStr for Scope {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let invalid = || Error::InvalidInput(format!("invalid scope: {s}"));
        let (kind, id) = s.split_once(':').ok_or_else(invalid)?;
        let valid_id = !id.is_empty()
            && id.len() <= 128
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
        if !valid_id {
            return Err(invalid());
        }
        match kind {
            "user" => Ok(Self::User(id.to_owned())),
            "agent" => Ok(Self::Agent(id.to_owned())),
            "shared" => Ok(Self::Shared(id.to_owned())),
            _ => Err(invalid()),
        }
    }
}

impl TryFrom<String> for Scope {
    type Error = Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}

impl From<Scope> for String {
    fn from(s: Scope) -> Self {
        s.to_string()
    }
}

/// Provenance trust, ordered lowest to highest. See docs/SPEC.md section 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Trust {
    External,
    Agent,
    User,
    System,
}

impl Trust {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::External => "external",
            Self::Agent => "agent",
            Self::User => "user",
            Self::System => "system",
        }
    }
}

impl FromStr for Trust {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "external" => Ok(Self::External),
            "agent" => Ok(Self::Agent),
            "user" => Ok(Self::User),
            "system" => Ok(Self::System),
            _ => Err(Error::InvalidInput(format!("invalid trust: {s}"))),
        }
    }
}

/// SPEC section 5: facts from this source are quarantined whatever they say, because `external`
/// content is untrusted anywhere and an agent cannot plant claims in user or shared memory.
pub fn source_requires_quarantine(scope: &Scope, trust: Trust) -> bool {
    match trust {
        Trust::External => true,
        Trust::Agent => !matches!(scope, Scope::Agent(_)),
        Trust::User | Trust::System => false,
    }
}

/// Key for entity identity: case-insensitive, with whitespace runs collapsed.
pub fn normalize_name(name: &str) -> String {
    name.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactStatus {
    Active,
    Quarantined,
    Rejected,
}

impl FactStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Quarantined => "quarantined",
            Self::Rejected => "rejected",
        }
    }
}

impl FromStr for FactStatus {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "active" => Ok(Self::Active),
            "quarantined" => Ok(Self::Quarantined),
            "rejected" => Ok(Self::Rejected),
            _ => Err(Error::InvalidInput(format!("invalid fact status: {s}"))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewEpisode {
    pub scope: Scope,
    pub source: String,
    pub trust: Trust,
    pub content: String,
    pub occurred_at: Option<DateTime<Utc>>,
    #[serde(default = "empty_object")]
    pub metadata: serde_json::Value,
    pub idempotency_key: Option<String>,
}

fn empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: Uuid,
    pub scope: Scope,
    pub source: String,
    pub trust: Trust,
    pub content: String,
    pub metadata: serde_json::Value,
    pub occurred_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Uuid,
    pub scope: Scope,
    pub kind: String,
    pub name: String,
    pub aliases: Vec<String>,
}

/// A fact is both a searchable statement and a graph edge.
///
/// Time semantics (docs/SPEC.md section 4):
/// - `valid_from`/`valid_to`: when the fact was true in the world. Superseding sets `valid_to`.
/// - `recorded_at`/`invalidated_at`: when we learned it, and when we retracted it as wrong.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: Uuid,
    pub scope: Scope,
    pub subject_id: Uuid,
    pub predicate: String,
    pub object_id: Option<Uuid>,
    pub statement: String,
    pub trust: Trust,
    pub status: FactStatus,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub recorded_at: DateTime<Utc>,
    pub invalidated_at: Option<DateTime<Utc>>,
    pub superseded_by: Option<Uuid>,
    /// Set on a quarantined fact: the fact that approval will close.
    pub pending_supersede_of: Option<Uuid>,
    pub source_episode_ids: Vec<Uuid>,
}

#[derive(Debug, Clone)]
pub struct NewEntity {
    pub scope: Scope,
    pub kind: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct NewFact {
    pub scope: Scope,
    pub subject_id: Uuid,
    pub predicate: String,
    pub object_id: Option<Uuid>,
    pub statement: String,
    pub embedding: Vec<f32>,
    pub trust: Trust,
    pub status: FactStatus,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
}

/// One decision of a commit, applied atomically by `MemoryStore::apply_commit`.
#[derive(Debug, Clone)]
pub enum FactOp {
    /// Insert a new fact, linked to the episode.
    Add(NewFact),
    /// Insert `new`, and close `old_fact_id` with `valid_to = new.valid_from`.
    Supersede { old_fact_id: Uuid, new: NewFact },
    /// The episode restates an existing fact: only link the episode as another source.
    Corroborate { fact_id: Uuid },
}

/// Time and visibility filter for fact reads.
#[derive(Debug, Clone, Default)]
pub struct FactFilter {
    /// Valid-time point. `None` means now.
    pub as_of: Option<DateTime<Utc>>,
    /// Knowledge-time point: only facts recorded by then and not yet retracted by then.
    pub known_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredFact {
    pub fact: Fact,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNeighborhood {
    pub root: Uuid,
    pub entities: Vec<Entity>,
    pub edges: Vec<Fact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: Uuid,
    pub agent_id: String,
    pub label: String,
    pub content: String,
    pub max_chars: i32,
    pub version: i64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockVersion {
    pub version: i64,
    pub content: String,
    pub edited_by: String,
    pub edited_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: Uuid,
    pub tenant: String,
    pub name: String,
    /// Scope strings the key may use. `*` allows all; a trailing `*` matches a prefix (`user:*`).
    pub scopes: Vec<String>,
    pub max_trust: Trust,
    pub is_admin: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PurgeReport {
    pub facts_deleted: u64,
    pub episodes_deleted: u64,
    pub entities_deleted: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_round_trips_and_rejects_bad_input() {
        let parsed: Scope = "user:u_123".parse().expect("valid scope");
        assert_eq!(parsed, Scope::user("u_123"));
        assert_eq!(parsed.to_string(), "user:u_123");
        for bad in ["user", "user:", "team:x", "user:a b", "user:a'b"] {
            assert!(bad.parse::<Scope>().is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn normalization_collapses_case_and_whitespace() {
        assert_eq!(normalize_name("  Ada \t LOVELACE "), "ada lovelace");
    }

    #[test]
    fn source_quarantine_covers_external_anywhere_and_agent_outside_its_scope() {
        for scope in [Scope::user("u"), Scope::agent("a"), Scope::shared("s")] {
            assert!(source_requires_quarantine(&scope, Trust::External));
            assert!(!source_requires_quarantine(&scope, Trust::User));
            assert!(!source_requires_quarantine(&scope, Trust::System));
        }
        assert!(source_requires_quarantine(&Scope::user("u"), Trust::Agent));
        assert!(source_requires_quarantine(
            &Scope::shared("s"),
            Trust::Agent
        ));
        assert!(!source_requires_quarantine(
            &Scope::agent("a"),
            Trust::Agent
        ));
    }

    #[test]
    fn trust_orders_low_to_high() {
        assert!(Trust::External < Trust::Agent);
        assert!(Trust::Agent < Trust::User);
        assert!(Trust::User < Trust::System);
    }
}
