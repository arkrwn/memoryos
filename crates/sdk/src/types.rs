//! Request and response types the API defines beyond `memoryos-core`. Declared here so the SDK
//! depends on nothing server-side.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use memoryos_core::{FactStatus, Scope, ScoredFact};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Midnight UTC on this date, for `.as_of()`.
///
/// # Panics
/// On a date that does not exist, such as month 13: this is for dates written in code.
pub fn date(year: i32, month: u32, day: u32) -> DateTime<Utc> {
    let naive = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("a valid calendar date");
    Utc.from_utc_datetime(&naive)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityRef {
    pub name: String,
    pub kind: String,
}

impl EntityRef {
    pub fn new(name: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: kind.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CommitFact {
    pub statement: String,
    pub subject: EntityRef,
    pub predicate: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<EntityRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corroborates: Option<Uuid>,
}

impl CommitFact {
    pub fn new(
        statement: impl Into<String>,
        subject: EntityRef,
        predicate: impl Into<String>,
    ) -> Self {
        Self {
            statement: statement.into(),
            subject,
            predicate: predicate.into(),
            object: None,
            valid_from: None,
            supersedes: None,
            corroborates: None,
        }
    }

    pub fn object(mut self, object: EntityRef) -> Self {
        self.object = Some(object);
        self
    }

    pub fn valid_from(mut self, at: DateTime<Utc>) -> Self {
        self.valid_from = Some(at);
        self
    }

    pub fn supersedes(mut self, fact_id: Uuid) -> Self {
        self.supersedes = Some(fact_id);
        self
    }

    pub fn corroborates(mut self, fact_id: Uuid) -> Self {
        self.corroborates = Some(fact_id);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Commit {
    pub scope: Scope,
    pub source: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    pub facts: Vec<CommitFact>,
}

impl Commit {
    pub fn new(
        scope: Scope,
        source: impl Into<String>,
        content: impl Into<String>,
        facts: Vec<CommitFact>,
    ) -> Self {
        Self {
            scope,
            source: source.into(),
            content: content.into(),
            idempotency_key: None,
            facts,
        }
    }

    pub fn idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CommitResponse {
    pub episode_id: Uuid,
    pub facts: Vec<CommittedFact>,
    /// The server masked a secret in a statement, an entity name or the content before storing it.
    /// False when absent, as from an older server.
    #[serde(default)]
    pub redacted: bool,
}

/// The fields after `status` default when absent, so a response from an older server still parses.
#[derive(Debug, Clone, Deserialize)]
pub struct CommittedFact {
    pub id: Uuid,
    pub status: FactStatus,
    /// No fact was created: `id` is an existing fact that gained this source.
    #[serde(default)]
    pub corroborated: bool,
    /// For a created fact, up to three current facts of the scope that resemble it, best first.
    #[serde(default)]
    pub related: Vec<RelatedFact>,
    /// With `corroborated`: the fact this request superseded, now closed with `id` as successor.
    #[serde(default)]
    pub superseded: Option<Uuid>,
}

/// A current fact that resembles a created fact without repeating it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RelatedFact {
    pub id: Uuid,
    pub statement: String,
    pub score: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecallResponse {
    pub facts: Vec<ScoredFact>,
}

/// Input to `MemoryOs::put_block`.
#[derive(Debug, Clone, Serialize)]
pub struct PutBlock {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<i32>,
    pub expected_version: Option<i64>,
}

impl PutBlock {
    pub fn create(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            max_chars: None,
            expected_version: None,
        }
    }

    pub fn update(content: impl Into<String>, expected_version: i64) -> Self {
        Self {
            content: content.into(),
            max_chars: None,
            expected_version: Some(expected_version),
        }
    }

    pub fn max_chars(mut self, max_chars: i32) -> Self {
        self.max_chars = Some(max_chars);
        self
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn date_is_midnight_utc() {
        assert_eq!(date(2025, 1, 2).to_rfc3339(), "2025-01-02T00:00:00+00:00");
    }

    #[test]
    fn commit_omits_what_is_unset() {
        let commit = Commit::new(
            Scope::user("u"),
            "chat",
            "text",
            vec![CommitFact::new(
                "Ada likes tea",
                EntityRef::new("Ada", "person"),
                "likes",
            )],
        );
        assert_eq!(
            serde_json::to_value(&commit).expect("serializes"),
            json!({
                "scope": "user:u",
                "source": "chat",
                "content": "text",
                "facts": [{
                    "statement": "Ada likes tea",
                    "subject": {"name": "Ada", "kind": "person"},
                    "predicate": "likes",
                }],
            })
        );
    }

    #[test]
    fn committed_fact_parses_old_and_new_responses() {
        let id = Uuid::now_v7();
        let old: CommittedFact =
            serde_json::from_value(json!({"id": id, "status": "active"})).expect("old shape");
        assert!(!old.corroborated && old.related.is_empty() && old.superseded.is_none());

        let new: CommittedFact = serde_json::from_value(json!({
            "id": id,
            "status": "active",
            "corroborated": true,
            "superseded": id,
            "related": [{"id": id, "statement": "s", "score": 0.5}],
        }))
        .expect("new shape");
        assert!(new.corroborated);
        assert_eq!(new.superseded, Some(id));
        assert_eq!(
            new.related,
            [RelatedFact {
                id,
                statement: "s".into(),
                score: 0.5,
            }]
        );
    }
}
