//! The plain commands: `recall`, `candidates`, `commit` and `quarantine`.

use std::io::{IsTerminal, Read, Write};
use std::path::Path;

use chrono::{DateTime, Utc};
use memoryos_sdk::{
    Commit, CommitFact, CommitResponse, EntityRef, Error, Fact, MemoryOs, Options, RecallResponse,
    Scope, ScoredFact,
};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::config::{self, process_env};

pub fn recall(
    query: &str,
    scopes: Vec<Scope>,
    limit: Option<i64>,
    json: bool,
) -> Result<(), String> {
    call(|client| async move {
        let mut scopes = scopes.into_iter();
        let first = scopes.next().ok_or("at least one --scope is required")?;
        let mut request = client.recall(first, query);
        for scope in scopes {
            request = request.scope(scope);
        }
        if let Some(limit) = limit {
            request = request.limit(limit);
        }
        let response = request.await.map_err(describe)?;
        Ok(if json {
            recall_json(&response)
        } else {
            recall_text(&response)
        })
    })
}

pub fn candidates(
    statement: &str,
    scopes: &[Scope],
    limit: Option<i64>,
    json: bool,
) -> Result<(), String> {
    call(|client| async move {
        let facts = client
            .candidates(scopes, statement, limit)
            .await
            .map_err(describe)?;
        Ok(if json {
            json!({"facts": facts}).to_string()
        } else {
            scored_text(&facts, "no related facts")
        })
    })
}

pub fn commit(file: Option<&Path>, json: bool) -> Result<(), String> {
    let text = read_input(file)?;
    let request = parse_commit(&text)?;
    call(|client| async move {
        let response = client.commit(&request).await.map_err(describe)?;
        Ok(if json {
            commit_json(&response)
        } else {
            commit_text(&response)
        })
    })
}

pub fn quarantine_list(scopes: &[Scope], limit: Option<i64>, json: bool) -> Result<(), String> {
    call(|client| async move {
        let facts = client.quarantine(scopes, limit).await.map_err(describe)?;
        Ok(if json {
            json!({"facts": facts}).to_string()
        } else {
            quarantine_text(&facts)
        })
    })
}

pub fn approve(id: Uuid, json: bool) -> Result<(), String> {
    call(|client| async move {
        let fact = client.approve(id).await.map_err(describe)?;
        Ok(review_output("approved", &fact, json))
    })
}

pub fn reject(id: Uuid, json: bool) -> Result<(), String> {
    call(|client| async move {
        let fact = client.reject(id).await.map_err(describe)?;
        Ok(review_output("rejected", &fact, json))
    })
}

/// Runs one call and prints what it returns.
fn call<F, Fut>(work: F) -> Result<(), String>
where
    F: FnOnce(MemoryOs) -> Fut,
    Fut: std::future::Future<Output = Result<String, String>>,
{
    let client = config::client(&process_env, Options::default())?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the runtime: {e}"))?;
    let output = runtime.block_on(work(client))?;
    // A closed pipe, as with `| head`, is not a failure of the command.
    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "{output}").and_then(|()| stdout.flush());
    Ok(())
}

fn describe(error: Error) -> String {
    match error {
        Error::Transport(cause) if cause.is_timeout() => {
            "the server did not answer in time. Check MEMORYOS_URL".to_owned()
        }
        Error::Transport(_) => "cannot reach the server. Is it running at MEMORYOS_URL?".to_owned(),
        other => other.to_string(),
    }
}

fn read_input(file: Option<&Path>) -> Result<String, String> {
    match file.filter(|path| *path != Path::new("-")) {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display())),
        None => {
            let mut stdin = std::io::stdin();
            if stdin.is_terminal() {
                return Err("no commit JSON: pass a file, or pipe it to stdin".to_owned());
            }
            let mut text = String::new();
            stdin
                .read_to_string(&mut text)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            Ok(text)
        }
    }
}

/// The request as a document. Unknown fields are refused, as the server refuses them: a misspelled
/// `supersedes` would otherwise be dropped here and silently become an add.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommitDocument {
    scope: Scope,
    source: String,
    content: String,
    #[serde(default)]
    idempotency_key: Option<String>,
    facts: Vec<FactDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FactDocument {
    statement: String,
    subject: EntityDocument,
    predicate: String,
    #[serde(default)]
    object: Option<EntityDocument>,
    #[serde(default)]
    valid_from: Option<DateTime<Utc>>,
    #[serde(default)]
    supersedes: Option<Uuid>,
    #[serde(default)]
    corroborates: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EntityDocument {
    name: String,
    kind: String,
}

impl From<EntityDocument> for EntityRef {
    fn from(entity: EntityDocument) -> Self {
        Self::new(entity.name, entity.kind)
    }
}

impl From<FactDocument> for CommitFact {
    fn from(fact: FactDocument) -> Self {
        let mut out = Self::new(fact.statement, fact.subject.into(), fact.predicate);
        out.object = fact.object.map(Into::into);
        out.valid_from = fact.valid_from;
        out.supersedes = fact.supersedes;
        out.corroborates = fact.corroborates;
        out
    }
}

fn parse_commit(text: &str) -> Result<Commit, String> {
    let document: CommitDocument =
        serde_json::from_str(text).map_err(|e| format!("invalid commit JSON: {e}"))?;
    let facts = document.facts.into_iter().map(Into::into).collect();
    let mut commit = Commit::new(document.scope, document.source, document.content, facts);
    commit.idempotency_key = document.idempotency_key;
    Ok(commit)
}

fn fact_line(fact: &Fact) -> String {
    format!(
        "{}  {}  {}  {}  {}",
        fact.id,
        fact.scope,
        fact.trust.as_str(),
        fact.status.as_str(),
        fact.statement
    )
}

fn scored_text(facts: &[ScoredFact], when_empty: &str) -> String {
    if facts.is_empty() {
        return when_empty.to_owned();
    }
    facts
        .iter()
        .map(|scored| format!("{:.3}  {}", scored.score, fact_line(&scored.fact)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn recall_text(response: &RecallResponse) -> String {
    scored_text(&response.facts, "no memories found")
}

fn recall_json(response: &RecallResponse) -> String {
    json!({"facts": response.facts}).to_string()
}

fn commit_text(response: &CommitResponse) -> String {
    let mut out = format!("episode {}", response.episode_id);
    for fact in &response.facts {
        out.push_str(&format!("\n{}  {}", fact.id, fact.status.as_str()));
    }
    out
}

fn commit_json(response: &CommitResponse) -> String {
    let facts: Vec<_> = response
        .facts
        .iter()
        .map(|fact| json!({"id": fact.id, "status": fact.status}))
        .collect();
    json!({"episode_id": response.episode_id, "facts": facts}).to_string()
}

fn quarantine_text(facts: &[Fact]) -> String {
    if facts.is_empty() {
        return "no quarantined facts".to_owned();
    }
    facts.iter().map(fact_line).collect::<Vec<_>>().join("\n")
}

fn review_output(action: &str, fact: &Fact, json: bool) -> String {
    if json {
        serde_json::to_string(fact).unwrap_or_else(|_| "{}".to_owned())
    } else {
        format!("{action}: {}", fact_line(fact))
    }
}

#[cfg(test)]
mod tests {
    use memoryos_sdk::{CommittedFact, FactStatus};
    use serde_json::Value;

    use super::*;

    const MINIMAL: &str = r#"{
        "scope": "user:u", "source": "chat", "content": "text",
        "facts": [{"statement": "s", "subject": {"name": "A", "kind": "person"}, "predicate": "p"}]
    }"#;

    #[test]
    fn a_commit_document_round_trips_to_the_wire_form() {
        let text = r#"{
            "scope": "user:u", "source": "chat", "content": "text", "idempotency_key": "k",
            "facts": [{
                "statement": "s",
                "subject": {"name": "A", "kind": "person"},
                "predicate": "p",
                "object": {"name": "B", "kind": "city"},
                "valid_from": "2025-01-02T00:00:00Z",
                "supersedes": "018f0000-0000-7000-8000-000000000001"
            }]
        }"#;
        let commit = parse_commit(text).expect("parse");
        let wire = serde_json::to_value(&commit).expect("serialize");
        let expected: Value = serde_json::from_str(text).expect("json");
        assert_eq!(wire["facts"][0]["object"], expected["facts"][0]["object"]);
        assert_eq!(
            wire["facts"][0]["supersedes"],
            expected["facts"][0]["supersedes"]
        );
        assert_eq!(wire["idempotency_key"], "k");
        assert_eq!(wire["scope"], "user:u");
    }

    #[test]
    fn a_minimal_commit_document_parses() {
        assert!(parse_commit(MINIMAL).is_ok());
    }

    #[test]
    fn unknown_fields_and_bad_input_are_refused() {
        let misspelled = MINIMAL.replace("\"predicate\"", "\"supersede\": null, \"predicate\"");
        let error = parse_commit(&misspelled).expect_err("unknown field");
        assert!(error.contains("supersede"), "{error}");
        assert!(parse_commit(&MINIMAL.replace("user:u", "nope")).is_err());
        assert!(parse_commit("").is_err());
        assert!(parse_commit("{}").is_err());
    }

    #[test]
    fn commit_output_has_both_forms() {
        let response = CommitResponse {
            episode_id: Uuid::nil(),
            facts: vec![CommittedFact {
                id: Uuid::nil(),
                status: FactStatus::Quarantined,
            }],
        };
        assert!(commit_text(&response).contains("quarantined"));
        let value: Value = serde_json::from_str(&commit_json(&response)).expect("json");
        assert_eq!(value["facts"][0]["status"], "quarantined");
        assert_eq!(value["episode_id"], Uuid::nil().to_string());
    }

    #[test]
    fn empty_results_say_so() {
        assert_eq!(quarantine_text(&[]), "no quarantined facts");
        let response = RecallResponse { facts: vec![] };
        assert_eq!(recall_text(&response), "no memories found");
        let value: Value = serde_json::from_str(&recall_json(&response)).expect("json");
        assert_eq!(value, json!({"facts": []}));
    }

    #[test]
    fn a_missing_file_names_the_path() {
        let error = read_input(Some(Path::new("/nonexistent/commit.json"))).expect_err("missing");
        assert!(error.contains("/nonexistent/commit.json"));
    }
}
