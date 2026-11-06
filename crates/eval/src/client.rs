//! HTTP client for the parts of `docs/openapi.yaml` the suites use.
//!
//! Response types are declared here rather than imported from `memoryos-core`, so a server that
//! drifts from the published contract shows up as a decode error.

use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::{Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

use crate::dataset::{EntityRef, FactSpec};

const MAX_RATE_LIMIT_RETRIES: u32 = 8;
const BACKOFF_START: Duration = Duration::from_millis(250);
const BACKOFF_MAX: Duration = Duration::from_secs(2);
const QUARANTINE_PAGE: &str = "200";

/// A case that could not reach a verdict. Never counted as a pass.
#[derive(Debug, Error)]
pub enum CaseError {
    #[error("http: {method} {path} returned {status}: {body}")]
    Status {
        method: String,
        path: String,
        status: u16,
        body: String,
    },
    #[error("http: {method} {path} failed: {source}")]
    Transport {
        method: String,
        path: String,
        source: reqwest::Error,
    },
    #[error("contract: {method} {path} returned an unexpected body: {detail}")]
    Decode {
        method: String,
        path: String,
        detail: String,
    },
    /// A failure from an earlier phase of the suite, carried over as text.
    #[error("{0}")]
    Earlier(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Trust {
    External,
    Agent,
    User,
    System,
}

/// One request fact. The field names follow `CommitRequest` in `docs/openapi.yaml`.
#[derive(Debug, Serialize)]
pub struct CommitFact<'a> {
    pub statement: &'a str,
    pub subject: &'a EntityRef,
    pub predicate: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<Uuid>,
}

impl<'a> CommitFact<'a> {
    pub fn new(spec: &'a FactSpec, supersedes: Option<Uuid>) -> Self {
        Self {
            statement: &spec.statement,
            subject: &spec.subject,
            predicate: &spec.predicate,
            valid_from: spec.valid_from,
            supersedes,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CommitRequest<'a> {
    pub scope: &'a str,
    pub source: &'a str,
    pub content: &'a str,
    pub idempotency_key: String,
    pub facts: Vec<CommitFact<'a>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CommittedFact {
    pub id: Uuid,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CommitResponse {
    pub episode_id: Uuid,
    pub facts: Vec<CommittedFact>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Fact {
    pub id: Uuid,
    pub scope: String,
    pub subject_id: Uuid,
    pub predicate: String,
    pub statement: String,
    pub trust: Trust,
    pub status: String,
    pub valid_from: DateTime<Utc>,
    #[serde(default)]
    pub valid_to: Option<DateTime<Utc>>,
    pub recorded_at: DateTime<Utc>,
    #[serde(default)]
    pub superseded_by: Option<Uuid>,
    #[serde(default)]
    pub pending_supersede_of: Option<Uuid>,
    pub source_episode_ids: Vec<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct ScoredFact {
    pub fact: Fact,
    pub score: f64,
}

/// The body of both `POST /v1/recall` and `POST /v1/candidates`.
#[derive(Debug, Deserialize)]
pub struct RecallResponse {
    pub facts: Vec<ScoredFact>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PurgeReport {
    pub facts_deleted: u64,
    pub episodes_deleted: u64,
    pub entities_deleted: u64,
}

#[derive(Debug, Deserialize)]
struct FactList {
    facts: Vec<Fact>,
}

#[derive(Serialize)]
struct RecallBody<'a> {
    scopes: [&'a str; 1],
    query: &'a str,
    limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    as_of: Option<DateTime<Utc>>,
}

#[derive(Serialize)]
struct CandidatesBody<'a> {
    scopes: [&'a str; 1],
    statement: &'a str,
    limit: u32,
}

/// The three keys the suites need. Commit trust comes from the key, never from the request.
pub struct ApiKeys {
    /// `max_trust` of `agent`: the host. Its commits have trust `agent`.
    pub agent: String,
    /// `max_trust` of `external`: its commits have trust `external`.
    pub external: String,
    /// `is_admin`: quarantine listing, approval and purge.
    pub admin: String,
}

pub struct Client {
    http: reqwest::Client,
    base_url: String,
    keys: ApiKeys,
}

impl Client {
    pub fn new(base_url: &str, keys: ApiKeys, timeout_secs: u64) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_owned(),
            keys,
        })
    }

    /// The key whose commits carry `trust`. Only `external` is lower than the host's `agent`.
    fn key_for(&self, trust: Trust) -> &str {
        if trust == Trust::External {
            &self.keys.external
        } else {
            &self.keys.agent
        }
    }

    /// Commits as the key for `trust`. A response without one entry per request fact breaks the
    /// contract, so it is an error here and callers can rely on the lengths matching.
    pub async fn commit(
        &self,
        trust: Trust,
        request: &CommitRequest<'_>,
    ) -> Result<CommitResponse, CaseError> {
        let path = "/v1/commits";
        let body = to_body(request);
        let (status, text) = self
            .send(self.key_for(trust), Method::POST, path, &[], Some(&body))
            .await?;
        let response: CommitResponse = decode(Method::POST, path, StatusCode::OK, status, text)?;
        if response.facts.len() != request.facts.len() {
            return Err(CaseError::Decode {
                method: Method::POST.to_string(),
                path: path.to_owned(),
                detail: format!(
                    "{} facts for a request of {}",
                    response.facts.len(),
                    request.facts.len()
                ),
            });
        }
        Ok(response)
    }

    pub async fn candidates(
        &self,
        trust: Trust,
        scope: &str,
        statement: &str,
        limit: u32,
    ) -> Result<RecallResponse, CaseError> {
        let path = "/v1/candidates";
        let body = to_body(&CandidatesBody {
            scopes: [scope],
            statement,
            limit,
        });
        let (status, text) = self
            .send(self.key_for(trust), Method::POST, path, &[], Some(&body))
            .await?;
        decode(Method::POST, path, StatusCode::OK, status, text)
    }

    pub async fn recall(
        &self,
        scope: &str,
        query: &str,
        limit: u32,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<RecallResponse, CaseError> {
        let body = to_body(&RecallBody {
            scopes: [scope],
            query,
            limit,
            as_of,
        });
        let (status, text) = self
            .send(
                &self.keys.agent,
                Method::POST,
                "/v1/recall",
                &[],
                Some(&body),
            )
            .await?;
        decode(Method::POST, "/v1/recall", StatusCode::OK, status, text)
    }

    pub async fn quarantine(&self, scope: &str) -> Result<Vec<Fact>, CaseError> {
        let path = "/v1/quarantine";
        let query = [("scopes", scope), ("limit", QUARANTINE_PAGE)];
        let (status, text) = self
            .send(&self.keys.admin, Method::GET, path, &query, None)
            .await?;
        let list: FactList = decode(Method::GET, path, StatusCode::OK, status, text)?;
        Ok(list.facts)
    }

    pub async fn approve(&self, id: Uuid) -> Result<Fact, CaseError> {
        let path = format!("/v1/quarantine/{id}/approve");
        let (status, text) = self
            .send(&self.keys.admin, Method::POST, &path, &[], None)
            .await?;
        decode(Method::POST, &path, StatusCode::OK, status, text)
    }

    pub async fn purge_episode(&self, id: Uuid) -> Result<PurgeReport, CaseError> {
        let path = format!("/v1/episodes/{id}/purge");
        let (status, text) = self
            .send(&self.keys.admin, Method::POST, &path, &[], None)
            .await?;
        decode(Method::POST, &path, StatusCode::OK, status, text)
    }

    pub async fn get_fact(&self, id: Uuid) -> Result<Fact, CaseError> {
        let path = format!("/v1/facts/{id}");
        let (status, text) = self
            .send(&self.keys.agent, Method::GET, &path, &[], None)
            .await?;
        decode(Method::GET, &path, StatusCode::OK, status, text)
    }

    pub async fn episode_exists(&self, id: Uuid) -> Result<bool, CaseError> {
        self.exists(&format!("/v1/episodes/{id}")).await
    }

    pub async fn fact_exists(&self, id: Uuid) -> Result<bool, CaseError> {
        self.exists(&format!("/v1/facts/{id}")).await
    }

    async fn exists(&self, path: &str) -> Result<bool, CaseError> {
        let (status, text) = self
            .send(&self.keys.agent, Method::GET, path, &[], None)
            .await?;
        match status {
            StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            other => Err(status_error(&Method::GET, path, other, text)),
        }
    }

    /// Sends one request. A 429 is the server asking us to slow down, so it waits and retries
    /// rather than failing the case. Every other status goes back to the caller.
    async fn send(
        &self,
        token: &str,
        method: Method,
        path: &str,
        query: &[(&str, &str)],
        body: Option<&Value>,
    ) -> Result<(StatusCode, String), CaseError> {
        let url = format!("{}{path}", self.base_url);
        let mut delay = BACKOFF_START;
        let mut retries = 0;
        loop {
            let mut request = self
                .http
                .request(method.clone(), &url)
                .bearer_auth(token)
                .query(query);
            if let Some(json) = body {
                request = request.json(json);
            }
            let transport = |source| CaseError::Transport {
                method: method.to_string(),
                path: path.to_owned(),
                source,
            };
            let response = request.send().await.map_err(transport)?;
            let status = response.status();
            let text = response.text().await.map_err(transport)?;
            if status == StatusCode::TOO_MANY_REQUESTS && retries < MAX_RATE_LIMIT_RETRIES {
                retries += 1;
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(BACKOFF_MAX);
                continue;
            }
            return Ok((status, text));
        }
    }
}

/// Serializing these request types cannot fail: they hold only strings, numbers and timestamps.
fn to_body<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

fn decode<T: DeserializeOwned>(
    method: Method,
    path: &str,
    expected: StatusCode,
    status: StatusCode,
    text: String,
) -> Result<T, CaseError> {
    if status != expected {
        return Err(status_error(&method, path, status, text));
    }
    serde_json::from_str(&text).map_err(|e| CaseError::Decode {
        method: method.to_string(),
        path: path.to_owned(),
        detail: e.to_string(),
    })
}

fn status_error(method: &Method, path: &str, status: StatusCode, text: String) -> CaseError {
    CaseError::Status {
        method: method.to_string(),
        path: path.to_owned(),
        status: status.as_u16(),
        body: text.chars().take(300).collect(),
    }
}
