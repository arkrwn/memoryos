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

const MAX_REQUEST_SECS: u64 = 30;
const MAX_RATE_LIMIT_RETRIES: u32 = 8;
const POLL_START: Duration = Duration::from_millis(250);
const POLL_MAX: Duration = Duration::from_secs(2);
const QUARANTINE_PAGE: &str = "200";

/// A case that could not reach a verdict. Never counted as a pass.
#[derive(Debug, Error)]
pub enum CaseError {
    #[error("timeout: episode {episode} not processed after {secs}s")]
    NotProcessed { episode: Uuid, secs: u64 },
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Trust {
    External,
    Agent,
    User,
    System,
}

#[derive(Debug, Serialize)]
pub struct NewEpisode<'a> {
    pub scope: &'a str,
    pub source: &'a str,
    pub trust: Trust,
    pub content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occurred_at: Option<DateTime<Utc>>,
    pub metadata: Value,
    pub idempotency_key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Episode {
    pub id: Uuid,
    pub scope: String,
    pub source: String,
    pub trust: Trust,
    pub content: String,
    pub metadata: Value,
    pub occurred_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub processed_at: Option<DateTime<Utc>>,
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
    pub recorded_at: DateTime<Utc>,
    pub source_episode_ids: Vec<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct ScoredFact {
    pub fact: Fact,
    pub score: f64,
}

#[derive(Debug, Deserialize)]
pub struct ScoredEpisode {
    pub episode: Episode,
    pub score: f64,
}

#[derive(Debug, Deserialize)]
pub struct RecallResponse {
    pub facts: Vec<ScoredFact>,
    pub pending_episodes: Vec<ScoredEpisode>,
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

pub struct Client {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    admin_api_key: String,
    processing_timeout: Duration,
}

impl Client {
    pub fn new(
        base_url: &str,
        api_key: &str,
        admin_api_key: &str,
        timeout_secs: u64,
    ) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs.min(MAX_REQUEST_SECS)))
            .build()?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: api_key.to_owned(),
            admin_api_key: admin_api_key.to_owned(),
            processing_timeout: Duration::from_secs(timeout_secs),
        })
    }

    pub async fn post_episode(&self, episode: &NewEpisode<'_>) -> Result<Episode, CaseError> {
        let body = to_body(episode);
        let (status, text) = self
            .send(
                &self.api_key,
                Method::POST,
                "/v1/episodes",
                &[],
                Some(&body),
            )
            .await?;
        decode(
            Method::POST,
            "/v1/episodes",
            StatusCode::ACCEPTED,
            status,
            text,
        )
    }

    pub async fn get_episode(&self, id: Uuid) -> Result<Episode, CaseError> {
        let path = format!("/v1/episodes/{id}");
        let (status, text) = self
            .send(&self.api_key, Method::GET, &path, &[], None)
            .await?;
        decode(Method::GET, &path, StatusCode::OK, status, text)
    }

    /// Polls until `processed_at` is set. Running out of time is an error, never a result.
    pub async fn wait_processed(&self, id: Uuid) -> Result<Episode, CaseError> {
        let deadline = tokio::time::Instant::now() + self.processing_timeout;
        let mut delay = POLL_START;
        loop {
            let episode = self.get_episode(id).await?;
            if episode.processed_at.is_some() {
                return Ok(episode);
            }
            if tokio::time::Instant::now() + delay >= deadline {
                return Err(CaseError::NotProcessed {
                    episode: id,
                    secs: self.processing_timeout.as_secs(),
                });
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(POLL_MAX);
        }
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
            .send(&self.api_key, Method::POST, "/v1/recall", &[], Some(&body))
            .await?;
        decode(Method::POST, "/v1/recall", StatusCode::OK, status, text)
    }

    pub async fn quarantine(&self, scope: &str) -> Result<Vec<Fact>, CaseError> {
        let path = "/v1/quarantine";
        let query = [("scopes", scope), ("limit", QUARANTINE_PAGE)];
        let (status, text) = self
            .send(&self.admin_api_key, Method::GET, path, &query, None)
            .await?;
        let list: FactList = decode(Method::GET, path, StatusCode::OK, status, text)?;
        Ok(list.facts)
    }

    pub async fn purge_episode(&self, id: Uuid) -> Result<PurgeReport, CaseError> {
        let path = format!("/v1/episodes/{id}/purge");
        let (status, text) = self
            .send(&self.admin_api_key, Method::POST, &path, &[], None)
            .await?;
        decode(Method::POST, &path, StatusCode::OK, status, text)
    }

    pub async fn episode_exists(&self, id: Uuid) -> Result<bool, CaseError> {
        self.exists(&format!("/v1/episodes/{id}")).await
    }

    pub async fn fact_exists(&self, id: Uuid) -> Result<bool, CaseError> {
        self.exists(&format!("/v1/facts/{id}")).await
    }

    async fn exists(&self, path: &str) -> Result<bool, CaseError> {
        let (status, text) = self
            .send(&self.api_key, Method::GET, path, &[], None)
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
        let mut delay = POLL_START;
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
                delay = (delay * 2).min(POLL_MAX);
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
