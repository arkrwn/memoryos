use std::fmt;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::time::Duration;

use chrono::{DateTime, Utc};
use memoryos_core::{
    Block, BlockVersion, Entity, Episode as StoredEpisode, Fact, GraphNeighborhood, PurgeReport,
    Scope, ScoredFact, Session, SessionMessage,
};
use reqwest::header::RETRY_AFTER;
use reqwest::{Client, Method, RequestBuilder, Response, StatusCode, Url};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::error::{from_response, Error, Result};
use crate::types::{Commit, CommitResponse, Episode, PutBlock, RecallResponse, TurnResult};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(310);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_READ_RETRIES: u32 = 2;
const RETRY_BASE: Duration = Duration::from_millis(200);
const MAX_RETRY_WAIT: Duration = Duration::from_secs(10);
const DEFAULT_RECALL_LIMIT: i64 = 10;

#[derive(Debug, Clone)]
pub struct Options {
    /// Limit for one attempt of any call except a session turn.
    pub timeout: Duration,
    /// Limit for a session turn, which makes several model calls on the server.
    pub turn_timeout: Duration,
    /// Extra attempts for reads. Writes never retry.
    pub read_retries: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            turn_timeout: DEFAULT_TURN_TIMEOUT,
            read_retries: DEFAULT_READ_RETRIES,
        }
    }
}

/// Cheap to clone: clones share a connection pool.
#[derive(Clone)]
pub struct MemoryOs {
    http: Client,
    base_url: Url,
    api_key: String,
    options: Options,
}

impl fmt::Debug for MemoryOs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryOs")
            .field("base_url", &self.base_url.as_str())
            .field("api_key", &"[REDACTED]")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Retry {
    Yes,
    No,
}

impl MemoryOs {
    /// Checks that the server answers. It does not check the key, which any later call will.
    pub async fn connect(base_url: impl AsRef<str>, api_key: impl Into<String>) -> Result<Self> {
        Self::connect_with(base_url, api_key, Options::default()).await
    }

    pub async fn connect_with(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        options: Options,
    ) -> Result<Self> {
        let client = Self::new(base_url, api_key, options)?;
        client.health().await?;
        Ok(client)
    }

    /// Does no I/O.
    pub fn new(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        options: Options,
    ) -> Result<Self> {
        let base_url =
            Url::parse(base_url.as_ref()).map_err(|error| Error::InvalidUrl(error.to_string()))?;
        if base_url.cannot_be_a_base() {
            return Err(Error::InvalidUrl("not a base URL".to_owned()));
        }
        let http = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .map_err(Error::Transport)?;
        Ok(Self {
            http,
            base_url,
            api_key: api_key.into(),
            options,
        })
    }

    // ---- Service ------------------------------------------------------------------------------

    pub async fn health(&self) -> Result<()> {
        let request = self.request(Method::GET, &["healthz"]);
        self.send_empty(request, Retry::Yes).await
    }

    /// `Ok` when the server is ready for traffic. A 503 is an error carrying the status.
    pub async fn ready(&self) -> Result<()> {
        let request = self.request(Method::GET, &["readyz"]);
        let response = request.send().await?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        Err(Error::UnexpectedResponse {
            status: status.as_u16(),
            detail: "the server is not ready".to_owned(),
        })
    }

    /// The Prometheus text of `/metrics`.
    pub async fn metrics(&self) -> Result<String> {
        let request = self.request(Method::GET, &["metrics"]);
        let response = self.execute(request, Retry::Yes).await?;
        Ok(response.text().await?)
    }

    // ---- Episodes -----------------------------------------------------------------------------

    /// Stores an episode for extraction. Worker mode only.
    pub async fn remember(&self, scope: Scope, episode: Episode) -> Result<StoredEpisode> {
        let body = json!({
            "scope": scope,
            "source": episode.source,
            "trust": episode.trust,
            "content": episode.content,
            "occurred_at": episode.occurred_at,
            "metadata": episode.metadata,
            "idempotency_key": episode.idempotency_key,
        });
        let request = self.request(Method::POST, &["v1", "episodes"]).json(&body);
        self.send(request, Retry::No).await
    }

    pub async fn get_episode(&self, id: Uuid) -> Result<StoredEpisode> {
        let request = self.request(Method::GET, &["v1", "episodes", &id.to_string()]);
        self.send(request, Retry::Yes).await
    }

    /// Admin only.
    pub async fn purge_episode(&self, id: Uuid) -> Result<PurgeReport> {
        let request = self.request(Method::POST, &["v1", "episodes", &id.to_string(), "purge"]);
        self.send(request, Retry::No).await
    }

    // ---- Search and commit --------------------------------------------------------------------

    /// Await it directly, or refine first: `.as_of(t)`, `.known_at(t)`, `.limit(n)`, `.scope(s)`.
    pub fn recall(&self, scope: Scope, query: impl Into<String>) -> RecallBuilder<'_> {
        RecallBuilder {
            client: self,
            body: RecallBody {
                scopes: vec![scope],
                query: query.into(),
                limit: DEFAULT_RECALL_LIMIT,
                as_of: None,
                known_at: None,
            },
        }
    }

    /// Current facts related to a statement you may commit.
    pub async fn candidates(
        &self,
        scopes: &[Scope],
        statement: impl Into<String>,
        limit: Option<i64>,
    ) -> Result<Vec<ScoredFact>> {
        let mut body = json!({"scopes": scopes, "statement": statement.into()});
        if let Some(limit) = limit {
            body["limit"] = json!(limit);
        }
        let request = self
            .request(Method::POST, &["v1", "candidates"])
            .json(&body);
        let response: FactsScored = self.send(request, Retry::Yes).await?;
        Ok(response.facts)
    }

    pub async fn commit(&self, commit: &Commit) -> Result<CommitResponse> {
        let request = self.request(Method::POST, &["v1", "commits"]).json(commit);
        self.send(request, Retry::No).await
    }

    // ---- Facts --------------------------------------------------------------------------------

    pub async fn get_fact(&self, id: Uuid) -> Result<Fact> {
        let request = self.request(Method::GET, &["v1", "facts", &id.to_string()]);
        self.send(request, Retry::Yes).await
    }

    pub async fn fact_history(&self, id: Uuid) -> Result<Vec<Fact>> {
        let request = self.request(Method::GET, &["v1", "facts", &id.to_string(), "history"]);
        let response: Facts = self.send(request, Retry::Yes).await?;
        Ok(response.facts)
    }

    pub async fn invalidate_fact(&self, id: Uuid, reason: impl Into<String>) -> Result<Fact> {
        let request = self
            .request(
                Method::POST,
                &["v1", "facts", &id.to_string(), "invalidate"],
            )
            .json(&json!({"reason": reason.into()}));
        self.send(request, Retry::No).await
    }

    // ---- Graph --------------------------------------------------------------------------------

    pub async fn get_entity(&self, id: Uuid) -> Result<Entity> {
        let request = self.request(Method::GET, &["v1", "entities", &id.to_string()]);
        self.send(request, Retry::Yes).await
    }

    pub async fn neighbors(
        &self,
        id: Uuid,
        scopes: &[Scope],
        depth: Option<i32>,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<GraphNeighborhood> {
        let mut query = vec![("scopes", join_scopes(scopes))];
        if let Some(depth) = depth {
            query.push(("depth", depth.to_string()));
        }
        if let Some(as_of) = as_of {
            query.push(("as_of", as_of.to_rfc3339()));
        }
        let request = self
            .request(
                Method::GET,
                &["v1", "entities", &id.to_string(), "neighbors"],
            )
            .query(&query);
        self.send(request, Retry::Yes).await
    }

    // ---- Quarantine: admin only ---------------------------------------------------------------

    pub async fn quarantine(&self, scopes: &[Scope], limit: Option<i64>) -> Result<Vec<Fact>> {
        let mut query = vec![("scopes", join_scopes(scopes))];
        if let Some(limit) = limit {
            query.push(("limit", limit.to_string()));
        }
        let request = self
            .request(Method::GET, &["v1", "quarantine"])
            .query(&query);
        let response: Facts = self.send(request, Retry::Yes).await?;
        Ok(response.facts)
    }

    /// `Error::InvalidInput` asking for another review means the chain moved to a higher-trust fact
    /// while this one waited. Look at that fact, then approve again.
    pub async fn approve(&self, id: Uuid) -> Result<Fact> {
        let request = self.request(
            Method::POST,
            &["v1", "quarantine", &id.to_string(), "approve"],
        );
        self.send(request, Retry::No).await
    }

    pub async fn reject(&self, id: Uuid) -> Result<Fact> {
        let request = self.request(
            Method::POST,
            &["v1", "quarantine", &id.to_string(), "reject"],
        );
        self.send(request, Retry::No).await
    }

    // ---- Blocks -------------------------------------------------------------------------------

    pub async fn list_blocks(&self, agent_id: &str) -> Result<Vec<Block>> {
        let request = self.request(Method::GET, &["v1", "agents", agent_id, "blocks"]);
        let response: Blocks = self.send(request, Retry::Yes).await?;
        Ok(response.blocks)
    }

    pub async fn get_block(&self, agent_id: &str, label: &str) -> Result<Block> {
        let request = self.request(Method::GET, &["v1", "agents", agent_id, "blocks", label]);
        self.send(request, Retry::Yes).await
    }

    pub async fn put_block(&self, agent_id: &str, label: &str, block: &PutBlock) -> Result<Block> {
        let request = self
            .request(Method::PUT, &["v1", "agents", agent_id, "blocks", label])
            .json(block);
        self.send(request, Retry::No).await
    }

    pub async fn block_versions(&self, agent_id: &str, label: &str) -> Result<Vec<BlockVersion>> {
        let request = self.request(
            Method::GET,
            &["v1", "agents", agent_id, "blocks", label, "versions"],
        );
        let response: Versions = self.send(request, Retry::Yes).await?;
        Ok(response.versions)
    }

    // ---- Sessions: worker mode only -----------------------------------------------------------

    pub async fn create_session(&self, agent_id: &str, scope: Scope) -> Result<Session> {
        let request = self
            .request(Method::POST, &["v1", "agents", agent_id, "sessions"])
            .json(&json!({"scope": scope}));
        self.send(request, Retry::No).await
    }

    pub async fn session_messages(&self, session_id: Uuid) -> Result<Vec<SessionMessage>> {
        let request = self.request(
            Method::GET,
            &["v1", "sessions", &session_id.to_string(), "messages"],
        );
        let response: Messages = self.send(request, Retry::Yes).await?;
        Ok(response.messages)
    }

    /// Runs one agent turn. Takes as long as the model does, up to `Options::turn_timeout`.
    pub async fn send_message(
        &self,
        session_id: Uuid,
        text: impl Into<String>,
    ) -> Result<TurnResult> {
        let request = self
            .request(
                Method::POST,
                &["v1", "sessions", &session_id.to_string(), "messages"],
            )
            .json(&json!({"text": text.into()}))
            .timeout(self.options.turn_timeout);
        self.send(request, Retry::No).await
    }

    // ---- Users --------------------------------------------------------------------------------

    /// Hard-deletes everything under `user:{user_id}`. Admin only.
    pub async fn delete_user_memory(&self, user_id: &str) -> Result<PurgeReport> {
        let request = self.request(Method::DELETE, &["v1", "users", user_id, "memory"]);
        self.send(request, Retry::No).await
    }

    // ---- Transport ----------------------------------------------------------------------------

    fn url(&self, segments: &[&str]) -> Url {
        let mut url = self.base_url.clone();
        if let Ok(mut path) = url.path_segments_mut() {
            path.pop_if_empty().extend(segments);
        }
        url
    }

    fn request(&self, method: Method, segments: &[&str]) -> RequestBuilder {
        self.http
            .request(method, self.url(segments))
            .bearer_auth(&self.api_key)
            .timeout(self.options.timeout)
    }

    async fn send<T: DeserializeOwned>(&self, request: RequestBuilder, retry: Retry) -> Result<T> {
        let response = self.execute(request, retry).await?;
        let status = response.status();
        let bytes = response.bytes().await?;
        serde_json::from_slice(&bytes).map_err(|error| Error::UnexpectedResponse {
            status: status.as_u16(),
            detail: format!("the body does not match the API: {error}"),
        })
    }

    async fn send_empty(&self, request: RequestBuilder, retry: Retry) -> Result<()> {
        self.execute(request, retry).await.map(drop)
    }

    async fn execute(&self, request: RequestBuilder, retry: Retry) -> Result<Response> {
        let retries = match retry {
            Retry::Yes => self.options.read_retries,
            Retry::No => 0,
        };
        let mut attempt = 0;
        loop {
            let outcome = match request.try_clone() {
                Some(copy) => attempt_once(copy).await,
                // Only a streaming body cannot be cloned, and this client sends none.
                None => return attempt_once(request).await,
            };
            match outcome {
                Err(error) if attempt < retries && error.is_retryable() => {
                    let wait = error
                        .retry_after()
                        .unwrap_or(RETRY_BASE * 2u32.saturating_pow(attempt))
                        .min(MAX_RETRY_WAIT);
                    tokio::time::sleep(wait).await;
                    attempt += 1;
                }
                other => return other,
            }
        }
    }
}

async fn attempt_once(request: RequestBuilder) -> Result<Response> {
    let response = request.send().await?;
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let retry_after = retry_after_of(&response, status);
    let body = response.bytes().await?;
    Err(from_response(status, retry_after, &body))
}

fn retry_after_of(response: &Response, status: StatusCode) -> Option<Duration> {
    if status != StatusCode::TOO_MANY_REQUESTS {
        return None;
    }
    let seconds = response.headers().get(RETRY_AFTER)?.to_str().ok()?;
    seconds.trim().parse().ok().map(Duration::from_secs)
}

fn join_scopes(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Serialize)]
struct RecallBody {
    scopes: Vec<Scope>,
    query: String,
    limit: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    as_of: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    known_at: Option<DateTime<Utc>>,
}

/// A recall that has not been sent. Await it to send.
pub struct RecallBuilder<'a> {
    client: &'a MemoryOs,
    body: RecallBody,
}

impl RecallBuilder<'_> {
    /// What was true at this time.
    pub fn as_of(mut self, at: DateTime<Utc>) -> Self {
        self.body.as_of = Some(at);
        self
    }

    /// What MemoryOS knew at this time.
    pub fn known_at(mut self, at: DateTime<Utc>) -> Self {
        self.body.known_at = Some(at);
        self
    }

    pub fn limit(mut self, limit: i64) -> Self {
        self.body.limit = limit;
        self
    }

    /// Searches this scope as well.
    pub fn scope(mut self, scope: Scope) -> Self {
        self.body.scopes.push(scope);
        self
    }
}

impl<'a> IntoFuture for RecallBuilder<'a> {
    type Output = Result<RecallResponse>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let request = self
                .client
                .request(Method::POST, &["v1", "recall"])
                .json(&self.body);
            // A POST, but a read: it changes nothing, so it can be retried.
            self.client.send(request, Retry::Yes).await
        })
    }
}

#[derive(Deserialize)]
struct Facts {
    facts: Vec<Fact>,
}

#[derive(Deserialize)]
struct FactsScored {
    facts: Vec<ScoredFact>,
}

#[derive(Deserialize)]
struct Blocks {
    blocks: Vec<Block>,
}

#[derive(Deserialize)]
struct Versions {
    versions: Vec<BlockVersion>,
}

#[derive(Deserialize)]
struct Messages {
    messages: Vec<SessionMessage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client(base: &str) -> MemoryOs {
        MemoryOs::new(base, "mos_secret", Options::default()).expect("client")
    }

    #[test]
    fn urls_keep_a_base_path_and_encode_segments() {
        assert_eq!(
            client("http://h:1/").url(&["v1", "recall"]).as_str(),
            "http://h:1/v1/recall"
        );
        assert_eq!(
            client("http://h:1/base/").url(&["v1", "recall"]).as_str(),
            "http://h:1/base/v1/recall"
        );
        assert_eq!(
            client("http://h:1").url(&["v1", "a/b c"]).as_str(),
            "http://h:1/v1/a%2Fb%20c"
        );
    }

    #[test]
    fn debug_output_hides_the_key() {
        assert!(!format!("{:?}", client("http://h:1")).contains("mos_secret"));
    }

    #[test]
    fn a_bad_base_url_is_an_error() {
        assert!(MemoryOs::new("not a url", "k", Options::default()).is_err());
        assert!(MemoryOs::new("mailto:a@b", "k", Options::default()).is_err());
    }

    #[test]
    fn recall_serializes_only_what_is_set() {
        let body = RecallBody {
            scopes: vec![Scope::user("u")],
            query: "q".into(),
            limit: 10,
            as_of: None,
            known_at: None,
        };
        assert_eq!(
            serde_json::to_value(&body).expect("serializes"),
            json!({"scopes": ["user:u"], "query": "q", "limit": 10})
        );
    }

    #[tokio::test]
    async fn a_write_is_not_retried_and_a_read_is() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        // Answers every connection with a 503, which a read retries and a write does not.
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let accepted = Arc::new(AtomicUsize::new(0));
        let counter = accepted.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                // Closing with the request unread would reset the connection instead of answering.
                let mut request = [0u8; 4096];
                let _ = socket.read(&mut request).await;
                let reply = "HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\
                             connection: close\r\n\r\n";
                let _ = socket.write_all(reply.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        let options = Options {
            read_retries: 2,
            ..Options::default()
        };
        let mem = MemoryOs::new(format!("http://{address}"), "k", options).expect("client");

        let id = Uuid::now_v7();
        assert!(mem.get_fact(id).await.is_err());
        assert_eq!(
            accepted.swap(0, Ordering::SeqCst),
            3,
            "a read tries 3 times"
        );

        assert!(mem.invalidate_fact(id, "r").await.is_err());
        assert_eq!(accepted.swap(0, Ordering::SeqCst), 1, "a write tries once");
    }
}
