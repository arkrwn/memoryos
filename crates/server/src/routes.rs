//! The REST surface of docs/openapi.yaml. Routes come from one table, so the router and the list
//! the contract test compares with the OpenAPI file cannot differ.

use std::time::Duration;

use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, QueryRejection};
use axum::extract::{FromRequest, FromRequestParts, Path, Query, Request, State};
use axum::http::request::Parts;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put, MethodRouter};
use axum::{Extension, Json, Router};
use chrono::{DateTime, Utc};
use memoryos_core::{FactListState, Scope};
use memoryos_ingest::CommitRequest;
use memoryos_retrieve::RecallRequest;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::app::AppState;
use crate::auth::Caller;
use crate::error::{ApiError, ApiResult};
use crate::mcp;
use crate::service::{CandidatesRequest, PutBlockRequest};

const READY_DATABASE_TIMEOUT: Duration = Duration::from_secs(2);

pub struct Route {
    pub method: &'static str,
    pub path: &'static str,
    /// Served without a token.
    pub public: bool,
    handler: MethodRouter<AppState>,
}

fn public(method: &'static str, path: &'static str, handler: MethodRouter<AppState>) -> Route {
    Route {
        method,
        path,
        public: true,
        handler,
    }
}

fn bearer(method: &'static str, path: &'static str, handler: MethodRouter<AppState>) -> Route {
    Route {
        method,
        path,
        public: false,
        handler,
    }
}

pub fn route_table() -> Vec<Route> {
    vec![
        public("GET", "/healthz", get(healthz)),
        public("GET", "/readyz", get(readyz)),
        public("GET", "/metrics", get(metrics)),
        bearer("GET", "/v1/episodes/{id}", get(get_episode)),
        bearer("POST", "/v1/episodes/{id}/purge", post(purge_episode)),
        bearer("POST", "/v1/recall", post(recall)),
        bearer("POST", "/v1/candidates", post(candidates)),
        bearer("POST", "/v1/commits", post(commit)),
        bearer("GET", "/v1/scopes", get(list_scopes)),
        bearer("GET", "/v1/facts", get(list_facts)),
        bearer("GET", "/v1/facts/{id}", get(get_fact)),
        bearer("GET", "/v1/facts/{id}/history", get(fact_history)),
        bearer("POST", "/v1/facts/{id}/invalidate", post(invalidate_fact)),
        bearer("GET", "/v1/entities/{id}", get(get_entity)),
        bearer("GET", "/v1/entities/{id}/neighbors", get(neighbors)),
        bearer("GET", "/v1/quarantine", get(list_quarantine)),
        bearer("POST", "/v1/quarantine/{id}/approve", post(approve)),
        bearer("POST", "/v1/quarantine/{id}/reject", post(reject)),
        bearer("GET", "/v1/agents/{agent_id}/blocks", get(list_blocks)),
        bearer(
            "GET",
            "/v1/agents/{agent_id}/blocks/{label}",
            get(get_block),
        ),
        bearer(
            "PUT",
            "/v1/agents/{agent_id}/blocks/{label}",
            put(put_block),
        ),
        bearer(
            "GET",
            "/v1/agents/{agent_id}/blocks/{label}/versions",
            get(block_versions),
        ),
        bearer("DELETE", "/v1/users/{user_id}/memory", delete(delete_user)),
        bearer("POST", "/mcp", post(mcp::handle)),
    ]
}

/// The method and path of every served route.
pub fn served_routes() -> Vec<(&'static str, &'static str, bool)> {
    route_table()
        .into_iter()
        .map(|route| (route.method, route.path, route.public))
        .collect()
}

pub fn router(state: &AppState) -> Router<AppState> {
    let mut open = Router::new();
    let mut protected = Router::new();
    for route in route_table() {
        if route.public {
            open = open.route(route.path, route.handler);
        } else {
            protected = protected.route(route.path, route.handler);
        }
    }
    let protected = protected.route_layer(axum::middleware::from_fn_with_state(
        state.clone(),
        crate::auth::authenticate,
    ));
    open.merge(protected)
        .fallback(|| async { ApiError::route_not_found() })
        .method_not_allowed_fallback(|| async { ApiError::method_not_allowed() })
}

// ---- Extractors -------------------------------------------------------------------------------
// Axum's own rejections answer in plain text. These keep every error in the SPEC section 9 body.

/// JSON body under the body limit. The content type is not checked: it protects nothing here,
/// since requests authenticate with a bearer token and not a cookie.
pub struct ApiJson<T>(pub T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> ApiResult<Self> {
        let bytes = read_body(request, state).await?;
        serde_json::from_slice(&bytes)
            .map(ApiJson)
            .map_err(|error| ApiError::invalid_input(format!("invalid JSON body: {error}")))
    }
}

pub async fn read_body<S: Send + Sync>(request: Request, state: &S) -> ApiResult<Bytes> {
    Bytes::from_request(request, state)
        .await
        .map_err(body_error)
}

fn body_error(rejection: BytesRejection) -> ApiError {
    if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::payload_too_large()
    } else {
        ApiError::invalid_input("the request body could not be read")
    }
}

/// `Uuid` here makes a malformed id a 400 with no store call.
pub struct ApiPath<T>(pub T);

impl<S, T> FromRequestParts<S> for ApiPath<T>
where
    S: Send + Sync,
    T: DeserializeOwned + Send,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> ApiResult<Self> {
        Path::<T>::from_request_parts(parts, state)
            .await
            .map(|Path(value)| ApiPath(value))
            .map_err(|_| ApiError::invalid_input("malformed path parameter: ids must be UUIDs"))
    }
}

pub struct ApiQuery<T>(pub T);

impl<S, T> FromRequestParts<S> for ApiQuery<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> ApiResult<Self> {
        Query::<T>::from_request_parts(parts, state)
            .await
            .map(|Query(value)| ApiQuery(value))
            .map_err(|rejection: QueryRejection| {
                ApiError::invalid_input(format!("invalid query: {}", rejection.body_text()))
            })
    }
}

/// `scopes=a,b` from a query string.
fn parse_scopes(text: &str) -> ApiResult<Vec<Scope>> {
    text.split(',')
        .map(|scope| scope.trim().parse::<Scope>().map_err(ApiError::from))
        .collect()
}

/// `scopes=user:a,user:*` from a query string. `Caller::select_scopes` validates each item.
fn parse_scope_patterns(text: &str) -> Vec<String> {
    text.split(',').map(|item| item.trim().to_owned()).collect()
}

// ---- Operations -------------------------------------------------------------------------------

async fn healthz() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn readyz(State(state): State<AppState>) -> Response {
    let ping = tokio::time::timeout(READY_DATABASE_TIMEOUT, state.services.store.ping()).await;
    let database_up = match ping {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            tracing::warn!(reason = %error, "not ready: database ping failed");
            false
        }
        Err(_) => {
            tracing::warn!(reason = "ping timed out", "not ready: database ping failed");
            false
        }
    };
    let embedder = &state.readiness;
    let ready = database_up && embedder.is_ready();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = json!({
        "status": if ready { "ready" } else { "not_ready" },
        "database": if database_up { "ok" } else { "unavailable" },
        "embedder": embedder.label(),
    });
    (status, Json(body)).into_response()
}

async fn metrics(State(state): State<AppState>) -> Response {
    match state.metrics.render() {
        Ok(text) => {
            let mut response = text.into_response();
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
            );
            response
        }
        Err(detail) => {
            tracing::error!(error = %detail, "metrics could not be rendered");
            ApiError::internal().into_response()
        }
    }
}

async fn get_episode(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.get_episode(&caller, id).await?))
}

async fn purge_episode(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.purge_episode(&caller, id).await?))
}

async fn recall(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiJson(req): ApiJson<RecallRequest>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.recall(&caller, req).await?))
}

async fn candidates(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiJson(req): ApiJson<CandidatesRequest>,
) -> ApiResult<impl IntoResponse> {
    let facts = state.services.candidates(&caller, req).await?;
    Ok(Json(json!({"facts": facts})))
}

async fn commit(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiJson(req): ApiJson<CommitRequest>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.commit(&caller, req).await?))
}

async fn list_scopes(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
) -> ApiResult<impl IntoResponse> {
    let (scopes, truncated) = state.services.list_scopes(&caller).await?;
    Ok(Json(json!({"scopes": scopes, "truncated": truncated})))
}

#[derive(Deserialize)]
struct FactsQuery {
    scope: Scope,
    state: Option<FactListState>,
    limit: Option<i64>,
    before: Option<Uuid>,
}

async fn list_facts(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiQuery(query): ApiQuery<FactsQuery>,
) -> ApiResult<impl IntoResponse> {
    let facts = state
        .services
        .list_facts(
            &caller,
            &query.scope,
            query.state,
            query.limit,
            query.before,
        )
        .await?;
    Ok(Json(json!({"facts": facts})))
}

async fn get_fact(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.get_fact(&caller, id).await?))
}

async fn fact_history(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
) -> ApiResult<impl IntoResponse> {
    let facts = state.services.fact_history(&caller, id).await?;
    Ok(Json(json!({"facts": facts})))
}

#[derive(Deserialize)]
struct InvalidateBody {
    reason: String,
}

async fn invalidate_fact(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
    ApiJson(body): ApiJson<InvalidateBody>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(
        state
            .services
            .invalidate_fact(&caller, id, &body.reason)
            .await?,
    ))
}

async fn get_entity(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.get_entity(&caller, id).await?))
}

#[derive(Deserialize)]
struct NeighborsQuery {
    scopes: String,
    depth: Option<i32>,
    as_of: Option<DateTime<Utc>>,
}

async fn neighbors(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
    ApiQuery(query): ApiQuery<NeighborsQuery>,
) -> ApiResult<impl IntoResponse> {
    let scopes = parse_scopes(&query.scopes)?;
    Ok(Json(
        state
            .services
            .neighbors(&caller, id, &scopes, query.depth, query.as_of)
            .await?,
    ))
}

#[derive(Deserialize)]
struct QuarantineQuery {
    scopes: String,
    limit: Option<i64>,
}

async fn list_quarantine(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiQuery(query): ApiQuery<QuarantineQuery>,
) -> ApiResult<impl IntoResponse> {
    let scopes = parse_scope_patterns(&query.scopes);
    let facts = state
        .services
        .list_quarantine(&caller, &scopes, query.limit)
        .await?;
    Ok(Json(json!({"facts": facts})))
}

async fn approve(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.approve(&caller, id).await?))
}

async fn reject(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(id): ApiPath<Uuid>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(state.services.reject(&caller, id).await?))
}

async fn list_blocks(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(agent_id): ApiPath<String>,
) -> ApiResult<impl IntoResponse> {
    let blocks = state.services.list_blocks(&caller, &agent_id).await?;
    Ok(Json(json!({"blocks": blocks})))
}

async fn get_block(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath((agent_id, label)): ApiPath<(String, String)>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(
        state.services.get_block(&caller, &agent_id, &label).await?,
    ))
}

async fn put_block(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath((agent_id, label)): ApiPath<(String, String)>,
    ApiJson(body): ApiJson<PutBlockRequest>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(
        state
            .services
            .put_block(&caller, &agent_id, &label, body)
            .await?,
    ))
}

async fn block_versions(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath((agent_id, label)): ApiPath<(String, String)>,
) -> ApiResult<impl IntoResponse> {
    let versions = state
        .services
        .block_versions(&caller, &agent_id, &label)
        .await?;
    Ok(Json(json!({"versions": versions})))
}

async fn delete_user(
    State(state): State<AppState>,
    Extension(caller): Extension<Caller>,
    ApiPath(user_id): ApiPath<String>,
) -> ApiResult<impl IntoResponse> {
    Ok(Json(
        state.services.delete_user_memory(&caller, &user_id).await?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_parse_from_a_comma_separated_list() {
        assert_eq!(
            parse_scopes("user:a, agent:b").expect("valid"),
            [Scope::user("a"), Scope::agent("b")]
        );
        assert!(parse_scopes("").is_err());
        assert!(parse_scopes("team:a").is_err());
    }

    #[test]
    fn every_route_is_unique() {
        let mut seen = std::collections::HashSet::new();
        for (method, path, _) in served_routes() {
            assert!(seen.insert((method, path)), "{method} {path}");
        }
    }
}
