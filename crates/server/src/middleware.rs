//! Request id, per-request log line and metrics, and the request timeout.

use std::time::{Duration, Instant};

use axum::extract::{MatchedPath, Request, State};
use axum::http::{HeaderName, HeaderValue, Method};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use tracing::Instrument;
use uuid::Uuid;

use crate::app::AppState;
use crate::error::ApiError;
use crate::routes::TURN_ROUTE;

const REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const MAX_REQUEST_ID_CHARS: usize = 128;
const UNMATCHED_ROUTE: &str = "unmatched";

pub async fn observe(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let request_id = incoming_request_id(&request).unwrap_or_else(|| Uuid::now_v7().to_string());
    // The template, not the path: a path holds ids and would give the metric unbounded labels.
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map_or(UNMATCHED_ROUTE, MatchedPath::as_str)
        .to_owned();
    let method = method_label(request.method());
    let limit = timeout_for(&state, request.method(), &route);

    let span = tracing::info_span!("request", request_id = %request_id, method, route = %route);
    let started = Instant::now();
    let outcome = tokio::time::timeout(limit, next.run(request))
        .instrument(span.clone())
        .await;
    let mut response = outcome.unwrap_or_else(|_| ApiError::timeout().into_response());
    let elapsed = started.elapsed();

    let status = response.status().as_u16();
    state
        .metrics
        .request_finished(method, &route, status, elapsed.as_secs_f64());
    span.in_scope(|| {
        tracing::info!(
            status,
            latency_ms = elapsed.as_millis() as u64,
            "request finished"
        );
    });
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(REQUEST_ID, value);
    }
    response
}

fn timeout_for(state: &AppState, method: &Method, route: &str) -> Duration {
    if method == Method::POST && route == TURN_ROUTE {
        state.turn_timeout
    } else {
        state.request_timeout
    }
}

/// A caller's id is kept only in a form that cannot forge log fields or a header.
fn incoming_request_id(request: &Request) -> Option<String> {
    let value = request.headers().get(REQUEST_ID)?.to_str().ok()?;
    is_clean_request_id(value).then(|| value.to_owned())
}

fn is_clean_request_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_REQUEST_ID_CHARS
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// HTTP allows any token as a method, so unknown ones share a label.
fn method_label(method: &Method) -> &'static str {
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::PATCH => "PATCH",
        _ => "OTHER",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_ids_are_kept_only_when_clean() {
        assert!(is_clean_request_id("abc-123_x.y"));
        for bad in ["", "a b", "a\"b", "a\nb", &"x".repeat(129)] {
            assert!(!is_clean_request_id(bad), "{bad}");
        }
    }

    #[test]
    fn unknown_methods_share_a_label() {
        let odd = Method::from_bytes(b"BREW").expect("valid token");
        assert_eq!(method_label(&odd), "OTHER");
        assert_eq!(method_label(&Method::PUT), "PUT");
    }
}
