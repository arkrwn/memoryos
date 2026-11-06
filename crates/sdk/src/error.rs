use std::time::Duration;

use reqwest::StatusCode;
use serde::Deserialize;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// One variant per error code of SPEC section 9, plus failures that never reached a server answer.
#[derive(Debug, Error)]
pub enum Error {
    #[error("unauthorized: {message}")]
    Unauthorized { message: String },
    #[error("forbidden: {message}")]
    Forbidden { message: String },
    #[error("not found: {message}")]
    NotFound { message: String },
    #[error("invalid input: {message}")]
    InvalidInput { message: String },
    #[error("version conflict: {message}")]
    VersionConflict { message: String },
    #[error("block too large: {message}")]
    BlockTooLarge { message: String },
    #[error("rate limited: {message}")]
    RateLimited {
        message: String,
        retry_after: Option<Duration>,
    },
    #[error("provider error: {message}")]
    Provider { message: String },
    #[error("server error: {message}")]
    Internal { message: String },
    /// An error code this version of the SDK does not know.
    #[error("{code} (HTTP {status}): {message}")]
    Api {
        status: u16,
        code: String,
        message: String,
    },
    /// The answer was not the documented shape, for example from a proxy in front of the server.
    #[error("unexpected response (HTTP {status}): {detail}")]
    UnexpectedResponse { status: u16, detail: String },
    /// No answer: connection failure or timeout.
    #[error("transport: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("invalid base URL: {0}")]
    InvalidUrl(String),
}

impl Error {
    /// True when calling again later can help without changing the request.
    pub(crate) fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimited { .. } => true,
            // Only reads retry, so a connection lost mid-request is safe to try again.
            Self::Transport(error) => {
                error.is_connect() || error.is_timeout() || error.is_request()
            }
            Self::Api { status, .. } | Self::UnexpectedResponse { status, .. } => {
                matches!(status, 502..=504)
            }
            Self::Provider { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

#[derive(Deserialize)]
struct Envelope {
    error: Body,
}

#[derive(Deserialize)]
struct Body {
    code: String,
    message: String,
}

pub(crate) fn from_response(
    status: StatusCode,
    retry_after: Option<Duration>,
    body: &[u8],
) -> Error {
    let Ok(Envelope { error }) = serde_json::from_slice(body) else {
        return Error::UnexpectedResponse {
            status: status.as_u16(),
            detail: "the body is not a MemoryOS error".to_owned(),
        };
    };
    let message = error.message;
    match error.code.as_str() {
        "unauthorized" => Error::Unauthorized { message },
        "forbidden" => Error::Forbidden { message },
        "not_found" => Error::NotFound { message },
        "invalid_input" => Error::InvalidInput { message },
        "version_conflict" => Error::VersionConflict { message },
        "block_too_large" => Error::BlockTooLarge { message },
        "rate_limited" => Error::RateLimited {
            message,
            retry_after,
        },
        "provider_error" => Error::Provider { message },
        "internal" => Error::Internal { message },
        _ => Error::Api {
            status: status.as_u16(),
            code: error.code,
            message,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(code: &str) -> Error {
        let body = format!(r#"{{"error":{{"code":"{code}","message":"m"}}}}"#);
        from_response(StatusCode::BAD_REQUEST, None, body.as_bytes())
    }

    #[test]
    fn every_documented_code_has_a_variant() {
        assert!(matches!(map("unauthorized"), Error::Unauthorized { .. }));
        assert!(matches!(map("forbidden"), Error::Forbidden { .. }));
        assert!(matches!(map("not_found"), Error::NotFound { .. }));
        assert!(matches!(map("invalid_input"), Error::InvalidInput { .. }));
        assert!(matches!(
            map("version_conflict"),
            Error::VersionConflict { .. }
        ));
        assert!(matches!(
            map("block_too_large"),
            Error::BlockTooLarge { .. }
        ));
        assert!(matches!(map("rate_limited"), Error::RateLimited { .. }));
        assert!(matches!(map("provider_error"), Error::Provider { .. }));
        assert!(matches!(map("internal"), Error::Internal { .. }));
    }

    #[test]
    fn an_unknown_code_keeps_its_detail() {
        match map("something_new") {
            Error::Api { status, code, .. } => {
                assert_eq!(status, 400);
                assert_eq!(code, "something_new");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_foreign_body_is_not_quoted() {
        let error = from_response(StatusCode::BAD_GATEWAY, None, b"<html>secret</html>");
        assert!(!error.to_string().contains("secret"));
        assert!(error.is_retryable());
    }

    #[test]
    fn retry_decisions() {
        assert!(map("rate_limited").is_retryable());
        assert!(!map("invalid_input").is_retryable());
        assert!(!map("internal").is_retryable());
        assert!(!map("version_conflict").is_retryable());
    }
}
