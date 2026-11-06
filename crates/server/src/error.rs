//! The error body and status mapping of SPEC section 9. Detail on storage and internal errors goes
//! to the log only.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use memoryos_core::Error;
use serde_json::json;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    pub retry_after_secs: Option<u64>,
}

pub type ApiResult<T> = std::result::Result<T, ApiError>;

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            retry_after_secs: None,
        }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_input", message)
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, "forbidden", message)
    }

    /// One body for "absent" and "in a scope you cannot see", so ids cannot be probed.
    pub fn not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", "not found")
    }

    pub fn route_not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", "route not found")
    }

    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing or invalid bearer token",
        )
    }

    pub fn rate_limited(retry_after_secs: u64) -> Self {
        Self {
            retry_after_secs: Some(retry_after_secs),
            ..Self::new(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "rate limit exceeded for this key",
            )
        }
    }

    pub fn internal() -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal error",
        )
    }

    pub fn payload_too_large() -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "request body exceeds the 1 MiB limit",
        )
    }

    pub fn method_not_allowed() -> Self {
        Self::new(
            StatusCode::METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "method not allowed on this route",
        )
    }

    pub fn timeout() -> Self {
        Self::new(
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            "the request did not finish in time",
        )
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        match error {
            Error::NotFound(_) => Self::not_found(),
            Error::InvalidInput(message) => Self::invalid_input(message),
            Error::Forbidden(message) => Self::forbidden(message),
            conflict @ Error::VersionConflict { .. } => Self::new(
                StatusCode::CONFLICT,
                "version_conflict",
                conflict.to_string(),
            ),
            too_large @ Error::BlockTooLarge { .. } => Self::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "block_too_large",
                too_large.to_string(),
            ),
            Error::Provider(detail) => {
                // Only the part before the first colon: what follows can be the provider's own
                // error text, which may quote the request.
                let summary = detail.split(':').next().unwrap_or_default();
                tracing::warn!(summary, "provider call failed");
                Self::new(
                    StatusCode::BAD_GATEWAY,
                    "provider_error",
                    "an upstream provider request failed",
                )
            }
            Error::Storage(detail) => {
                tracing::error!(error = %detail, "storage error");
                Self::internal()
            }
            Error::Internal(detail) => {
                tracing::error!(error = %detail, "internal error");
                Self::internal()
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {"code": self.code, "message": self.message}});
        let mut response = (self.status, Json(body)).into_response();
        if let Some(seconds) = self.retry_after_secs {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_errors_map_per_spec_section_9() {
        let cases = [
            (Error::NotFound("x".into()), 404, "not_found"),
            (Error::InvalidInput("x".into()), 400, "invalid_input"),
            (Error::Forbidden("x".into()), 403, "forbidden"),
            (
                Error::VersionConflict {
                    expected: 1,
                    current: 2,
                },
                409,
                "version_conflict",
            ),
            (
                Error::BlockTooLarge { len: 2, max: 1 },
                422,
                "block_too_large",
            ),
            (Error::Provider("x".into()), 502, "provider_error"),
            (Error::Storage("x".into()), 500, "internal"),
            (Error::Internal("x".into()), 500, "internal"),
        ];
        for (error, status, code) in cases {
            let mapped = ApiError::from(error);
            assert_eq!(mapped.status.as_u16(), status);
            assert_eq!(mapped.code, code);
        }
    }

    #[test]
    fn internal_detail_never_reaches_the_message() {
        for error in [
            Error::Storage("password=hunter2".into()),
            Error::Internal("password=hunter2".into()),
            Error::Provider("openai: password=hunter2".into()),
            Error::NotFound("row in tenant secret".into()),
        ] {
            let mapped = ApiError::from(error);
            assert!(!mapped.message.contains("hunter2"));
            assert!(!mapped.message.contains("secret"));
        }
    }
}
