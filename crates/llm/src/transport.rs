//! HTTP shared by every client: timeouts, retries and error mapping.

use std::time::{Duration, SystemTime};

use memoryos_core::{Error, Result};
use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
use reqwest::{redirect, Client, StatusCode};
use serde::de::DeserializeOwned;
use serde_json::Value;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_RETRIES: u32 = 3;
const BACKOFF_BASE: Duration = Duration::from_millis(500);
/// A longer wait is not worth holding a worker for. The job queue has its own backoff.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
const MAX_ERROR_MESSAGE_CHARS: usize = 500;
const REDACTED: &str = "[REDACTED]";

/// How the API key is sent.
#[derive(Clone, Copy)]
pub(crate) enum Auth {
    XApiKey,
    Bearer,
}

/// Deliberately has no `Debug`: it holds the API key.
pub(crate) struct Transport {
    http: Client,
    provider: &'static str,
    api_key: String,
    auth: Auth,
}

struct Failure {
    message: String,
    retry: Retry,
}

enum Retry {
    Never,
    /// Retry, waiting at least the duration the server asked for, if any.
    Allowed(Option<Duration>),
}

impl Transport {
    /// # Panics
    ///
    /// When the TLS backend cannot initialize, which the public constructors have no way to
    /// return as an error (SPEC section 2).
    pub(crate) fn new(
        provider: &'static str,
        api_key: String,
        auth: Auth,
        default_headers: HeaderMap,
    ) -> Self {
        let http = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            // Cross-host redirects would carry x-api-key along. These APIs do not redirect POSTs.
            .redirect(redirect::Policy::none())
            .default_headers(default_headers)
            .build()
            .expect("TLS backend failed to initialize");
        Self {
            http,
            provider,
            api_key,
            auth,
        }
    }

    pub(crate) async fn post_json<T: DeserializeOwned>(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<T> {
        let mut retries = 0;
        loop {
            let failure = match self.attempt(url, body).await {
                Ok(parsed) => return Ok(parsed),
                Err(failure) => failure,
            };
            let message = self.scrub(&failure.message);
            let retry_after = match failure.retry {
                Retry::Never => return Err(Error::Provider(message)),
                Retry::Allowed(retry_after) => retry_after,
            };
            if retry_after.is_some_and(|wait| wait > MAX_RETRY_AFTER) {
                return Err(Error::Provider(format!(
                    "{message} (Retry-After above {}s, not waiting)",
                    MAX_RETRY_AFTER.as_secs()
                )));
            }
            if retries == MAX_RETRIES {
                return Err(Error::Provider(format!(
                    "{message} (after {MAX_RETRIES} retries)"
                )));
            }
            let delay = retry_delay(retry_after, retries, fastrand::f64());
            tracing::warn!(
                provider = self.provider,
                retry = retries + 1,
                delay_ms = delay.as_millis() as u64,
                error = %message,
                "retrying provider request"
            );
            tokio::time::sleep(delay).await;
            retries += 1;
        }
    }

    async fn attempt<T: DeserializeOwned>(
        &self,
        url: &str,
        body: &Value,
    ) -> std::result::Result<T, Failure> {
        let response = self
            .http
            .post(url)
            .headers(self.auth_header()?)
            .json(body)
            .send()
            .await
            .map_err(|err| self.transport_failure(err))?;

        let status = response.status();
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| parse_retry_after(value, SystemTime::now()));
        let bytes = response
            .bytes()
            .await
            .map_err(|err| self.transport_failure(err))?;

        if !status.is_success() {
            return Err(Failure {
                message: format!(
                    "{} returned {}: {}",
                    self.provider,
                    status.as_u16(),
                    error_message(&bytes, status)
                ),
                retry: if is_retryable(status) {
                    Retry::Allowed(retry_after)
                } else {
                    Retry::Never
                },
            });
        }
        serde_json::from_slice(&bytes).map_err(|err| Failure {
            message: format!("{} returned a malformed response: {err}", self.provider),
            retry: Retry::Never,
        })
    }

    fn auth_header(&self) -> std::result::Result<HeaderMap, Failure> {
        let (name, value) = match self.auth {
            Auth::XApiKey => ("x-api-key", self.api_key.clone()),
            Auth::Bearer => ("authorization", format!("Bearer {}", self.api_key)),
        };
        let mut value = HeaderValue::from_str(&value).map_err(|_| Failure {
            message: format!("{} API key is not a valid header value", self.provider),
            retry: Retry::Never,
        })?;
        value.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(name, value);
        Ok(headers)
    }

    fn transport_failure(&self, err: reqwest::Error) -> Failure {
        let retryable = err.is_connect() || err.is_timeout();
        Failure {
            message: format!("{} request failed: {}", self.provider, err.without_url()),
            retry: if retryable {
                Retry::Allowed(None)
            } else {
                Retry::Never
            },
        }
    }

    /// Some providers echo part of a rejected key back in their error message.
    fn scrub(&self, message: &str) -> String {
        if self.api_key.is_empty() {
            return message.to_owned();
        }
        message.replace(&self.api_key, REDACTED)
    }
}

fn is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

/// Accepts both RFC 9110 forms: delay in seconds, or an HTTP date.
fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let at = httpdate::parse_http_date(value).ok()?;
    Some(at.duration_since(now).unwrap_or_default())
}

/// Exponential backoff with jitter, added on top of any `Retry-After` so that clients told the
/// same wait do not all return at once. `unit` is a random value in `[0, 1)`.
fn retry_delay(retry_after: Option<Duration>, retries: u32, unit: f64) -> Duration {
    let backoff = BACKOFF_BASE.saturating_mul(2u32.saturating_pow(retries));
    retry_after.unwrap_or_default() + backoff.mul_f64(0.5 + 0.5 * unit)
}

/// Both providers use `{"error": {"message": ...}}`. Anything else (a gateway's HTML page, say)
/// is not quoted, because it is not known to be free of request content.
fn error_message(body: &[u8], status: StatusCode) -> String {
    let parsed: Option<Value> = serde_json::from_slice(body).ok();
    let message = parsed
        .as_ref()
        .and_then(|value| value.pointer("/error/message"))
        .and_then(Value::as_str);
    match message {
        Some(message) => message.chars().take(MAX_ERROR_MESSAGE_CHARS).collect(),
        None => status
            .canonical_reason()
            .unwrap_or("unknown error")
            .to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retries_429_408_and_5xx_only() {
        for code in [429, 408, 500, 502, 503, 529] {
            let status = StatusCode::from_u16(code).expect("valid status");
            assert!(is_retryable(status), "{code}");
        }
        for code in [400, 401, 403, 404, 409, 422] {
            let status = StatusCode::from_u16(code).expect("valid status");
            assert!(!is_retryable(status), "{code}");
        }
    }

    #[test]
    fn parses_retry_after_seconds_and_date() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        assert_eq!(parse_retry_after(" 7 ", now), Some(Duration::from_secs(7)));
        let later = httpdate::fmt_http_date(now + Duration::from_secs(30));
        assert_eq!(
            parse_retry_after(&later, now),
            Some(Duration::from_secs(30))
        );
        let earlier = httpdate::fmt_http_date(now - Duration::from_secs(30));
        assert_eq!(parse_retry_after(&earlier, now), Some(Duration::ZERO));
        assert_eq!(parse_retry_after("soon", now), None);
    }

    #[test]
    fn backoff_doubles_and_jitter_stays_in_bounds() {
        assert_eq!(retry_delay(None, 0, 0.0), Duration::from_millis(250));
        assert_eq!(retry_delay(None, 1, 0.0), Duration::from_millis(500));
        assert_eq!(retry_delay(None, 2, 0.0), Duration::from_millis(1000));
        assert!(retry_delay(None, 2, 0.999) < Duration::from_millis(2000));
    }

    #[test]
    fn retry_after_is_a_lower_bound() {
        let wait = Duration::from_secs(5);
        assert!(retry_delay(Some(wait), 0, 0.0) >= wait);
    }

    #[test]
    fn error_message_prefers_provider_message_and_never_quotes_unknown_bodies() {
        let json = br#"{"type":"error","error":{"type":"x","message":"bad tool"}}"#;
        assert_eq!(error_message(json, StatusCode::BAD_REQUEST), "bad tool");
        assert_eq!(
            error_message(b"<html>echoed secret</html>", StatusCode::BAD_GATEWAY),
            "Bad Gateway"
        );
    }

    #[test]
    fn scrub_removes_the_key() {
        let transport = Transport::new("t", "sk-secret".into(), Auth::Bearer, HeaderMap::new());
        assert_eq!(
            transport.scrub("Incorrect API key: sk-secret"),
            "Incorrect API key: [REDACTED]"
        );
    }
}
