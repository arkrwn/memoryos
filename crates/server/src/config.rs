//! Environment configuration (SPEC section 10). Missing requirements fail here, at startup, with a
//! message naming the variable.

use std::fmt;
use std::time::Duration;

#[derive(Clone)]
pub struct Config {
    pub database_url: String,
    pub bind: String,
    pub db_max_connections: u32,
    pub rate_limit_rps: u32,
    pub request_timeout: Duration,
    /// `MEMORYOS_RERANKER`: whether recall reorders its top facts with the local reranker.
    pub reranker: RerankerMode,
    /// `MEMORYOS_RERANK_TOP`: the most fused facts one recall sends to the reranker.
    pub rerank_top: usize,
    /// `MEMORYOS_TEST_EMBEDDER`, in milliseconds per call: use the test embedder, not the model.
    #[cfg(feature = "test-embedder")]
    pub test_embedder_delay: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerMode {
    Local,
    Off,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("bind", &self.bind)
            .field("database_url", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

pub type ConfigResult<T> = std::result::Result<T, String>;

impl Config {
    pub fn from_env() -> ConfigResult<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// An empty variable counts as unset, as it does for most tools reading the same environment.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> ConfigResult<Self> {
        let get = |name: &str| lookup(name).filter(|value| !value.trim().is_empty());

        Ok(Self {
            database_url: database_url_from(&get)?,
            bind: get("MEMORYOS_BIND").unwrap_or_else(|| "0.0.0.0:8080".into()),
            db_max_connections: number(&get, "MEMORYOS_DB_MAX_CONNECTIONS", 20)?,
            rate_limit_rps: number(&get, "MEMORYOS_RATE_LIMIT_RPS", 100)?,
            request_timeout: Duration::from_secs(number(
                &get,
                "MEMORYOS_REQUEST_TIMEOUT_SECS",
                30,
            )?),
            reranker: reranker_mode(&get)?,
            rerank_top: number(&get, "MEMORYOS_RERANK_TOP", 20)?,
            #[cfg(feature = "test-embedder")]
            test_embedder_delay: get("MEMORYOS_TEST_EMBEDDER")
                .map(|millis| {
                    millis
                        .trim()
                        .parse()
                        .map(Duration::from_millis)
                        .map_err(|_| "MEMORYOS_TEST_EMBEDDER must be a number of milliseconds")
                })
                .transpose()?,
        })
    }
}

/// For the commands that only touch the database.
pub fn database_url_from_env() -> ConfigResult<String> {
    database_url_from(&|name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty()))
}

fn database_url_from(get: &impl Fn(&str) -> Option<String>) -> ConfigResult<String> {
    get("DATABASE_URL").ok_or_else(|| "DATABASE_URL is required".to_owned())
}

fn reranker_mode(get: &impl Fn(&str) -> Option<String>) -> ConfigResult<RerankerMode> {
    let Some(text) = get("MEMORYOS_RERANKER") else {
        return Ok(RerankerMode::Local);
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "local" => Ok(RerankerMode::Local),
        "off" => Ok(RerankerMode::Off),
        _ => Err("MEMORYOS_RERANKER must be local or off".to_owned()),
    }
}

/// Every number here is a size or a period, where zero would mean the service cannot work.
fn number<T>(get: &impl Fn(&str) -> Option<String>, name: &str, default: T) -> ConfigResult<T>
where
    T: std::str::FromStr + PartialEq + From<u8>,
{
    let Some(text) = get(name) else {
        return Ok(default);
    };
    match text.trim().parse::<T>() {
        Ok(value) if value != T::from(0) => Ok(value),
        _ => Err(format!("{name} must be a positive integer")),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn load(vars: &[(&str, &str)]) -> ConfigResult<Config> {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        Config::from_lookup(|name| map.get(name).cloned())
    }

    fn message(result: ConfigResult<Config>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(message) => message,
        }
    }

    const DB: (&str, &str) = ("DATABASE_URL", "postgres://x");

    #[test]
    fn only_the_database_is_required() {
        let config = load(&[DB]).expect("valid");
        assert_eq!(config.rate_limit_rps, 100);
        assert_eq!(config.bind, "0.0.0.0:8080");
    }

    #[test]
    fn empty_variables_count_as_unset() {
        assert!(message(load(&[("DATABASE_URL", " ")])).contains("DATABASE_URL"));
        let config = load(&[DB, ("MEMORYOS_RATE_LIMIT_RPS", "")]).expect("valid");
        assert_eq!(config.rate_limit_rps, 100);
    }

    #[test]
    fn bad_values_name_the_variable() {
        let bad_rate = load(&[DB, ("MEMORYOS_RATE_LIMIT_RPS", "0")]);
        assert!(message(bad_rate).contains("MEMORYOS_RATE_LIMIT_RPS"));
    }

    #[test]
    fn the_reranker_is_local_with_a_top_of_20_unless_set() {
        let config = load(&[DB]).expect("valid");
        assert_eq!(config.reranker, RerankerMode::Local);
        assert_eq!(config.rerank_top, 20);
        let config = load(&[
            DB,
            ("MEMORYOS_RERANKER", " OFF "),
            ("MEMORYOS_RERANK_TOP", "8"),
        ])
        .expect("valid");
        assert_eq!(config.reranker, RerankerMode::Off);
        assert_eq!(config.rerank_top, 8);
        let config = load(&[DB, ("MEMORYOS_RERANKER", "local")]).expect("valid");
        assert_eq!(config.reranker, RerankerMode::Local);
    }

    #[test]
    fn bad_reranker_settings_name_the_variable() {
        let bad_mode = load(&[DB, ("MEMORYOS_RERANKER", "remote")]);
        assert!(message(bad_mode).contains("MEMORYOS_RERANKER"));
        let bad_top = load(&[DB, ("MEMORYOS_RERANK_TOP", "0")]);
        assert!(message(bad_top).contains("MEMORYOS_RERANK_TOP"));
    }

    #[cfg(feature = "test-embedder")]
    #[test]
    fn the_test_embedder_is_off_unless_asked_for() {
        assert_eq!(load(&[DB]).expect("valid").test_embedder_delay, None);
        let delay = load(&[DB, ("MEMORYOS_TEST_EMBEDDER", "0")]).expect("valid");
        assert_eq!(delay.test_embedder_delay, Some(Duration::ZERO));
        let bad = load(&[DB, ("MEMORYOS_TEST_EMBEDDER", "soon")]);
        assert!(message(bad).contains("MEMORYOS_TEST_EMBEDDER"));
    }

    #[test]
    fn debug_output_hides_secrets() {
        let config = load(&[("DATABASE_URL", "postgres://user:hunter2@db")]).expect("valid");
        assert!(!format!("{config:?}").contains("hunter2"));
    }
}
