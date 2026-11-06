//! Environment configuration (SPEC sections 10 and 14). Missing requirements fail here, at startup,
//! with a message naming the variable. Values of keys never appear in errors or `Debug` output.

use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Host,
    Worker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderKind {
    Local,
    OpenAi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    Anthropic,
    OpenAi,
}

/// The LLM client settings, present only in worker mode.
#[derive(Clone)]
pub struct LlmConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
}

#[derive(Clone)]
pub struct Config {
    pub database_url: String,
    pub bind: String,
    pub mode: Mode,
    pub embedder: EmbedderKind,
    pub llm: Option<LlmConfig>,
    pub embedding_model: String,
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub db_max_connections: u32,
    pub rate_limit_rps: u32,
    pub worker_concurrency: usize,
    pub request_timeout: Duration,
    /// An agent turn makes several model calls, so it gets its own limit.
    pub turn_timeout: Duration,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("bind", &self.bind)
            .field("mode", &self.mode)
            .field("embedder", &self.embedder)
            .field("database_url", &"[REDACTED]")
            .field("api_keys", &"[REDACTED]")
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

        let mode = match get("MEMORYOS_MODE").as_deref() {
            None | Some("host") => Mode::Host,
            Some("worker") => Mode::Worker,
            Some(_) => return Err("MEMORYOS_MODE must be host or worker".into()),
        };
        let embedder = match get("MEMORYOS_EMBEDDER").as_deref() {
            None => match mode {
                Mode::Host => EmbedderKind::Local,
                Mode::Worker => EmbedderKind::OpenAi,
            },
            Some("local") => EmbedderKind::Local,
            Some("openai") => EmbedderKind::OpenAi,
            Some(_) => return Err("MEMORYOS_EMBEDDER must be local or openai".into()),
        };

        let openai_api_key = get("OPENAI_API_KEY");
        let openai_base_url = get("OPENAI_BASE_URL");
        if embedder == EmbedderKind::OpenAi && openai_api_key.is_none() {
            return Err(
                "the openai embedder requires OPENAI_API_KEY (or set MEMORYOS_EMBEDDER=local)"
                    .into(),
            );
        }

        let llm = match mode {
            Mode::Host => None,
            Mode::Worker => Some(llm_config(&get, &openai_api_key, &openai_base_url)?),
        };

        Ok(Self {
            database_url: database_url_from(&get)?,
            bind: get("MEMORYOS_BIND").unwrap_or_else(|| "0.0.0.0:8080".into()),
            mode,
            embedder,
            llm,
            embedding_model: get("MEMORYOS_EMBEDDING_MODEL")
                .unwrap_or_else(|| "text-embedding-3-small".into()),
            openai_api_key,
            openai_base_url,
            db_max_connections: number(&get, "MEMORYOS_DB_MAX_CONNECTIONS", 20)?,
            rate_limit_rps: number(&get, "MEMORYOS_RATE_LIMIT_RPS", 100)?,
            worker_concurrency: number(&get, "MEMORYOS_WORKER_CONCURRENCY", 4)?,
            request_timeout: Duration::from_secs(number(
                &get,
                "MEMORYOS_REQUEST_TIMEOUT_SECS",
                30,
            )?),
            turn_timeout: Duration::from_secs(number(&get, "MEMORYOS_TURN_TIMEOUT_SECS", 300)?),
        })
    }
}

/// For the commands that only touch the database, which must not demand provider keys.
pub fn database_url_from_env() -> ConfigResult<String> {
    database_url_from(&|name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty()))
}

fn database_url_from(get: &impl Fn(&str) -> Option<String>) -> ConfigResult<String> {
    get("DATABASE_URL").ok_or_else(|| "DATABASE_URL is required".to_owned())
}

fn llm_config(
    get: &impl Fn(&str) -> Option<String>,
    openai_api_key: &Option<String>,
    openai_base_url: &Option<String>,
) -> ConfigResult<LlmConfig> {
    let provider = match get("MEMORYOS_LLM_PROVIDER").as_deref() {
        None | Some("anthropic") => LlmProvider::Anthropic,
        Some("openai") => LlmProvider::OpenAi,
        Some(_) => return Err("MEMORYOS_LLM_PROVIDER must be anthropic or openai".into()),
    };
    let (key_name, api_key, base_url, default_model) = match provider {
        LlmProvider::Anthropic => (
            "ANTHROPIC_API_KEY",
            get("ANTHROPIC_API_KEY"),
            get("ANTHROPIC_BASE_URL"),
            "claude-haiku-5-5",
        ),
        LlmProvider::OpenAi => (
            "OPENAI_API_KEY",
            openai_api_key.clone(),
            openai_base_url.clone(),
            "gpt-5.1",
        ),
    };
    let api_key = api_key.ok_or_else(|| {
        format!("MEMORYOS_MODE=worker requires {key_name} for the configured LLM provider")
    })?;
    Ok(LlmConfig {
        provider,
        model: get("MEMORYOS_LLM_MODEL").unwrap_or_else(|| default_model.to_owned()),
        api_key,
        base_url,
    })
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
    fn host_mode_is_the_default_and_needs_no_key() {
        let config = load(&[DB]).expect("valid");
        assert_eq!(config.mode, Mode::Host);
        assert_eq!(config.embedder, EmbedderKind::Local);
        assert!(config.llm.is_none());
        assert_eq!(config.rate_limit_rps, 100);
        assert_eq!(config.bind, "0.0.0.0:8080");
    }

    #[test]
    fn worker_mode_requires_the_key_of_the_configured_provider() {
        let openai = ("OPENAI_API_KEY", "k");
        let worker = ("MEMORYOS_MODE", "worker");
        assert!(message(load(&[DB, worker, openai])).contains("ANTHROPIC_API_KEY"));
        let provider = ("MEMORYOS_LLM_PROVIDER", "openai");
        assert!(load(&[DB, worker, openai, provider]).is_ok());
        let anthropic = ("ANTHROPIC_API_KEY", "k");
        let config = load(&[DB, worker, openai, anthropic]).expect("valid");
        let llm = config.llm.expect("llm");
        assert_eq!(llm.model, "claude-haiku-5-5");
    }

    #[test]
    fn worker_mode_defaults_to_openai_embeddings_and_allows_local() {
        let worker = ("MEMORYOS_MODE", "worker");
        let anthropic = ("ANTHROPIC_API_KEY", "k");
        assert!(message(load(&[DB, worker, anthropic])).contains("OPENAI_API_KEY"));
        let local = ("MEMORYOS_EMBEDDER", "local");
        let config = load(&[DB, worker, anthropic, local]).expect("valid");
        assert_eq!(config.embedder, EmbedderKind::Local);
    }

    #[test]
    fn empty_variables_count_as_unset() {
        assert!(message(load(&[("DATABASE_URL", " ")])).contains("DATABASE_URL"));
        let config = load(&[DB, ("MEMORYOS_MODE", "")]).expect("valid");
        assert_eq!(config.mode, Mode::Host);
    }

    #[test]
    fn bad_values_name_the_variable() {
        assert!(message(load(&[DB, ("MEMORYOS_MODE", "x")])).contains("MEMORYOS_MODE"));
        let bad_rate = load(&[DB, ("MEMORYOS_RATE_LIMIT_RPS", "0")]);
        assert!(message(bad_rate).contains("MEMORYOS_RATE_LIMIT_RPS"));
    }

    #[test]
    fn debug_output_hides_secrets() {
        let config = load(&[
            ("DATABASE_URL", "postgres://user:hunter2@db"),
            ("MEMORYOS_EMBEDDER", "openai"),
            ("OPENAI_API_KEY", "sk-secret"),
        ])
        .expect("valid");
        let text = format!("{config:?}");
        assert!(!text.contains("hunter2"));
        assert!(!text.contains("sk-secret"));
    }
}
