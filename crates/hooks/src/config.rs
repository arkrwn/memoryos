//! Reading the environment. Every function takes a lookup, so tests never read or change the real
//! environment.

use std::path::PathBuf;

use memoryos_sdk::{MemoryOs, Options, Scope};
use url::Url;

pub const DEFAULT_URL: &str = "http://localhost:8080";
const DEFAULT_HOOK_SCOPES: &str = "agent:coding-agent";
const DEFAULT_STOP_EVERY: u64 = 10;

/// A variable's value. Empty counts as unset.
pub type Env<'a> = &'a dyn Fn(&str) -> Option<String>;

pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// The server URL, without a trailing slash. Refuses credentials in the URL, because it gets
/// written to configuration files.
pub fn base_url(env: Env<'_>) -> Result<String, String> {
    let raw = env("MEMORYOS_URL").unwrap_or_else(|| DEFAULT_URL.to_owned());
    let url = Url::parse(raw.trim()).map_err(|e| format!("MEMORYOS_URL is not a URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("MEMORYOS_URL must use http or https".to_owned());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("MEMORYOS_URL must not contain credentials".to_owned());
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

pub fn client(env: Env<'_>, options: Options) -> Result<MemoryOs, String> {
    let key = env("MEMORYOS_API_KEY").ok_or("MEMORYOS_API_KEY is not set")?;
    MemoryOs::new(base_url(env)?, key, options).map_err(|e| e.to_string())
}

pub fn hook_scopes(env: Env<'_>) -> Result<Vec<Scope>, String> {
    let raw = env("MEMORYOS_HOOK_SCOPES").unwrap_or_else(|| DEFAULT_HOOK_SCOPES.to_owned());
    let scopes = raw
        .split(',')
        .map(str::trim)
        .filter(|scope| !scope.is_empty())
        .map(|scope| {
            scope
                .parse()
                .map_err(|_| format!("MEMORYOS_HOOK_SCOPES has an invalid scope: {scope}"))
        })
        .collect::<Result<Vec<Scope>, String>>()?;
    if scopes.is_empty() {
        return Err("MEMORYOS_HOOK_SCOPES names no scope".to_owned());
    }
    Ok(scopes)
}

pub fn stop_every(env: Env<'_>) -> Result<u64, String> {
    let Some(raw) = env("MEMORYOS_HOOK_STOP_EVERY") else {
        return Ok(DEFAULT_STOP_EVERY);
    };
    match raw.trim().parse() {
        Ok(every) if every > 0 => Ok(every),
        _ => Err("MEMORYOS_HOOK_STOP_EVERY must be a positive integer".to_owned()),
    }
}

pub fn home(env: Env<'_>) -> Result<PathBuf, String> {
    env("HOME")
        .or_else(|| env("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| "HOME is not set".to_owned())
}

/// Where hook state lives: `$XDG_STATE_HOME`, or `~/.local/state`.
pub fn state_dir(env: Env<'_>) -> Result<PathBuf, String> {
    // The XDG specification says a relative value must be ignored.
    let base = match env("XDG_STATE_HOME").map(PathBuf::from) {
        Some(path) if path.is_absolute() => path,
        _ => home(env)?.join(".local").join("state"),
    };
    Ok(base.join("memoryos").join("hook-turns"))
}

/// Where OpenCode's global configuration lives: `$XDG_CONFIG_HOME/opencode`, or `~/.config/opencode`.
pub fn opencode_config_dir(env: Env<'_>) -> Result<PathBuf, String> {
    let base = match env("XDG_CONFIG_HOME").map(PathBuf::from) {
        Some(path) if path.is_absolute() => path,
        _ => home(env)?.join(".config"),
    };
    Ok(base.join("opencode"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn url_defaults_and_loses_its_trailing_slash() {
        assert_eq!(base_url(&with(&[])).expect("url"), "http://localhost:8080");
        let env = with(&[("MEMORYOS_URL", "https://m.example/base/")]);
        assert_eq!(base_url(&env).expect("url"), "https://m.example/base");
    }

    #[test]
    fn url_rejects_credentials_and_other_schemes() {
        for bad in ["http://u:p@h", "http://u@h", "ftp://h", "not a url"] {
            let pairs = [("MEMORYOS_URL", bad)];
            assert!(base_url(&with(&pairs)).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_missing_key_is_an_error() {
        let error = client(&with(&[]), Options::default()).expect_err("no key");
        assert!(error.contains("MEMORYOS_API_KEY"));
    }

    #[test]
    fn hook_scopes_default_and_parse() {
        assert_eq!(
            hook_scopes(&with(&[])).expect("scopes"),
            [Scope::agent("coding-agent")]
        );
        let env = with(&[("MEMORYOS_HOOK_SCOPES", "user:a, agent:b,")]);
        assert_eq!(
            hook_scopes(&env).expect("scopes"),
            [Scope::user("a"), Scope::agent("b")]
        );
        assert!(hook_scopes(&with(&[("MEMORYOS_HOOK_SCOPES", "bad")])).is_err());
        assert!(hook_scopes(&with(&[("MEMORYOS_HOOK_SCOPES", ",")])).is_err());
    }

    #[test]
    fn stop_every_must_be_positive() {
        assert_eq!(stop_every(&with(&[])).expect("default"), 10);
        assert_eq!(
            stop_every(&with(&[("MEMORYOS_HOOK_STOP_EVERY", "3")])).expect("set"),
            3
        );
        for bad in ["0", "-1", "x"] {
            assert!(stop_every(&with(&[("MEMORYOS_HOOK_STOP_EVERY", bad)])).is_err());
        }
    }

    #[test]
    fn state_dir_follows_xdg_when_absolute() {
        let home = [("HOME", "/h")];
        assert_eq!(
            state_dir(&with(&home)).expect("dir"),
            PathBuf::from("/h/.local/state/memoryos/hook-turns")
        );
        let xdg = [("HOME", "/h"), ("XDG_STATE_HOME", "/s")];
        assert_eq!(
            state_dir(&with(&xdg)).expect("dir"),
            PathBuf::from("/s/memoryos/hook-turns")
        );
        let relative = [("HOME", "/h"), ("XDG_STATE_HOME", "s")];
        assert_eq!(
            state_dir(&with(&relative)).expect("dir"),
            PathBuf::from("/h/.local/state/memoryos/hook-turns")
        );
        assert!(state_dir(&with(&[])).is_err());
    }

    #[test]
    fn opencode_dir_follows_xdg_when_absolute() {
        assert_eq!(
            opencode_config_dir(&with(&[("HOME", "/h")])).expect("dir"),
            PathBuf::from("/h/.config/opencode")
        );
        let xdg = [("HOME", "/h"), ("XDG_CONFIG_HOME", "/c")];
        assert_eq!(
            opencode_config_dir(&with(&xdg)).expect("dir"),
            PathBuf::from("/c/opencode")
        );
    }
}
