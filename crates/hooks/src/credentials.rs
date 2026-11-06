//! `<config>/memoryos/credentials.json`, written by `memoryos setup` with mode 0600.
//!
//! The agent key is what hooks and `memoryos mcp` use. The admin key is only for `memoryos
//! quarantine`. This type has no `Debug`, so a key cannot reach a log through `{:?}`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{self, Env};
use crate::files::write_atomic;

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Credentials {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admin_key: Option<String>,
}

pub fn path(env: Env<'_>) -> Result<PathBuf, String> {
    Ok(config::memoryos_dir(env)?.join("credentials.json"))
}

/// `None` when there is no file, or no home directory to look in. The error never quotes the file,
/// because it holds keys.
pub fn load(env: Env<'_>) -> Result<Option<Credentials>, String> {
    let Ok(path) = path(env) else {
        return Ok(None);
    };
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
    };
    serde_json::from_str(&text).map(Some).map_err(|_| {
        format!(
            "{} is not valid. Delete it and run `memoryos setup`",
            path.display()
        )
    })
}

pub fn save(env: Env<'_>, credentials: &Credentials) -> Result<(), String> {
    let path = path(env)?;
    let mut text = serde_json::to_string_pretty(credentials)
        .map_err(|e| format!("cannot encode the credentials: {e}"))?;
    text.push('\n');
    write_private(&path, text.as_bytes())
}

/// Writes a secret: mode 0600, in a directory with mode 0700.
pub fn write_private(path: &Path, content: &[u8]) -> Result<(), String> {
    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt;
        Some(fs::Permissions::from_mode(0o600))
    };
    #[cfg(not(unix))]
    let permissions = None;
    write_atomic(path, content, permissions)?;
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("cannot protect {}: {e}", parent.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Credentials {
        Credentials {
            url: "http://127.0.0.1:8080".to_owned(),
            agent_key: Some("mos_agent".to_owned()),
            admin_key: Some("mos_admin".to_owned()),
        }
    }

    fn env_for(home: &Path) -> impl Fn(&str) -> Option<String> {
        let home = home.display().to_string();
        move |name: &str| (name == "HOME").then(|| home.clone())
    }

    #[test]
    fn saved_credentials_round_trip() {
        let dir = tempfile::tempdir().expect("dir");
        let env = env_for(dir.path());
        assert!(load(&env).expect("empty").is_none());
        save(&env, &sample()).expect("save");
        let loaded = load(&env).expect("load").expect("present");
        assert!(loaded == sample());
    }

    #[cfg(unix)]
    #[test]
    fn the_file_is_0600_and_its_directory_0700_even_over_a_loose_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("dir");
        let env = env_for(dir.path());
        let file = path(&env).expect("path");
        fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
        fs::write(&file, "{\"url\":\"x\"}").expect("seed");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).expect("chmod");

        save(&env, &sample()).expect("save");

        let mode = |p: &Path| fs::metadata(p).expect("stat").permissions().mode() & 0o777;
        assert_eq!(mode(&file), 0o600);
        assert_eq!(mode(file.parent().expect("parent")), 0o700);
    }

    #[test]
    fn a_broken_file_is_an_error_that_quotes_nothing_from_it() {
        let dir = tempfile::tempdir().expect("dir");
        let env = env_for(dir.path());
        let file = path(&env).expect("path");
        fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
        fs::write(&file, "{\"url\": 1, \"agent_key\": \"mos_leak\"").expect("seed");
        let Err(error) = load(&env) else {
            panic!("expected an error");
        };
        assert!(!error.contains("mos_leak"), "{error}");
    }

    #[test]
    fn no_home_means_no_credentials() {
        assert!(load(&|_: &str| None).expect("none").is_none());
    }
}
