//! Writing files so that a reader sees the old content or the new, never a mix.

use std::fs::{self, File, OpenOptions, Permissions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Writes a temporary file in the same directory, then renames it over `path`. Creates the parent
/// directory. `permissions` keeps an existing file's mode: `~/.claude.json` is usually private.
pub fn write_atomic(
    path: &Path,
    content: &[u8],
    permissions: Option<Permissions>,
) -> Result<(), String> {
    let describe = |action: &str, error: std::io::Error| {
        format!("cannot {action} {}: {error}", path.display())
    };
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).map_err(|e| describe("create the directory of", e))?;
    let temp = temp_path(path);
    let result = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(content)?;
        if let Some(permissions) = permissions {
            file.set_permissions(permissions)?;
        }
        file.sync_all()?;
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.map_err(|e| describe("write", e))
}

/// Unique across processes and across threads of one process.
fn temp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".{name}.{}.{suffix}.tmp", std::process::id()))
}

/// Saves `content` next to `path`, as `<path>.bak`. Never overwrites a backup that holds something
/// else, since that may be the user's own file or the only copy of the original: the next free
/// `<path>.bak.N` is used instead. Returns the backup's path.
pub fn back_up(path: &Path, content: &[u8]) -> Result<PathBuf, String> {
    let permissions = fs::metadata(path).ok().map(|m| m.permissions());
    for attempt in 0..1000 {
        let suffix = if attempt == 0 {
            ".bak".to_owned()
        } else {
            format!(".bak.{attempt}")
        };
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        let candidate = PathBuf::from(name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return fill_backup(file, &candidate, content, permissions),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if fs::read(&candidate).is_ok_and(|existing| existing == content) {
                    return Ok(candidate);
                }
            }
            Err(error) => return Err(format!("cannot write {}: {error}", candidate.display())),
        }
    }
    Err(format!("too many backups of {}", path.display()))
}

fn fill_backup(
    mut file: File,
    path: &Path,
    content: &[u8],
    permissions: Option<Permissions>,
) -> Result<PathBuf, String> {
    let result = (|| -> std::io::Result<()> {
        if let Some(permissions) = permissions {
            file.set_permissions(permissions)?;
        }
        file.write_all(content)?;
        file.sync_all()
    })();
    match result {
        Ok(()) => Ok(path.to_owned()),
        Err(error) => {
            let _ = fs::remove_file(path);
            Err(format!("cannot write {}: {error}", path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_creates_parents_and_leaves_no_temporary_file() {
        let dir = tempfile::tempdir().expect("dir");
        let path = dir.path().join("a/b/file.json");
        write_atomic(&path, b"one", None).expect("write");
        write_atomic(&path, b"two", None).expect("overwrite");
        assert_eq!(fs::read(&path).expect("read"), b"two");
        let names: Vec<_> = fs::read_dir(path.parent().expect("parent"))
            .expect("list")
            .collect();
        assert_eq!(names.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn write_and_backup_keep_a_private_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("dir");
        let path = dir.path().join("private.json");
        fs::write(&path, "old").expect("seed");
        fs::set_permissions(&path, Permissions::from_mode(0o600)).expect("chmod");
        let old = fs::metadata(&path).expect("stat").permissions();

        let backup = back_up(&path, b"old").expect("backup");
        write_atomic(&path, b"new", Some(old)).expect("write");

        for file in [&path, &backup] {
            let mode = fs::metadata(file).expect("stat").permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "{}", file.display());
        }
    }

    #[test]
    fn a_backup_never_replaces_different_content() {
        let dir = tempfile::tempdir().expect("dir");
        let path = dir.path().join("f.json");
        fs::write(dir.path().join("f.json.bak"), "the user's own").expect("seed");

        let first = back_up(&path, b"v1").expect("backup");
        assert_eq!(first, dir.path().join("f.json.bak.1"));
        assert_eq!(
            fs::read(dir.path().join("f.json.bak")).expect("read"),
            b"the user's own"
        );

        // The same content again reuses that backup.
        assert_eq!(back_up(&path, b"v1").expect("again"), first);
        assert_eq!(
            back_up(&path, b"v2").expect("next"),
            dir.path().join("f.json.bak.2")
        );
    }
}
