//! Per-session state for the hooks: turn counts for `hook stop`, and the fact ids already shown by
//! `hook session-start` and `hook prompt`. Each session has its own small files, written atomically,
//! so concurrent sessions never share a file and a reader never sees a partial one.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::files::write_atomic;

/// A session idle this long has finished, or will simply start counting again.
const MAX_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Adds one turn and returns the new count. An unreadable count starts again rather than failing.
pub fn count_turn(dir: &Path, session_id: &str) -> Result<u64, String> {
    let path = state_file(dir, session_id);
    let previous = fs::read_to_string(&path)
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let count = previous.saturating_add(1);
    write_atomic(&path, count.to_string().as_bytes(), None)?;
    Ok(count)
}

/// The fact ids already shown in this session. A missing or unreadable file means none.
pub fn seen(dir: &Path, session_id: &str) -> HashSet<Uuid> {
    fs::read_to_string(seen_file(dir, session_id))
        .map(|text| {
            text.lines()
                .filter_map(|line| line.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Replaces the session's shown ids with `ids`. Session start calls it, so a new or cleared session
/// starts from what it printed.
pub fn reset_seen(dir: &Path, session_id: &str, ids: &[Uuid]) -> Result<(), String> {
    write_seen(dir, session_id, ids)
}

/// Adds `ids` to the session's shown ids.
pub fn add_seen(dir: &Path, session_id: &str, ids: &[Uuid]) -> Result<(), String> {
    if ids.is_empty() {
        return Ok(());
    }
    let mut all: Vec<Uuid> = seen(dir, session_id).into_iter().collect();
    all.extend(ids.iter().copied());
    all.sort_unstable();
    all.dedup();
    write_seen(dir, session_id, &all)
}

fn write_seen(dir: &Path, session_id: &str, ids: &[Uuid]) -> Result<(), String> {
    let text: String = ids.iter().map(|id| format!("{id}\n")).collect();
    write_atomic(&seen_file(dir, session_id), text.as_bytes(), None)
}

/// The session id comes from the host's JSON, so it is hashed rather than used as a file name: a
/// value like `../../x` must not reach outside the state directory.
fn state_file(dir: &Path, session_id: &str) -> PathBuf {
    dir.join(format!("{}.turns", hashed(session_id)))
}

fn seen_file(dir: &Path, session_id: &str) -> PathBuf {
    dir.join(format!("{}.seen", hashed(session_id)))
}

fn hashed(session_id: &str) -> String {
    let digest = Sha256::digest(session_id.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Removes files older than [`MAX_AGE`], including temporary files left by a killed process. Best
/// effort: another process may be doing the same.
pub fn remove_old(dir: &Path, now: SystemTime) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let expired = metadata
            .modified()
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > MAX_AGE);
        if metadata.is_file() && expired {
            let _ = fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_each_session_separately() {
        let dir = tempfile::tempdir().expect("dir");
        assert_eq!(count_turn(dir.path(), "a").expect("count"), 1);
        assert_eq!(count_turn(dir.path(), "a").expect("count"), 2);
        assert_eq!(count_turn(dir.path(), "b").expect("count"), 1);
        assert_eq!(count_turn(dir.path(), "a").expect("count"), 3);
    }

    #[test]
    fn creates_the_directory() {
        let dir = tempfile::tempdir().expect("dir");
        let nested = dir.path().join("x/y");
        assert_eq!(count_turn(&nested, "a").expect("count"), 1);
    }

    #[test]
    fn a_session_id_cannot_escape_the_directory() {
        let dir = tempfile::tempdir().expect("dir");
        let state = dir.path().join("state");
        count_turn(&state, "../../escaped/name\0").expect("count");
        let outside: Vec<_> = fs::read_dir(dir.path()).expect("list").flatten().collect();
        assert_eq!(outside.len(), 1, "only the state directory");
        assert_eq!(fs::read_dir(&state).expect("list").count(), 1);
    }

    #[test]
    fn a_corrupt_count_starts_again() {
        let dir = tempfile::tempdir().expect("dir");
        count_turn(dir.path(), "a").expect("count");
        let file = state_file(dir.path(), "a");
        fs::write(&file, "not a number").expect("corrupt");
        assert_eq!(count_turn(dir.path(), "a").expect("count"), 1);
    }

    #[test]
    fn a_file_in_the_way_of_the_directory_is_an_error() {
        let dir = tempfile::tempdir().expect("dir");
        let blocker = dir.path().join("blocker");
        fs::write(&blocker, "").expect("seed");
        assert!(count_turn(&blocker, "a").is_err());
    }

    #[test]
    fn only_old_files_are_removed() {
        let dir = tempfile::tempdir().expect("dir");
        count_turn(dir.path(), "old").expect("count");
        count_turn(dir.path(), "new").expect("count");
        let stale_temp = dir.path().join(".x.1.2.tmp");
        fs::write(&stale_temp, "").expect("seed");
        let eight_days = SystemTime::now() - Duration::from_secs(8 * 24 * 60 * 60);
        for path in [state_file(dir.path(), "old"), stale_temp.clone()] {
            let file = fs::File::options().write(true).open(&path).expect("open");
            file.set_modified(eight_days).expect("set time");
        }

        remove_old(dir.path(), SystemTime::now());

        assert!(!state_file(dir.path(), "old").exists());
        assert!(!stale_temp.exists());
        assert!(state_file(dir.path(), "new").exists());
    }

    #[test]
    fn seen_ids_are_reset_added_and_kept_per_session() {
        let dir = tempfile::tempdir().expect("dir");
        let (a, b, c) = (Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7());
        assert!(seen(dir.path(), "s").is_empty(), "nothing before a start");
        reset_seen(dir.path(), "s", &[a]).expect("reset");
        add_seen(dir.path(), "s", &[b, a]).expect("add");
        assert_eq!(seen(dir.path(), "s"), HashSet::from([a, b]));
        assert!(seen(dir.path(), "other").is_empty());
        reset_seen(dir.path(), "s", &[c]).expect("reset");
        assert_eq!(seen(dir.path(), "s"), HashSet::from([c]));
        reset_seen(dir.path(), "s", &[]).expect("reset");
        assert!(seen(dir.path(), "s").is_empty());
    }

    #[test]
    fn a_corrupt_seen_file_keeps_its_valid_ids() {
        let dir = tempfile::tempdir().expect("dir");
        let id = Uuid::now_v7();
        fs::write(seen_file(dir.path(), "s"), format!("junk\n{id}\n")).expect("seed");
        assert_eq!(seen(dir.path(), "s"), HashSet::from([id]));
    }

    #[test]
    fn a_missing_directory_is_not_a_problem() {
        let dir = tempfile::tempdir().expect("dir");
        remove_old(&dir.path().join("missing"), SystemTime::now());
    }

    #[test]
    fn concurrent_sessions_keep_exact_counts_and_valid_files() {
        let dir = tempfile::tempdir().expect("dir");
        let threads: Vec<_> = (0..16)
            .map(|index| {
                let path = dir.path().to_owned();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        count_turn(&path, &format!("session-{index}")).expect("count");
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("thread");
        }
        for index in 0..16 {
            let file = state_file(dir.path(), &format!("session-{index}"));
            assert_eq!(fs::read_to_string(file).expect("read"), "50");
        }
        assert_eq!(
            fs::read_dir(dir.path()).expect("list").count(),
            16,
            "no temporary file is left"
        );
    }

    #[test]
    fn writers_racing_on_one_session_leave_a_valid_count() {
        let dir = tempfile::tempdir().expect("dir");
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let path = dir.path().to_owned();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        count_turn(&path, "same").expect("count");
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("thread");
        }
        let text = fs::read_to_string(state_file(dir.path(), "same")).expect("read");
        let count: u64 = text.parse().expect("a whole number, never a partial write");
        assert!((1..=400).contains(&count));
    }
}
