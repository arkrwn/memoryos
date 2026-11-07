//! Keeps what the user typed in a session that saved nothing, so that a later session's model can
//! save it. `hook session-end` and `hook pre-compact` capture, and `hook session-start` shows each
//! pending file once and deletes it. No model runs here and nothing leaves the machine: the text stays
//! in a private file until the next session's host reads it.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

use regex::{Captures, Regex};
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::credentials::write_private;
use crate::files::write_atomic;
use crate::hook::{data_text, CLOSE as MEMORIES_CLOSE, OPEN as MEMORIES_OPEN};
use crate::turns::hashed;

/// The most a pending file holds, and the most `hook session-start` shows, counted as printed.
pub const MAX_PENDING_CHARS: usize = 4_000;
/// A pending file nobody has seen in this long is deleted without being shown.
const MAX_PENDING_AGE: Duration = Duration::from_secs(14 * 24 * 60 * 60);

pub(crate) const UNSAVED_OPEN: &str = "<memoryos-unsaved>";
pub(crate) const UNSAVED_CLOSE: &str = "</memoryos-unsaved>";

/// The tool's name with the server from `memoryos setup` and with the plugin's server.
const COMMIT_TOOLS: [&str; 2] = [
    "mcp__memoryos__commit_facts",
    "mcp__plugin_memoryos_memoryos__commit_facts",
];

/// Text the host or a hook put into the user's turn. Each block is removed up to its closing tag, or
/// to the end when it is not closed.
const INJECTED_BLOCKS: [(&str, &str); 3] = [
    ("<system-reminder>", "</system-reminder>"),
    (MEMORIES_OPEN, MEMORIES_CLOSE),
    (UNSAVED_OPEN, UNSAVED_CLOSE),
];

/// A user line that starts with one of these is the host's, not something the user typed.
const INJECTED_PREFIXES: [&str; 11] = [
    "<task-notification>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<local-command-",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<user-prompt-submit-hook>",
    "<agent-message",
    "[Request interrupted by user",
];

/// What a [`SECRET_PATTERNS`] match must also contain to be replaced.
#[derive(Clone, Copy)]
pub(crate) enum Check {
    Always,
    /// A digit, an uppercase and a lowercase letter, as random tokens have and words rarely do.
    Mixed,
}

/// Each match becomes `[redacted]` before anything is written.
pub(crate) const SECRET_PATTERNS: [(&str, Check); 10] = [
    (
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?s:.*?)(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\z)",
        Check::Always,
    ),
    (r"\bsk-[A-Za-z0-9_-]{16,}", Check::Always),
    (r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}", Check::Always),
    (r"\bgithub_pat_[A-Za-z0-9_]{20,}", Check::Always),
    (r"\bA(?:KIA|SIA)[0-9A-Z]{16}\b", Check::Always),
    (r"\bmos_[A-Za-z0-9_-]{8,}", Check::Always),
    (r"\bxox[abpr]-[A-Za-z0-9-]{10,}", Check::Always),
    (
        r"(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key)\s*[=:]\s*\S+",
        Check::Always,
    ),
    (r"\b[0-9a-fA-F]{33,}\b", Check::Always),
    (r"[A-Za-z0-9+]{33,}={0,2}", Check::Mixed),
];

const REDACTED: &str = "[redacted]";

fn secret_patterns() -> &'static [(Regex, Check)] {
    static COMPILED: OnceLock<Vec<(Regex, Check)>> = OnceLock::new();
    COMPILED.get_or_init(|| {
        SECRET_PATTERNS
            .iter()
            .map(|(pattern, check)| (Regex::new(pattern).expect("a valid pattern"), *check))
            .collect()
    })
}

/// `text` with every [`SECRET_PATTERNS`] match replaced by `[redacted]`.
pub(crate) fn redact(text: &str) -> String {
    let mut out = text.to_owned();
    for (pattern, check) in secret_patterns() {
        let replaced = pattern.replace_all(&out, |found: &Captures<'_>| {
            let found = &found[0];
            let secret = match check {
                Check::Always => true,
                Check::Mixed => {
                    found.chars().any(|c| c.is_ascii_digit())
                        && found.chars().any(|c| c.is_ascii_uppercase())
                        && found.chars().any(|c| c.is_ascii_lowercase())
                }
            };
            if secret {
                REDACTED.to_owned()
            } else {
                found.to_owned()
            }
        });
        out = replaced.into_owned();
    }
    out
}

/// One prompt as `hook session-start` prints it. The caps count these characters, so a file always
/// fits the space it is shown in.
pub(crate) fn line(prompt: &str) -> String {
    format!("- {}\n", data_text(prompt))
}

/// Reads the transcript from where the last capture of this session stopped. When that part has no
/// `commit_facts` call, adds the user's typed prompts in it to the session's pending file.
pub fn capture(
    dir: &Path,
    session_id: &str,
    transcript: &Path,
    now: SystemTime,
) -> Result<(), String> {
    remove_expired(dir, now);
    let marker = dir.join(format!("{}.captured", hashed(session_id)));
    let from = fs::read_to_string(&marker)
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let mut file =
        File::open(transcript).map_err(|e| format!("cannot open the transcript: {e}"))?;
    let length = file
        .metadata()
        .map_err(|e| format!("cannot read the transcript: {e}"))?
        .len();
    // A shorter file is a different one. Start again.
    let from = if from > length { 0 } else { from };
    file.seek(SeekFrom::Start(from))
        .map_err(|e| format!("cannot read the transcript: {e}"))?;
    let scan = scan(BufReader::new(file));
    if !scan.saved && !scan.prompts.is_empty() {
        append_pending(dir, session_id, scan.prompts)?;
    }
    write_atomic(&marker, (from + scan.consumed).to_string().as_bytes(), None)
}

#[derive(Debug, Default)]
struct Scan {
    prompts: Vec<String>,
    saved: bool,
    /// Bytes of whole lines read. A last line without its newline may still be being written.
    consumed: u64,
}

fn scan(mut reader: impl BufRead) -> Scan {
    let mut scan = Scan::default();
    let mut buffer = Vec::new();
    loop {
        buffer.clear();
        match reader.read_until(b'\n', &mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(_) if !buffer.ends_with(b"\n") => break,
            Ok(read) => scan.consumed += read as u64,
        }
        read_line(&buffer, &mut scan);
    }
    scan
}

#[derive(Deserialize)]
struct Line<'a> {
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(rename = "isMeta", default)]
    is_meta: Option<bool>,
    #[serde(rename = "isSidechain", default)]
    is_sidechain: Option<bool>,
    #[serde(rename = "isCompactSummary", default)]
    is_compact_summary: Option<bool>,
    #[serde(borrow)]
    message: Option<Message<'a>>,
}

#[derive(Deserialize)]
struct Message<'a> {
    #[serde(borrow)]
    content: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: Option<String>,
    text: Option<String>,
    name: Option<String>,
}

fn read_line(bytes: &[u8], scan: &mut Scan) {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return;
    };
    let mentions_commit = text.contains("commit_facts");
    // Tool results are most of a transcript's bytes and never hold a typed prompt. Inside a JSON
    // string a quote is escaped, so this key only matches as a key.
    if !mentions_commit && text.contains("\"tool_use_id\"") {
        return;
    }
    let Ok(line) = serde_json::from_str::<Line<'_>>(text) else {
        return;
    };
    let Some(content) = line.message.and_then(|message| message.content) else {
        return;
    };
    if mentions_commit && calls_commit(content) {
        scan.saved = true;
    }
    let injected = line.is_meta == Some(true)
        || line.is_sidechain == Some(true)
        || line.is_compact_summary == Some(true);
    if line.kind.as_deref() == Some("user") && !injected {
        if let Some(prompt) = typed_prompt(content) {
            scan.prompts.push(prompt);
        }
    }
}

fn calls_commit(content: &RawValue) -> bool {
    let Ok(blocks) = serde_json::from_str::<Vec<ContentBlock>>(content.get()) else {
        return false;
    };
    blocks.iter().any(|block| {
        block.kind.as_deref() == Some("tool_use")
            && block
                .name
                .as_deref()
                .is_some_and(|name| COMMIT_TOOLS.contains(&name))
    })
}

/// The text the user typed in this message, without what the host or a hook added to it.
fn typed_prompt(content: &RawValue) -> Option<String> {
    let raw = content.get();
    let text = if raw.starts_with('"') {
        serde_json::from_str::<String>(raw).ok()?
    } else {
        let blocks: Vec<ContentBlock> = serde_json::from_str(raw).ok()?;
        if blocks
            .iter()
            .any(|block| block.kind.as_deref() == Some("tool_result"))
        {
            return None;
        }
        blocks
            .into_iter()
            .filter(|block| block.kind.as_deref() == Some("text"))
            .filter_map(|block| block.text)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let cleaned = strip_injected(&text);
    let cleaned = cleaned.trim();
    let from_host = INJECTED_PREFIXES
        .iter()
        .any(|prefix| cleaned.starts_with(prefix));
    (!cleaned.is_empty() && !from_host).then(|| cleaned.to_owned())
}

fn strip_injected(text: &str) -> String {
    let mut out = text.to_owned();
    for (open, close) in INJECTED_BLOCKS {
        while let Some(start) = out.find(open) {
            let end = out[start..]
                .find(close)
                .map_or(out.len(), |at| start + at + close.len());
            out.replace_range(start..end, "");
        }
    }
    out
}

fn pending_dir(dir: &Path) -> PathBuf {
    dir.join("pending")
}

fn pending_file(dir: &Path, session_id: &str) -> PathBuf {
    pending_dir(dir).join(format!("{}.txt", hashed(session_id)))
}

/// Adds `prompts` after the file's own, redacted, keeping the newest that fit
/// [`MAX_PENDING_CHARS`]. One JSON string per line, mode 0600.
fn append_pending(dir: &Path, session_id: &str, prompts: Vec<String>) -> Result<(), String> {
    let path = pending_file(dir, session_id);
    let mut all = read_prompts(&path);
    all.extend(prompts.iter().map(|prompt| redact(prompt)));
    let kept = keep_newest(all, MAX_PENDING_CHARS);
    let mut text = String::new();
    for prompt in &kept {
        text.push_str(&serde_json::to_string(prompt).map_err(|e| e.to_string())?);
        text.push('\n');
    }
    write_private(&path, text.as_bytes())
}

fn read_prompts(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .map(|text| {
            text.lines()
                .filter_map(|line| serde_json::from_str::<String>(line).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// The newest prompts whose printed lines fit in `cap`, oldest first. When even the newest does not
/// fit alone, its start is kept.
fn keep_newest(prompts: Vec<String>, cap: usize) -> Vec<String> {
    let mut kept = Vec::new();
    let mut used = 0;
    for prompt in prompts.into_iter().rev() {
        let size = line(&prompt).chars().count();
        if used + size <= cap {
            used += size;
            kept.push(prompt);
        } else {
            if kept.is_empty() {
                kept.push(truncate_to_fit(&prompt, cap));
            }
            break;
        }
    }
    kept.reverse();
    kept
}

/// The longest start of `prompt` whose printed line fits in `cap`.
fn truncate_to_fit(prompt: &str, cap: usize) -> String {
    let mut used = line("").chars().count();
    let mut end = 0;
    for (at, c) in prompt.char_indices() {
        // The escaped form of one character, without the quotes around it.
        let size = data_text(c.encode_utf8(&mut [0; 4])).chars().count() - 2;
        if used + size > cap {
            break;
        }
        used += size;
        end = at + c.len_utf8();
    }
    prompt[..end].to_owned()
}

/// One session's prompts waiting to be shown.
#[derive(Debug)]
pub struct Pending {
    pub path: PathBuf,
    pub prompts: Vec<String>,
}

/// Every pending file, oldest first. Files older than 14 days are deleted instead.
pub fn load_pending(dir: &Path, now: SystemTime) -> Vec<Pending> {
    let mut found: Vec<(SystemTime, PathBuf)> = live_files(dir, now)
        .into_iter()
        .filter(|(_, path)| {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            name.is_some_and(|name| name.ends_with(".txt") && !name.starts_with('.'))
        })
        .collect();
    found.sort();
    found
        .into_iter()
        .map(|(_, path)| Pending {
            prompts: read_prompts(&path),
            path,
        })
        .collect()
}

/// Deletes pending files older than 14 days, including temporary files left by a killed process.
pub fn remove_expired(dir: &Path, now: SystemTime) {
    live_files(dir, now);
}

/// The files in the pending directory with their times, after deleting the expired ones.
fn live_files(dir: &Path, now: SystemTime) -> Vec<(SystemTime, PathBuf)> {
    let Ok(entries) = fs::read_dir(pending_dir(dir)) else {
        return Vec::new();
    };
    let mut live = Vec::new();
    for entry in entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let modified = metadata.modified().unwrap_or(now);
        let expired = now
            .duration_since(modified)
            .is_ok_and(|age| age > MAX_PENDING_AGE);
        if expired {
            let _ = fs::remove_file(entry.path());
        } else {
            live.push((modified, entry.path()));
        }
    }
    live
}

/// Best effort: a file left behind is shown again, which is the safe direction.
pub fn remove(path: &Path) {
    let _ = fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::time::Instant;

    use serde_json::json;

    use super::*;

    fn user(content: serde_json::Value) -> String {
        json!({"type": "user", "message": {"role": "user", "content": content}}).to_string()
    }

    fn assistant_calls(tool: &str) -> String {
        json!({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "saving"},
            {"type": "tool_use", "id": "t1", "name": tool, "input": {"facts": []}},
        ]}})
        .to_string()
    }

    fn tool_result(text: &str) -> String {
        json!({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": text},
        ]}})
        .to_string()
    }

    fn scan_lines(lines: &[String]) -> Scan {
        let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
        scan(Cursor::new(text.into_bytes()))
    }

    #[test]
    fn extraction_keeps_only_what_the_user_typed() {
        let lines = [
            user(json!("my editor is helix")),
            user(json!([{"type": "text", "text": "we deploy on Fridays"}, {"type": "image"}])),
            tool_result("I am a tool result, not a prompt"),
            json!({"type": "user", "isMeta": true, "message": {"content": "Stop hook feedback: x"}})
                .to_string(),
            json!({"type": "user", "isSidechain": true, "message": {"content": "subagent task"}})
                .to_string(),
            json!({"type": "user", "isCompactSummary": true, "message": {"content": "summary"}})
                .to_string(),
            user(json!("<task-notification>\n<task-id>1</task-id></task-notification>")),
            user(json!("<bash-input>ls</bash-input>")),
            user(json!("[Request interrupted by user]")),
            user(json!(
                "<system-reminder>\nhook text\n</system-reminder>\ncall me Fer"
            )),
            user(json!(format!(
                "{MEMORIES_OPEN}\n- \"old fact\"\n{MEMORIES_CLOSE}"
            ))),
            user(json!("   ")),
            json!({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}})
                .to_string(),
            "not json".to_owned(),
        ];
        let scan = scan_lines(&lines);
        assert_eq!(
            scan.prompts,
            ["my editor is helix", "we deploy on Fridays", "call me Fer"]
        );
        assert!(!scan.saved);
    }

    #[test]
    fn commit_facts_is_detected_under_both_tool_names() {
        for tool in COMMIT_TOOLS {
            let scan = scan_lines(&[user(json!("my name is Ada")), assistant_calls(tool)]);
            assert!(scan.saved, "{tool}");
        }
        for other in [
            "mcp__memoryos__recall",
            "commit_facts",
            "mcp__x__commit_facts",
        ] {
            let scan = scan_lines(&[assistant_calls(other)]);
            assert!(!scan.saved, "{other}");
        }
        // Typing the tool's name is not calling it.
        let scan = scan_lines(&[user(json!("why did mcp__memoryos__commit_facts fail?"))]);
        assert!(!scan.saved);
        assert_eq!(scan.prompts.len(), 1);
    }

    #[test]
    fn a_partial_last_line_is_not_consumed() {
        let whole = user(json!("first"));
        let text = format!("{whole}\n{{\"type\":\"user\"");
        let scan = scan(Cursor::new(text.into_bytes()));
        assert_eq!(scan.prompts, ["first"]);
        assert_eq!(scan.consumed, whole.len() as u64 + 1);
    }

    #[test]
    fn secrets_are_redacted() {
        let cases = [
            ("key sk-abcdefghijklmnop1234 ok", "key [redacted] ok"),
            ("ghp_abcdefghijklmnopqrstuvwxyz0123", "[redacted]"),
            ("github_pat_11ABCDEFG0123456789_abcdef", "[redacted]"),
            ("aws AKIAIOSFODNN7EXAMPLE here", "aws [redacted] here"),
            ("use mos_0123456789abcdef", "use [redacted]"),
            ("slack xoxb-1234-5678-abcdefghij", "slack [redacted]"),
            ("xoxp-1234567890-abc", "[redacted]"),
            ("password=hunter2 then", "[redacted] then"),
            ("API_KEY: abc123", "[redacted]"),
            ("db Password = s3cret!", "db [redacted]"),
            (
                "sha 0123456789abcdef0123456789abcdef0123 done",
                "sha [redacted] done",
            ),
            (
                "token aGVsbG8gd29ybGQgdGhpcyBpcyBiYXNlNjQK1Z== end",
                "token [redacted] end",
            ),
            (
                "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----\nafter",
                "[redacted]\nafter",
            ),
            (
                "-----BEGIN PRIVATE KEY-----\nMIIabc unterminated",
                "[redacted]",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(redact(input), expected, "{input}");
        }
    }

    #[test]
    fn ordinary_text_is_not_redacted() {
        for text in [
            "my editor is helix",
            "the task-runner sk-ip is fine",
            "/Users/someone/Project/memoryos/crates/hooks/src/capture.rs",
            "the_plugin_hooks_match_what_install_writes_and_more",
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmnop",
            "short hex deadbeef",
            "we use a token bucket",
        ] {
            assert_eq!(redact(text), text, "{text}");
        }
    }

    #[test]
    fn every_pattern_compiles() {
        assert_eq!(secret_patterns().len(), SECRET_PATTERNS.len());
    }

    #[test]
    fn the_newest_prompts_are_kept_under_the_cap() {
        let prompts: Vec<String> = (0..100)
            .map(|i| format!("{i:03}{}", "x".repeat(97)))
            .collect();
        let kept = keep_newest(prompts, MAX_PENDING_CHARS);
        let size: usize = kept.iter().map(|p| line(p).chars().count()).sum();
        assert!(size <= MAX_PENDING_CHARS, "{size}");
        assert!(kept.last().expect("newest").starts_with("099"));
        let numbers: Vec<&str> = kept.iter().map(|p| &p[..3]).collect();
        let mut sorted = numbers.clone();
        sorted.sort_unstable();
        assert_eq!(numbers, sorted, "oldest first");
        assert!(kept.len() < 100);
    }

    #[test]
    fn one_huge_prompt_keeps_its_start() {
        let huge = format!("start {}", "<\n".repeat(10_000));
        let kept = keep_newest(vec!["older".to_owned(), huge], MAX_PENDING_CHARS);
        assert_eq!(kept.len(), 1);
        assert!(kept[0].starts_with("start "));
        let size = line(&kept[0]).chars().count();
        assert!(
            size <= MAX_PENDING_CHARS && size > MAX_PENDING_CHARS - 10,
            "{size}"
        );
    }

    fn write_transcript(path: &Path, lines: &[String]) {
        let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
        fs::write(path, text).expect("transcript");
    }

    fn append_transcript(path: &Path, lines: &[String]) {
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open");
        for line in lines {
            writeln!(file, "{line}").expect("append");
        }
    }

    #[test]
    fn capture_writes_a_private_redacted_file_and_skips_saved_parts() {
        let dir = tempfile::tempdir().expect("dir");
        let state = dir.path().join("state");
        let transcript = dir.path().join("t.jsonl");
        let now = SystemTime::now();
        write_transcript(
            &transcript,
            &[user(json!("my key is sk-abcdefghijklmnop1234"))],
        );
        capture(&state, "s", &transcript, now).expect("capture");
        let file = pending_file(&state, "s");
        assert_eq!(read_prompts(&file), ["my key is [redacted]"]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&file).expect("stat").permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        // Only what came after the last capture is read: nothing new, nothing added.
        capture(&state, "s", &transcript, now).expect("again");
        assert_eq!(read_prompts(&file).len(), 1);

        // A part with a save adds nothing.
        append_transcript(
            &transcript,
            &[
                user(json!("I prefer tabs")),
                assistant_calls(COMMIT_TOOLS[1]),
            ],
        );
        capture(&state, "s", &transcript, now).expect("saved part");
        assert_eq!(read_prompts(&file).len(), 1);

        // A later part without one is added after the first.
        append_transcript(&transcript, &[user(json!("we use Postgres"))]);
        capture(&state, "s", &transcript, now).expect("later part");
        assert_eq!(
            read_prompts(&file),
            ["my key is [redacted]", "we use Postgres"]
        );
    }

    #[test]
    fn a_session_that_saved_writes_nothing() {
        let dir = tempfile::tempdir().expect("dir");
        let transcript = dir.path().join("t.jsonl");
        write_transcript(
            &transcript,
            &[
                user(json!("my name is Ada")),
                assistant_calls(COMMIT_TOOLS[0]),
            ],
        );
        capture(dir.path(), "s", &transcript, SystemTime::now()).expect("capture");
        assert!(!pending_dir(dir.path()).exists());
    }

    #[test]
    fn a_missing_transcript_is_an_error() {
        let dir = tempfile::tempdir().expect("dir");
        let missing = dir.path().join("missing.jsonl");
        assert!(capture(dir.path(), "s", &missing, SystemTime::now()).is_err());
    }

    #[test]
    fn capture_is_fast_on_a_large_transcript() {
        let dir = tempfile::tempdir().expect("dir");
        let transcript = dir.path().join("t.jsonl");
        let big = tool_result(&"output line\n".repeat(2_000));
        let mut lines = vec![user(json!("my editor is helix"))];
        lines.extend(std::iter::repeat_n(big, 1_000));
        lines.push(user(json!("we deploy on Fridays")));
        write_transcript(&transcript, &lines);
        let size = fs::metadata(&transcript).expect("stat").len();
        assert!(size > 20_000_000, "{size}");

        let started = Instant::now();
        capture(dir.path(), "s", &transcript, SystemTime::now()).expect("capture");
        let took = started.elapsed();
        assert!(took < Duration::from_secs(5), "{took:?}");
        assert_eq!(
            read_prompts(&pending_file(dir.path(), "s")),
            ["my editor is helix", "we deploy on Fridays"]
        );
    }

    #[test]
    fn pending_files_load_oldest_first_and_expire_after_fourteen_days() {
        let dir = tempfile::tempdir().expect("dir");
        let now = SystemTime::now();
        let day = Duration::from_secs(24 * 60 * 60);
        let pending = pending_dir(dir.path());
        fs::create_dir_all(&pending).expect("mkdir");
        let ages = [("new", 1), ("old", 5), ("expired", 15)];
        for (name, days) in ages {
            let path = pending_file(dir.path(), name);
            fs::write(&path, format!("{}\n", json!(name))).expect("seed");
            let file = fs::File::options().write(true).open(&path).expect("open");
            file.set_modified(now - day * days).expect("time");
        }
        let stale_temp = pending.join(".x.txt.1.2.tmp");
        fs::write(&stale_temp, "").expect("seed");
        fs::File::options()
            .write(true)
            .open(&stale_temp)
            .expect("open")
            .set_modified(now - day * 20)
            .expect("time");

        let loaded = load_pending(dir.path(), now);
        let prompts: Vec<&str> = loaded.iter().map(|p| p.prompts[0].as_str()).collect();
        assert_eq!(prompts, ["old", "new"]);
        assert!(!pending_file(dir.path(), "expired").exists());
        assert!(!stale_temp.exists());
    }

    #[test]
    fn a_missing_pending_directory_is_empty() {
        let dir = tempfile::tempdir().expect("dir");
        assert!(load_pending(&dir.path().join("missing"), SystemTime::now()).is_empty());
    }
}
