//! `memoryos stats`: how much the agent used MemoryOS, and how well the prompt cache held, read from
//! Claude Code transcripts. Only counts leave this module. Transcript text can hold secrets, so none
//! of it is ever printed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::config::{self, process_env, Env};
use crate::hook::OPEN as MEMORIES_OPEN;

/// Tool names when the server comes from `memoryos setup` (`mcp__<server>__<tool>`) and when it
/// comes from the plugin (`mcp__plugin_<plugin>_<server>__<tool>`).
const TOOL_PREFIXES: [&str; 2] = ["mcp__memoryos__", "mcp__plugin_memoryos_memoryos__"];
const RECALL: &str = "recall";
const RECALL_CANDIDATES: &str = "recall_candidates";
const COMMIT_FACTS: &str = "commit_facts";
const SECONDS_PER_DAY: u64 = 86_400;

pub fn run(dir: Option<&Path>, days: u64, json: bool) -> Result<(), String> {
    let env: Env<'_> = &process_env;
    let dir = match dir {
        Some(dir) => dir.to_path_buf(),
        None => config::home(env)?.join(".claude").join("projects"),
    };
    let since = SystemTime::now()
        .checked_sub(Duration::from_secs(days.saturating_mul(SECONDS_PER_DAY)))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let stats = scan(&dir, since)?;
    if json {
        let text = serde_json::to_string_pretty(&stats).map_err(|e| e.to_string())?;
        println!("{text}");
    } else {
        print!("{}", render(&stats, &dir, days));
    }
    Ok(())
}

#[derive(Debug, Default, PartialEq, Serialize)]
pub(crate) struct Stats {
    pub sessions: usize,
    /// Calls per MemoryOS tool, by the tool's own name.
    pub tool_calls: BTreeMap<String, u64>,
    pub sessions_with_recall: usize,
    pub sessions_with_save: usize,
    pub sessions_with_hook_memories: usize,
    pub cache: Cache,
}

#[derive(Debug, Default, PartialEq, Serialize)]
pub(crate) struct Cache {
    /// API responses counted, each once.
    pub responses: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub input_tokens: u64,
    /// read / (read + creation + input) over every response, absent when there were no tokens.
    pub read_share: Option<f64>,
    /// The median of each session's read share, over sessions with tokens.
    pub median_session_read_share: Option<f64>,
}

#[derive(Debug, Default)]
struct Session {
    tools: HashMap<String, u64>,
    hook_memories: bool,
    read: u64,
    creation: u64,
    input: u64,
    responses: u64,
}

#[derive(Deserialize)]
struct Line<'a> {
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(rename = "isSidechain", default)]
    is_sidechain: bool,
    message: Option<Message>,
    #[serde(borrow)]
    attachment: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct Message {
    id: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    input_tokens: u64,
}

/// Scans `<dir>/*/*.jsonl` modified at or after `since`. A file or line that cannot be read is
/// skipped.
pub(crate) fn scan(dir: &Path, since: SystemTime) -> Result<Stats, String> {
    let projects = fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    let mut files: Vec<PathBuf> = Vec::new();
    for project in projects.flatten() {
        let Ok(entries) = fs::read_dir(project.path()) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let recent = entry
                .metadata()
                .and_then(|meta| meta.modified())
                .is_ok_and(|modified| modified >= since);
            if recent && path.extension().is_some_and(|ext| ext == "jsonl") && path.is_file() {
                files.push(path);
            }
        }
    }
    let sessions: Vec<Session> = files.iter().filter_map(|path| read_session(path)).collect();
    Ok(summarize(&sessions))
}

fn read_session(path: &Path) -> Option<Session> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut session = Session::default();
    let mut responses: HashSet<String> = HashSet::new();
    let mut tool_uses: HashSet<String> = HashSet::new();
    let mut buffer = Vec::new();
    loop {
        buffer.clear();
        match reader.read_until(b'\n', &mut buffer) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let Ok(line) = serde_json::from_slice::<Line<'_>>(&buffer) else {
            continue;
        };
        add_line(&line, &mut session, &mut responses, &mut tool_uses);
    }
    Some(session)
}

fn add_line(
    line: &Line<'_>,
    session: &mut Session,
    responses: &mut HashSet<String>,
    tool_uses: &mut HashSet<String>,
) {
    match line.kind.as_deref() {
        Some("attachment")
            if line
                .attachment
                .is_some_and(|raw| raw.get().contains(MEMORIES_OPEN)) =>
        {
            session.hook_memories = true;
        }
        Some("assistant") => {
            let Some(message) = &line.message else {
                return;
            };
            add_tool_uses(message, session, tool_uses);
            if line.is_sidechain {
                return;
            }
            let (Some(id), Some(usage)) = (&message.id, &message.usage) else {
                return;
            };
            // One response is written as several lines that repeat its id and usage.
            if responses.insert(id.clone()) {
                session.responses += 1;
                session.read += usage.cache_read_input_tokens;
                session.creation += usage.cache_creation_input_tokens;
                session.input += usage.input_tokens;
            }
        }
        _ => {}
    }
}

fn add_tool_uses(message: &Message, session: &mut Session, tool_uses: &mut HashSet<String>) {
    let Some(serde_json::Value::Array(blocks)) = &message.content else {
        return;
    };
    for block in blocks {
        if block["type"] != "tool_use" {
            continue;
        }
        let Some(name) = block["name"].as_str() else {
            continue;
        };
        let Some(tool) = TOOL_PREFIXES
            .iter()
            .find_map(|prefix| name.strip_prefix(prefix))
        else {
            continue;
        };
        // A block repeated on another line keeps its id, so it counts once.
        if let Some(id) = block["id"].as_str() {
            if !tool_uses.insert(id.to_owned()) {
                continue;
            }
        }
        *session.tools.entry(tool.to_owned()).or_default() += 1;
    }
}

fn summarize(sessions: &[Session]) -> Stats {
    let mut stats = Stats {
        sessions: sessions.len(),
        ..Stats::default()
    };
    for tool in [RECALL, RECALL_CANDIDATES, COMMIT_FACTS] {
        stats.tool_calls.insert(tool.to_owned(), 0);
    }
    let mut shares = Vec::new();
    for session in sessions {
        for (tool, count) in &session.tools {
            *stats.tool_calls.entry(tool.clone()).or_default() += count;
        }
        if session.tools.contains_key(RECALL) {
            stats.sessions_with_recall += 1;
        }
        if session.tools.contains_key(COMMIT_FACTS) {
            stats.sessions_with_save += 1;
        }
        if session.hook_memories {
            stats.sessions_with_hook_memories += 1;
        }
        let cache = &mut stats.cache;
        cache.responses += session.responses;
        cache.cache_read_input_tokens += session.read;
        cache.cache_creation_input_tokens += session.creation;
        cache.input_tokens += session.input;
        if let Some(share) = share(session.read, session.creation, session.input) {
            shares.push(share);
        }
    }
    let cache = &mut stats.cache;
    cache.read_share = share(
        cache.cache_read_input_tokens,
        cache.cache_creation_input_tokens,
        cache.input_tokens,
    );
    cache.median_session_read_share = median(&mut shares);
    stats
}

fn share(read: u64, creation: u64, input: u64) -> Option<f64> {
    let total = read + creation + input;
    (total > 0).then(|| read as f64 / total as f64)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        f64::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    })
}

fn percent(share: Option<f64>) -> String {
    share.map_or_else(
        || "n/a".to_owned(),
        |share| format!("{:.1}%", share * 100.0),
    )
}

fn render(stats: &Stats, dir: &Path, days: u64) -> String {
    let mut out = format!(
        "Sessions scanned: {} (last {days} days, {})\n\nMemoryOS tool calls:\n",
        stats.sessions,
        dir.display()
    );
    for (tool, count) in &stats.tool_calls {
        out.push_str(&format!("  {tool:<20} {count}\n"));
    }
    let cache = &stats.cache;
    out.push_str(&format!(
        "\nSessions with a recall:        {}\n\
         Sessions with a save:          {}\n\
         Sessions with hook memories:   {}\n\
         \nPrompt cache, main thread ({} responses):\n\
         \x20 cache read tokens      {}\n\
         \x20 cache creation tokens  {}\n\
         \x20 input tokens           {}\n\
         \x20 read share             {}\n\
         \x20 median session share   {}\n",
        stats.sessions_with_recall,
        stats.sessions_with_save,
        stats.sessions_with_hook_memories,
        cache.responses,
        cache.cache_read_input_tokens,
        cache.cache_creation_input_tokens,
        cache.input_tokens,
        percent(cache.read_share),
        percent(cache.median_session_read_share),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(id: &str, tools: &[(&str, &str)], read: u64, creation: u64, input: u64) -> String {
        let content: Vec<serde_json::Value> = tools
            .iter()
            .map(|(block_id, name)| {
                serde_json::json!({"type": "tool_use", "id": block_id, "name": name, "input": {}})
            })
            .collect();
        serde_json::json!({
            "type": "assistant",
            "isSidechain": false,
            "message": {
                "id": id,
                "role": "assistant",
                "content": content,
                "usage": {
                    "cache_read_input_tokens": read,
                    "cache_creation_input_tokens": creation,
                    "input_tokens": input,
                    "output_tokens": 7,
                },
            },
        })
        .to_string()
    }

    fn hook_attachment() -> String {
        serde_json::json!({
            "type": "attachment",
            "attachment": {
                "type": "hook_success",
                "hookEvent": "SessionStart",
                "content": "<memoryos-memories>\nsecret value\n</memoryos-memories>",
            },
        })
        .to_string()
    }

    fn write(dir: &Path, project: &str, session: &str, lines: &[String]) {
        let project = dir.join(project);
        fs::create_dir_all(&project).expect("mkdir");
        fs::write(project.join(format!("{session}.jsonl")), lines.join("\n")).expect("write");
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("a share");
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    fn scan_all(dir: &Path) -> Stats {
        scan(dir, SystemTime::UNIX_EPOCH).expect("scan")
    }

    #[test]
    fn a_response_split_over_lines_counts_once() {
        let temp = tempfile::tempdir().expect("tempdir");
        // Claude Code writes one line per content block, each repeating the id and usage.
        write(
            temp.path(),
            "p",
            "s1",
            &[
                assistant("msg_1", &[], 900, 50, 50),
                assistant(
                    "msg_1",
                    &[("toolu_1", "mcp__memoryos__recall")],
                    900,
                    50,
                    50,
                ),
                assistant(
                    "msg_1",
                    &[("toolu_1", "mcp__memoryos__recall")],
                    900,
                    50,
                    50,
                ),
            ],
        );
        let stats = scan_all(temp.path());
        assert_eq!(stats.sessions, 1);
        assert_eq!(stats.cache.responses, 1);
        assert_eq!(stats.cache.cache_read_input_tokens, 900);
        assert_eq!(stats.cache.cache_creation_input_tokens, 50);
        assert_eq!(stats.cache.input_tokens, 50);
        assert_close(stats.cache.read_share, 0.9);
        assert_eq!(stats.tool_calls[RECALL], 1);
    }

    #[test]
    fn tools_sessions_and_hook_text_are_counted() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(
            temp.path(),
            "a",
            "s1",
            &[
                hook_attachment(),
                assistant("m1", &[("t1", "mcp__memoryos__recall")], 0, 100, 0),
                assistant(
                    "m2",
                    &[
                        ("t2", "mcp__memoryos__recall_candidates"),
                        ("t3", "mcp__memoryos__commit_facts"),
                        ("t4", "Bash"),
                    ],
                    100,
                    0,
                    0,
                ),
            ],
        );
        write(
            temp.path(),
            "b",
            "s2",
            &[
                "not json".to_owned(),
                assistant(
                    "m3",
                    &[("t5", "mcp__plugin_memoryos_memoryos__recall")],
                    30,
                    0,
                    70,
                ),
                assistant("m4", &[("t6", "mcp__memoryos__block_read")], 0, 0, 0),
                "{\"type\":\"assistant\"".to_owned(),
            ],
        );
        write(temp.path(), "b", "s3", &[]);
        let stats = scan_all(temp.path());
        assert_eq!(stats.sessions, 3);
        assert_eq!(stats.tool_calls[RECALL], 2);
        assert_eq!(stats.tool_calls[RECALL_CANDIDATES], 1);
        assert_eq!(stats.tool_calls[COMMIT_FACTS], 1);
        assert_eq!(stats.tool_calls["block_read"], 1);
        assert!(!stats.tool_calls.contains_key("Bash"));
        assert_eq!(stats.sessions_with_recall, 2);
        assert_eq!(stats.sessions_with_save, 1);
        assert_eq!(stats.sessions_with_hook_memories, 1);
        // s1: 100 / 200 = 0.5, s2: 30 / 100 = 0.3, s3 has no tokens.
        assert_close(stats.cache.median_session_read_share, 0.4);
        assert_close(stats.cache.read_share, 130.0 / 300.0);
    }

    #[test]
    fn sidechain_usage_is_skipped() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut side: serde_json::Value =
            serde_json::from_str(&assistant("m9", &[], 500, 500, 500)).expect("json");
        side["isSidechain"] = serde_json::json!(true);
        write(
            temp.path(),
            "p",
            "s1",
            &[side.to_string(), assistant("m1", &[], 10, 0, 0)],
        );
        let stats = scan_all(temp.path());
        assert_eq!(stats.cache.responses, 1);
        assert_eq!(stats.cache.cache_read_input_tokens, 10);
    }

    #[test]
    fn old_files_and_other_extensions_are_ignored() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "p", "s1", &[assistant("m1", &[], 1, 0, 0)]);
        fs::write(temp.path().join("p").join("notes.txt"), "x").expect("write");
        assert_eq!(scan_all(temp.path()).sessions, 1);
        let future = SystemTime::now() + Duration::from_secs(SECONDS_PER_DAY);
        assert_eq!(scan(temp.path(), future).expect("scan").sessions, 0);
    }

    #[test]
    fn output_never_holds_transcript_text() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "p", "s1", &[hook_attachment()]);
        let stats = scan_all(temp.path());
        let text = render(&stats, temp.path(), 14);
        let json = serde_json::to_string(&stats).expect("json");
        assert!(!text.contains("secret value") && !json.contains("secret value"));
        assert!(text.contains("Sessions with hook memories:   1"));
    }

    #[test]
    fn median_handles_odd_even_and_empty() {
        assert_eq!(median(&mut []), None);
        assert_close(median(&mut [0.3, 0.1, 0.2]), 0.2);
        assert_close(median(&mut [0.4, 0.2]), 0.3);
    }
}
