//! `hook install` and `hook uninstall`. Everything is planned first and written only when the whole
//! plan is valid, so a bad file means nothing is written anywhere. The API key is never read here,
//! so it cannot reach a file: configuration holds a reference to the variable.

use std::fs;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use serde_json::{json, Value};

use crate::config::{self, process_env, Env};
use crate::doc::Doc;
use crate::files::{back_up, write_atomic};
use crate::InstallArgs;

const SERVER_NAME: &str = "memoryos";
const SESSION_START_EVENT: &str = "SessionStart";
const STOP_EVENT: &str = "Stop";
const SESSION_START_COMMAND: &str = "memoryos hook session-start";
const STOP_COMMAND: &str = "memoryos hook stop";
const SESSION_START_MATCHER: &str = "startup|resume|clear|compact";
/// Every matcher an install has written, so uninstall also removes what older installs left.
const SESSION_START_MATCHERS: &[&str] = &[SESSION_START_MATCHER, "startup|resume|compact"];
const CLAUDE_AUTHORIZATION: &str = "Bearer ${MEMORYOS_API_KEY}";
const OPENCODE_AUTHORIZATION: &str = "Bearer {env:MEMORYOS_API_KEY}";

/// From `README.md`, "OpenCode setup".
///
/// `.nothrow()` stops Bun Shell raising on a non-zero exit, so a missing `memoryos` cannot fail the
/// compaction hook: Bun 1.3 returns empty text for a command that does not exist.
const OPENCODE_PLUGIN: &str = r#"export const MemoryOS = async ({ $ }) => ({
  "experimental.session.compacting": async (_input, output) => {
    const memories = await $`memoryos hook session-start`.nothrow().text();
    if (memories.trim()) output.context.push(memories);
  },
});
"#;

/// What older installs wrote, so uninstall still recognizes it.
const OLD_OPENCODE_PLUGIN: &str = r#"export const MemoryOS = async ({ $ }) => ({
  "experimental.session.compacting": async (_input, output) => {
    const memories = await $`memoryos hook session-start`.text();
    output.context.push(memories);
  },
});
"#;

const KEY_NOTE: &str =
    "MEMORYOS_API_KEY must be set in the agent's environment. The key itself is \
                        not stored in any file.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Agent {
    ClaudeCode,
    Opencode,
}

pub fn run_install(args: &InstallArgs) -> Result<(), String> {
    let url = config::base_url(&process_env)?;
    let plans = plan(
        &Op::Install { base_url: url },
        args.agent,
        args.global,
        &current_dir()?,
        &process_env,
    )?;
    println!("{}", apply(&plans, args.dry_run)?);
    println!("{KEY_NOTE}");
    Ok(())
}

pub fn run_uninstall(args: &InstallArgs) -> Result<(), String> {
    let plans = plan(
        &Op::Uninstall,
        args.agent,
        args.global,
        &current_dir()?,
        &process_env,
    )?;
    println!("{}", apply(&plans, args.dry_run)?);
    Ok(())
}

fn current_dir() -> Result<PathBuf, String> {
    std::env::current_dir().map_err(|e| format!("cannot read the current directory: {e}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Install { base_url: String },
    Uninstall,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum After {
    Unchanged,
    Write(String),
    Remove,
}

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    after: After,
    notes: Vec<String>,
}

#[derive(Debug)]
struct FilePlan {
    path: PathBuf,
    before: Option<String>,
    after: After,
    notes: Vec<String>,
}

fn plan(
    op: &Op,
    agent: Agent,
    global: bool,
    project: &Path,
    env: Env<'_>,
) -> Result<Vec<FilePlan>, String> {
    match agent {
        Agent::ClaudeCode => plan_claude(op, global, project, env),
        Agent::Opencode => plan_opencode(op, global, project, env),
    }
}

fn plan_claude(
    op: &Op,
    global: bool,
    project: &Path,
    env: Env<'_>,
) -> Result<Vec<FilePlan>, String> {
    let (settings, mcp) = if global {
        // User-scoped MCP servers live in ~/.claude.json, not in ~/.claude/settings.json.
        let home = config::home(env)?;
        (
            home.join(".claude").join("settings.json"),
            home.join(".claude.json"),
        )
    } else {
        (
            project.join(".claude").join("settings.json"),
            project.join(".mcp.json"),
        )
    };
    let plan_settings = plan_file(&settings, |existing| match op {
        Op::Install { .. } => install_claude_hooks(existing),
        Op::Uninstall => uninstall_claude_hooks(existing),
    })?;
    let plan_mcp = plan_file(&mcp, |existing| match op {
        Op::Install { base_url } => install_server(existing, "mcpServers", &claude_entry(base_url)),
        Op::Uninstall => uninstall_server(existing, "mcpServers", is_claude_entry),
    })?;
    Ok(vec![plan_settings, plan_mcp])
}

fn plan_opencode(
    op: &Op,
    global: bool,
    project: &Path,
    env: Env<'_>,
) -> Result<Vec<FilePlan>, String> {
    let dir = if global {
        config::opencode_config_dir(env)?
    } else {
        project.to_owned()
    };
    let plugin = if global {
        dir.join("plugins").join("memoryos.ts")
    } else {
        dir.join(".opencode").join("plugins").join("memoryos.ts")
    };
    let plan_plugin = plan_file(&plugin, |existing| match op {
        Op::Install { .. } => Ok(install_plugin(existing)),
        Op::Uninstall => Ok(uninstall_plugin(existing)),
    })?;
    let config_file = opencode_config_file(&dir);
    let plan_config = plan_file(&config_file, |existing| match op {
        Op::Install { base_url } => install_opencode_server(existing, base_url),
        Op::Uninstall => uninstall_opencode_server(existing),
    })
    .map_err(|message| match op {
        Op::Install { .. } if existing_has_comments(&config_file) => format!(
            "{message}\nAlso create {} with:\n{OPENCODE_PLUGIN}",
            plugin.display()
        ),
        _ => message,
    })?;
    Ok(vec![plan_plugin, plan_config])
}

fn existing_has_comments(path: &Path) -> bool {
    fs::read_to_string(path).is_ok_and(|text| has_comments(&text))
}

/// OpenCode reads either name. A new file is `opencode.json`.
fn opencode_config_file(dir: &Path) -> PathBuf {
    let json = dir.join("opencode.json");
    let jsonc = dir.join("opencode.jsonc");
    if !json.exists() && jsonc.exists() {
        jsonc
    } else {
        json
    }
}

fn plan_file(
    path: &Path,
    outcome: impl FnOnce(Option<&str>) -> Result<Outcome, String>,
) -> Result<FilePlan, String> {
    let before = read_existing(path)?;
    let Outcome { after, notes } = outcome(before.as_deref())
        .map_err(|message| format!("{}: {message}\nNothing was written.", path.display()))?;
    Ok(FilePlan {
        path: path.to_owned(),
        before,
        after,
        notes,
    })
}

fn read_existing(path: &Path) -> Result<Option<String>, String> {
    let is_link = fs::symlink_metadata(path).is_ok();
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !is_link => Ok(None),
        Err(error) => Err(format!(
            "cannot read {}: {error}\nNothing was written.",
            path.display()
        )),
    }
}

/// Writes the plan and describes it. With `dry_run` it only describes.
fn apply(plans: &[FilePlan], dry_run: bool) -> Result<String, String> {
    let mut lines = Vec::new();
    for plan in plans {
        let mut notes = plan.notes.clone();
        let verb = match &plan.after {
            After::Unchanged => "unchanged",
            After::Write(text) => {
                if !dry_run {
                    if let Some(backup) = write_plan(plan, text)? {
                        notes.push(format!("backup at {}", backup.display()));
                    }
                }
                match (plan.before.is_some(), dry_run) {
                    (true, false) => "updated",
                    (true, true) => "would update",
                    (false, false) => "created",
                    (false, true) => "would create",
                }
            }
            After::Remove => {
                if !dry_run {
                    if let Some(backup) = remove_plan(plan)? {
                        notes.push(format!("backup at {}", backup.display()));
                    }
                }
                if dry_run {
                    "would remove"
                } else {
                    "removed"
                }
            }
        };
        let detail = if notes.is_empty() {
            String::new()
        } else {
            format!(": {}", notes.join("; "))
        };
        lines.push(format!("{verb} {}{detail}", plan.path.display()));
    }
    Ok(lines.join("\n"))
}

/// A symbolic link stays one: a dotfile manager may own the link, so the file it points at is what
/// gets replaced.
fn real_path(path: &Path) -> Result<PathBuf, String> {
    if fs::symlink_metadata(path).is_ok() {
        fs::canonicalize(path).map_err(|e| format!("cannot resolve {}: {e}", path.display()))
    } else {
        Ok(path.to_owned())
    }
}

fn write_plan(plan: &FilePlan, text: &str) -> Result<Option<PathBuf>, String> {
    let target = real_path(&plan.path)?;
    let backup = plan
        .before
        .as_ref()
        .map(|before| back_up(&target, before.as_bytes()))
        .transpose()?;
    let permissions = fs::metadata(&target).ok().map(|m| m.permissions());
    write_atomic(&target, text.as_bytes(), permissions)?;
    Ok(backup)
}

fn remove_plan(plan: &FilePlan) -> Result<Option<PathBuf>, String> {
    let backup = plan
        .before
        .as_ref()
        .map(|before| back_up(&real_path(&plan.path)?, before.as_bytes()))
        .transpose()?;
    fs::remove_file(&plan.path)
        .map_err(|e| format!("cannot remove {}: {e}", plan.path.display()))?;
    Ok(backup)
}

// ---- Documents ---------------------------------------------------------------------------------

fn load_object(existing: Option<&str>) -> Result<Doc, String> {
    let Some(text) = existing else {
        return Ok(Doc::Obj(vec![]));
    };
    let doc = Doc::parse(text).map_err(|e| format!("not valid JSON ({e})"))?;
    match doc {
        Doc::Obj(_) => Ok(doc),
        _ => Err("the top level is not a JSON object".to_owned()),
    }
}

fn finish(root: &Doc, changed: bool, remove_when_empty: bool, notes: Vec<String>) -> Outcome {
    let after = if !changed {
        After::Unchanged
    } else if remove_when_empty && root.is_empty_container() {
        After::Remove
    } else {
        After::Write(root.render())
    };
    Outcome { after, notes }
}

/// The object at `key`, created if absent.
fn object_at<'a>(parent: &'a mut Doc, key: &str) -> Result<&'a mut Doc, String> {
    child_at(parent, key, Doc::Obj(vec![]), "an object")
}

/// The array at `key`, created if absent.
fn array_at<'a>(parent: &'a mut Doc, key: &str) -> Result<&'a mut Doc, String> {
    child_at(parent, key, Doc::Arr(vec![]), "an array")
}

fn child_at<'a>(
    parent: &'a mut Doc,
    key: &str,
    empty: Doc,
    expected: &str,
) -> Result<&'a mut Doc, String> {
    if parent.get(key).is_none() {
        parent.push_entry(key, empty.clone());
    }
    match parent.get_mut(key) {
        Some(child) if std::mem::discriminant(child) == std::mem::discriminant(&empty) => Ok(child),
        _ => Err(format!("\"{key}\" is not {expected}")),
    }
}

// ---- Claude Code hooks -------------------------------------------------------------------------

fn install_claude_hooks(existing: Option<&str>) -> Result<Outcome, String> {
    let mut root = load_object(existing)?;
    let mut notes = Vec::new();
    let hooks = object_at(&mut root, "hooks")?;
    let added_start = add_hook(
        hooks,
        SESSION_START_EVENT,
        Some(SESSION_START_MATCHER),
        SESSION_START_COMMAND,
        &mut notes,
    )?;
    let added_stop = add_hook(hooks, STOP_EVENT, None, STOP_COMMAND, &mut notes)?;
    Ok(finish(&root, added_start || added_stop, false, notes))
}

fn add_hook(
    hooks: &mut Doc,
    event: &str,
    matcher: Option<&str>,
    command: &str,
    notes: &mut Vec<String>,
) -> Result<bool, String> {
    let Doc::Arr(groups) = array_at(hooks, event)? else {
        return Err(format!("\"{event}\" is not an array"));
    };
    if groups.iter().any(|group| group_runs(group, command)) {
        notes.push(format!("{event} hook already present"));
        return Ok(false);
    }
    let handler = format!(r#"{{"type":"command","command":{}}}"#, json_string(command));
    let group = match matcher {
        Some(matcher) => format!(
            r#"{{"matcher":{},"hooks":[{handler}]}}"#,
            json_string(matcher)
        ),
        None => format!(r#"{{"hooks":[{handler}]}}"#),
    };
    groups.push(Doc::parse(&group).map_err(|e| e.to_string())?);
    notes.push(format!("added {event} hook"));
    Ok(true)
}

fn group_runs(group: &Doc, command: &str) -> bool {
    let Some(Doc::Arr(handlers)) = group.get("hooks") else {
        return false;
    };
    handlers.iter().any(|handler| runs(handler, command))
}

fn runs(handler: &Doc, command: &str) -> bool {
    handler
        .get("command")
        .is_some_and(|value| value.to_value() == json!(command))
}

fn uninstall_claude_hooks(existing: Option<&str>) -> Result<Outcome, String> {
    if existing.is_none() {
        return Ok(unchanged());
    }
    let mut root = load_object(existing)?;
    let mut notes = Vec::new();
    let mut changed = false;
    let Some(hooks) = root.get_mut("hooks") else {
        return Ok(unchanged());
    };
    if matches!(hooks, Doc::Obj(_)) {
        changed |= remove_hook(
            hooks,
            SESSION_START_EVENT,
            SESSION_START_MATCHERS,
            SESSION_START_COMMAND,
            &mut notes,
        );
        changed |= remove_hook(hooks, STOP_EVENT, &[], STOP_COMMAND, &mut notes);
        if changed && hooks.is_empty_container() {
            root.remove_key("hooks");
        }
    }
    Ok(finish(&root, changed, true, notes))
}

/// Removes only the handler `install` writes, from a group with a matcher `install` writes, where
/// an empty list means no matcher. A handler the user changed, for example by adding a timeout, is
/// theirs now and is reported.
fn remove_hook(
    hooks: &mut Doc,
    event: &str,
    matchers: &[&str],
    command: &str,
    notes: &mut Vec<String>,
) -> bool {
    let Some(Doc::Arr(groups)) = hooks.get_mut(event) else {
        return false;
    };
    let ours = json!({"type": "command", "command": command});
    let mut removed = false;
    let mut left = false;
    groups.retain_mut(|group| {
        let same_matcher = has_matcher(group, matchers);
        let only_known_keys = match group {
            Doc::Obj(entries) => entries.iter().all(|(k, _)| k == "matcher" || k == "hooks"),
            _ => false,
        };
        let Some(Doc::Arr(handlers)) = group.get_mut("hooks") else {
            return true;
        };
        let count = handlers.len();
        if same_matcher {
            handlers.retain(|handler| handler.to_value() != ours);
        }
        let shrunk = handlers.len() != count;
        removed |= shrunk;
        left |= handlers.iter().any(|handler| runs(handler, command));
        !(shrunk && handlers.is_empty() && only_known_keys)
    });
    if removed {
        notes.push(format!("removed {event} hook"));
    }
    if left {
        notes.push(format!(
            "left a {event} hook running \"{command}\" that differs from what install adds"
        ));
    }
    if removed && groups.is_empty() {
        hooks.remove_key(event);
    }
    removed
}

fn has_matcher(group: &Doc, matchers: &[&str]) -> bool {
    let actual = group.get("matcher").map(Doc::to_value);
    if matchers.is_empty() {
        return actual.is_none() || actual == Some(json!(""));
    }
    matchers
        .iter()
        .any(|matcher| actual == Some(json!(matcher)))
}

fn unchanged() -> Outcome {
    Outcome {
        after: After::Unchanged,
        notes: Vec::new(),
    }
}

// ---- MCP entries -------------------------------------------------------------------------------

fn mcp_url(base_url: &str) -> String {
    format!("{}/mcp", base_url.trim_end_matches('/'))
}

fn json_string(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_owned())
}

/// `.mcp.json` format, from https://code.claude.com/docs/en/mcp.
fn claude_entry(base_url: &str) -> String {
    format!(
        r#"{{"type":"http","url":{},"headers":{{"Authorization":{}}}}}"#,
        json_string(&mcp_url(base_url)),
        json_string(CLAUDE_AUTHORIZATION)
    )
}

/// `opencode.json` format, from https://opencode.ai/docs/mcp-servers/. OAuth is off because the
/// server takes a key, and OpenCode would otherwise try OAuth on a 401.
fn opencode_entry(base_url: &str) -> String {
    format!(
        r#"{{"type":"remote","url":{},"enabled":true,"oauth":false,"headers":{{"Authorization":{}}}}}"#,
        json_string(&mcp_url(base_url)),
        json_string(OPENCODE_AUTHORIZATION)
    )
}

fn is_claude_entry(value: &Value) -> bool {
    value["type"] == "http"
        && is_memoryos_url(&value["url"])
        && value["headers"] == json!({"Authorization": CLAUDE_AUTHORIZATION})
        && value.as_object().is_some_and(|object| object.len() == 3)
}

fn is_opencode_entry(value: &Value) -> bool {
    value["type"] == "remote"
        && is_memoryos_url(&value["url"])
        && value["enabled"] == json!(true)
        && value["oauth"] == json!(false)
        && value["headers"] == json!({"Authorization": OPENCODE_AUTHORIZATION})
        && value.as_object().is_some_and(|object| object.len() == 5)
}

/// The URL depends on `MEMORYOS_URL` at install time, so only its form can be checked afterwards.
fn is_memoryos_url(url: &Value) -> bool {
    url.as_str().is_some_and(|url| url.ends_with("/mcp"))
}

fn install_server(existing: Option<&str>, container: &str, entry: &str) -> Result<Outcome, String> {
    let mut root = load_object(existing)?;
    let wanted = Doc::parse(entry).map_err(|e| e.to_string())?;
    let servers = object_at(&mut root, container)?;
    let mut notes = Vec::new();
    let changed = match servers.get(SERVER_NAME) {
        None => {
            servers.push_entry(SERVER_NAME, wanted);
            notes.push(format!("added MCP server {SERVER_NAME}"));
            true
        }
        Some(current) if current.to_value() == wanted.to_value() => {
            notes.push(format!("MCP server {SERVER_NAME} already present"));
            false
        }
        Some(_) => {
            notes.push(format!(
                "left MCP server {SERVER_NAME} as it is, because it differs from what install \
                 would add. Uninstall or remove it to replace it"
            ));
            false
        }
    };
    Ok(finish(&root, changed, false, notes))
}

fn uninstall_server(
    existing: Option<&str>,
    container: &str,
    is_ours: fn(&Value) -> bool,
) -> Result<Outcome, String> {
    if existing.is_none() {
        return Ok(unchanged());
    }
    let mut root = load_object(existing)?;
    let Some(servers) = root.get_mut(container) else {
        return Ok(unchanged());
    };
    let Some(current) = servers.get(SERVER_NAME) else {
        return Ok(unchanged());
    };
    if !is_ours(&current.to_value()) {
        let note = format!("left MCP server {SERVER_NAME}, because install did not write it");
        return Ok(Outcome {
            after: After::Unchanged,
            notes: vec![note],
        });
    }
    servers.remove_key(SERVER_NAME);
    if servers.is_empty_container() {
        root.remove_key(container);
    }
    let notes = vec![format!("removed MCP server {SERVER_NAME}")];
    Ok(finish(&root, true, true, notes))
}

// ---- OpenCode ----------------------------------------------------------------------------------

fn install_opencode_server(existing: Option<&str>, base_url: &str) -> Result<Outcome, String> {
    let entry = opencode_entry(base_url);
    if existing.is_some_and(has_comments) {
        let pretty = Doc::parse(&entry).map_err(|e| e.to_string())?.render();
        return Err(format!(
            "this file has comments, which rewriting it would destroy. Add this by hand under \
             \"mcp\":\n\"{SERVER_NAME}\": {pretty}"
        ));
    }
    install_server(existing, "mcp", &entry)
}

fn uninstall_opencode_server(existing: Option<&str>) -> Result<Outcome, String> {
    if existing.is_some_and(has_comments) {
        if existing.is_some_and(|text| text.contains(SERVER_NAME)) {
            return Err(format!(
                "this file has comments, which rewriting it would destroy. Remove \
                 \"{SERVER_NAME}\" from \"mcp\" by hand"
            ));
        }
        return Ok(unchanged());
    }
    uninstall_server(existing, "mcp", is_opencode_entry)
}

/// True when the text has a `//` or `/*` outside a string.
fn has_comments(text: &str) -> bool {
    let mut in_string = false;
    let mut escaped = false;
    let mut previous_slash = false;
    for c in text.chars() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        if previous_slash && matches!(c, '/' | '*') {
            return true;
        }
        previous_slash = c == '/';
        in_string = c == '"';
    }
    false
}

fn install_plugin(existing: Option<&str>) -> Outcome {
    match existing {
        None => Outcome {
            after: After::Write(OPENCODE_PLUGIN.to_owned()),
            notes: vec!["added the MemoryOS plugin".to_owned()],
        },
        Some(OPENCODE_PLUGIN) => Outcome {
            after: After::Unchanged,
            notes: vec!["plugin already present".to_owned()],
        },
        Some(_) => Outcome {
            after: After::Unchanged,
            notes: vec!["left this file as it is, because it differs from the plugin".to_owned()],
        },
    }
}

fn uninstall_plugin(existing: Option<&str>) -> Outcome {
    match existing {
        None => unchanged(),
        Some(OPENCODE_PLUGIN | OLD_OPENCODE_PLUGIN) => Outcome {
            after: After::Remove,
            notes: vec!["removed the MemoryOS plugin".to_owned()],
        },
        Some(_) => Outcome {
            after: After::Unchanged,
            notes: vec!["left this file, because install did not write it".to_owned()],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const URL: &str = "http://localhost:8080";

    fn written(outcome: Outcome) -> String {
        match outcome.after {
            After::Write(text) => text,
            other => panic!("expected a write, got {other:?}"),
        }
    }

    fn value(text: &str) -> Value {
        serde_json::from_str(text).expect("valid JSON")
    }

    fn start_group() -> Value {
        json!({
            "matcher": "startup|resume|clear|compact",
            "hooks": [{"type": "command", "command": "memoryos hook session-start"}],
        })
    }

    fn stop_group() -> Value {
        json!({"hooks": [{"type": "command", "command": "memoryos hook stop"}]})
    }

    // ---- Claude hooks ----

    #[test]
    fn hooks_into_nothing_match_the_readme() {
        let text = written(install_claude_hooks(None).expect("install"));
        assert_eq!(
            value(&text),
            json!({"hooks": {"SessionStart": [start_group()], "Stop": [stop_group()]}})
        );
    }

    #[test]
    fn hooks_merge_into_many_existing_shapes() {
        let shapes = [
            "{}",
            r#"{"model":"x"}"#,
            r#"{"hooks":{}}"#,
            r#"{"hooks":{"Stop":[]}}"#,
            r#"{"hooks":{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"a"}]}]}}"#,
            r#"{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"foreign"}]}]}}"#,
            r#"{"hooks":{"SessionStart":[{"matcher":"startup","hooks":[{"type":"command","command":"b","timeout":5}]}]}}"#,
            r#"{"permissions":{"allow":["Bash(ls)"]},"env":{"A":"1"},"hooks":{"Stop":[{"hooks":[]}]}}"#,
            r#"{"hooks":{"Stop":["odd entry",{"hooks":"odd"}]}}"#,
        ];
        for shape in shapes {
            let original = value(shape);
            let merged = value(&written(install_claude_hooks(Some(shape)).expect(shape)));

            for (key, kept) in original.as_object().expect("object") {
                if key != "hooks" {
                    assert_eq!(&merged[key], kept, "{shape}");
                }
            }
            for (event, groups) in original["hooks"].as_object().into_iter().flatten() {
                for group in groups.as_array().expect("array") {
                    let now = merged["hooks"][event].as_array().expect("array");
                    assert!(now.contains(group), "lost {group} from {shape}");
                }
            }
            let start = merged["hooks"]["SessionStart"].as_array().expect("array");
            assert!(start.contains(&start_group()), "{shape}");
            let stop = merged["hooks"]["Stop"].as_array().expect("array");
            assert!(stop.contains(&stop_group()), "{shape}");
        }
    }

    #[test]
    fn foreign_entries_keep_their_order_and_exact_numbers() {
        let shape = r#"{"z":1.10,"big":12345678901234567890123,"hooks":{"Stop":[{"hooks":[{"type":"command","command":"foreign","timeout":30.50}]}]},"a":"\u00e9"}"#;
        let text = written(install_claude_hooks(Some(shape)).expect("install"));
        for kept in ["1.10", "12345678901234567890123", "30.50", "\\u00e9"] {
            assert!(text.contains(kept), "{kept} in {text}");
        }
        let z = text.find("\"z\"").expect("z");
        let hooks = text.find("\"hooks\"").expect("hooks");
        let a = text.find("\"a\"").expect("a");
        assert!(z < hooks && hooks < a, "{text}");
        let foreign = text.find("foreign").expect("foreign");
        assert!(foreign < text.find(STOP_COMMAND).expect("ours"));
    }

    #[test]
    fn hooks_install_is_idempotent() {
        let first = written(install_claude_hooks(Some(r#"{"a":1}"#)).expect("first"));
        let second = install_claude_hooks(Some(&first)).expect("second");
        assert_eq!(second.after, After::Unchanged);
        assert_eq!(second.notes.len(), 2);
    }

    #[test]
    fn a_hook_already_there_under_another_matcher_is_not_doubled() {
        let shape = r#"{"hooks":{"SessionStart":[{"matcher":"startup","hooks":[{"type":"command","command":"memoryos hook session-start"}]}]}}"#;
        let merged = value(&written(
            install_claude_hooks(Some(shape)).expect("install"),
        ));
        assert_eq!(
            merged["hooks"]["SessionStart"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(merged["hooks"]["Stop"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn malformed_or_misshapen_settings_are_errors() {
        for bad in [
            "",
            "{",
            "{\"a\":1,}",
            "// comment\n{}",
            "[]",
            "null",
            "3",
            r#"{"hooks":[]}"#,
            r#"{"hooks":null}"#,
            r#"{"hooks":{"Stop":{}}}"#,
            r#"{"hooks":{"SessionStart":"x"}}"#,
        ] {
            assert!(install_claude_hooks(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn hooks_round_trip_to_the_original_meaning() {
        for shape in [
            r#"{"a":1,"hooks":{"Stop":[{"hooks":[{"type":"command","command":"foreign"}]}]}}"#,
            r#"{"model":"x"}"#,
            r#"{"hooks":{"SessionStart":[{"matcher":"startup","hooks":[{"type":"command","command":"b"}]}],"PreToolUse":[]}}"#,
        ] {
            let installed = written(install_claude_hooks(Some(shape)).expect("install"));
            let removed = written(uninstall_claude_hooks(Some(&installed)).expect("uninstall"));
            assert_eq!(value(&removed), value(shape), "{shape}");
        }
    }

    #[test]
    fn hooks_uninstall_from_only_ours_removes_the_file() {
        let installed = written(install_claude_hooks(None).expect("install"));
        let outcome = uninstall_claude_hooks(Some(&installed)).expect("uninstall");
        assert_eq!(outcome.after, After::Remove);
    }

    #[test]
    fn uninstall_keeps_a_foreign_handler_in_a_shared_group() {
        let shape = json!({"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": "memoryos hook stop"},
            {"type": "command", "command": "foreign"},
        ]}]}})
        .to_string();
        let removed = value(&written(uninstall_claude_hooks(Some(&shape)).expect("ok")));
        assert_eq!(
            removed,
            json!({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "foreign"}]}]}})
        );
    }

    #[test]
    fn uninstall_leaves_a_changed_hook_and_says_so() {
        let shape = json!({"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": "memoryos hook stop", "timeout": 3},
        ]}]}})
        .to_string();
        let outcome = uninstall_claude_hooks(Some(&shape)).expect("ok");
        assert_eq!(outcome.after, After::Unchanged);
        assert!(outcome.notes[0].contains("differs"), "{:?}", outcome.notes);
    }

    #[test]
    fn uninstall_leaves_a_hook_under_a_matcher_it_did_not_write() {
        let shape = json!({"hooks": {"SessionStart": [{"matcher": "startup", "hooks": [
            {"type": "command", "command": "memoryos hook session-start"},
        ]}]}})
        .to_string();
        let outcome = uninstall_claude_hooks(Some(&shape)).expect("ok");
        assert_eq!(outcome.after, After::Unchanged);
    }

    #[test]
    fn session_start_runs_after_clear() {
        let text = written(install_claude_hooks(None).expect("install"));
        let matcher = value(&text)["hooks"]["SessionStart"][0]["matcher"]
            .as_str()
            .expect("matcher")
            .to_owned();
        let events: Vec<_> = matcher.split('|').collect();
        assert_eq!(events, ["startup", "resume", "clear", "compact"]);
    }

    #[test]
    fn uninstall_removes_the_matcher_an_older_install_wrote() {
        let shape =
            json!({"hooks": {"SessionStart": [{"matcher": "startup|resume|compact", "hooks": [
                {"type": "command", "command": "memoryos hook session-start"},
            ]}]}})
            .to_string();
        let outcome = uninstall_claude_hooks(Some(&shape)).expect("ok");
        assert_eq!(outcome.after, After::Remove);
    }

    #[test]
    fn uninstall_with_nothing_to_do_changes_nothing() {
        for shape in [
            r#"{"a":1}"#,
            r#"{"hooks":{}}"#,
            r#"{"hooks":{"Stop":[]}}"#,
            r#"{"hooks":[]}"#,
        ] {
            let outcome = uninstall_claude_hooks(Some(shape)).expect(shape);
            assert_eq!(outcome.after, After::Unchanged, "{shape}");
        }
        assert_eq!(
            uninstall_claude_hooks(None).expect("none").after,
            After::Unchanged
        );
        assert!(uninstall_claude_hooks(Some("{")).is_err());
    }

    // ---- MCP ----

    #[test]
    fn the_claude_entry_has_the_documented_form() {
        let text = written(install_server(None, "mcpServers", &claude_entry(URL)).expect("ok"));
        assert_eq!(
            value(&text),
            json!({"mcpServers": {"memoryos": {
                "type": "http",
                "url": "http://localhost:8080/mcp",
                "headers": {"Authorization": "Bearer ${MEMORYOS_API_KEY}"},
            }}})
        );
    }

    #[test]
    fn the_opencode_entry_has_the_documented_form() {
        let text = written(install_opencode_server(None, URL).expect("ok"));
        assert_eq!(
            value(&text),
            json!({"mcp": {"memoryos": {
                "type": "remote",
                "url": "http://localhost:8080/mcp",
                "enabled": true,
                "oauth": false,
                "headers": {"Authorization": "Bearer {env:MEMORYOS_API_KEY}"},
            }}})
        );
    }

    #[test]
    fn a_base_path_and_slash_are_handled() {
        assert_eq!(mcp_url("https://h/base/"), "https://h/base/mcp");
    }

    #[test]
    fn servers_merge_beside_foreign_ones_and_are_idempotent() {
        let shape = r#"{"other":true,"mcpServers":{"foreign":{"type":"http","url":"http://f"}}}"#;
        let entry = claude_entry(URL);
        let first = written(install_server(Some(shape), "mcpServers", &entry).expect("ok"));
        let merged = value(&first);
        assert_eq!(merged["other"], true);
        assert_eq!(merged["mcpServers"]["foreign"]["url"], "http://f");
        assert!(merged["mcpServers"]["memoryos"].is_object());
        assert_eq!(
            install_server(Some(&first), "mcpServers", &entry)
                .expect("again")
                .after,
            After::Unchanged
        );
        let removed =
            written(uninstall_server(Some(&first), "mcpServers", is_claude_entry).expect("ok"));
        assert_eq!(value(&removed), value(shape));
    }

    #[test]
    fn a_foreign_server_with_our_name_is_never_replaced_or_removed() {
        let shape = r#"{"mcpServers":{"memoryos":{"type":"stdio","command":"mine"}}}"#;
        let install = install_server(Some(shape), "mcpServers", &claude_entry(URL)).expect("ok");
        assert_eq!(install.after, After::Unchanged);
        assert!(install.notes[0].contains("differs"));
        let uninstall = uninstall_server(Some(shape), "mcpServers", is_claude_entry).expect("ok");
        assert_eq!(uninstall.after, After::Unchanged);
    }

    #[test]
    fn a_different_url_is_reported_and_kept_but_still_ours_to_uninstall() {
        let first =
            written(install_server(None, "mcpServers", &claude_entry("http://a")).expect("ok"));
        let second =
            install_server(Some(&first), "mcpServers", &claude_entry("http://b")).expect("ok");
        assert_eq!(second.after, After::Unchanged);
        assert!(second.notes[0].contains("differs"));
        let removed = uninstall_server(Some(&first), "mcpServers", is_claude_entry).expect("ok");
        assert_eq!(removed.after, After::Remove);
    }

    #[test]
    fn misshapen_mcp_files_are_errors() {
        for bad in ["[]", "{", r#"{"mcpServers":[]}"#, r#"{"mcpServers":"x"}"#] {
            assert!(
                install_server(Some(bad), "mcpServers", &claude_entry(URL)).is_err(),
                "{bad}"
            );
        }
    }

    // ---- OpenCode ----

    #[test]
    fn comments_are_detected_only_outside_strings() {
        for commented in [
            "// c\n{}",
            "{} // c",
            "/* c */ {}",
            "{\"a\": 1 /* c */}",
            "{\n  // \"a\": 1\n  \"b\": 2\n}",
            "{\"a\":\"\\\\\" // c\n}",
        ] {
            assert!(has_comments(commented), "{commented}");
        }
        for plain in [
            "{}",
            r#"{"u":"http://x"}"#,
            r#"{"g":"src/**/*.ts"}"#,
            r#"{"e":"a\"//b"}"#,
            r#"{"e":"\\","f":"//"}"#,
            "{\"a\":1}/",
            "/ /",
        ] {
            assert!(!has_comments(plain), "{plain}");
        }
    }

    #[test]
    fn opencode_json_with_comments_stops_and_shows_what_to_add() {
        let shape = "{\n  // keep me\n  \"model\": \"x\"\n}";
        let error = install_opencode_server(Some(shape), URL).expect_err("stop");
        assert!(error.contains("by hand"), "{error}");
        assert!(error.contains("{env:MEMORYOS_API_KEY}"), "{error}");
        assert!(error.contains("http://localhost:8080/mcp"), "{error}");
    }

    #[test]
    fn opencode_uninstall_with_comments_stops_only_if_there_is_something_to_remove() {
        let with_entry = "{ // c\n \"mcp\": {\"memoryos\": {}} }";
        assert!(uninstall_opencode_server(Some(with_entry)).is_err());
        let without = "{ // c\n \"model\": \"x\" }";
        assert_eq!(
            uninstall_opencode_server(Some(without)).expect("ok").after,
            After::Unchanged
        );
    }

    #[test]
    fn opencode_json_merges_and_round_trips() {
        let shape = r#"{"$schema":"https://opencode.ai/config.json","mcp":{"foreign":{"type":"local","command":["x"]}},"plugin":["p"]}"#;
        let merged = written(install_opencode_server(Some(shape), URL).expect("ok"));
        assert!(value(&merged)["mcp"]["memoryos"].is_object());
        assert_eq!(
            install_opencode_server(Some(&merged), URL)
                .expect("again")
                .after,
            After::Unchanged
        );
        let removed = written(uninstall_opencode_server(Some(&merged)).expect("ok"));
        assert_eq!(value(&removed), value(shape));
    }

    #[test]
    fn the_plugin_is_written_once_and_a_foreign_file_is_kept() {
        assert!(matches!(install_plugin(None).after, After::Write(_)));
        assert_eq!(
            install_plugin(Some(OPENCODE_PLUGIN)).after,
            After::Unchanged
        );
        assert_eq!(install_plugin(Some("mine")).after, After::Unchanged);
        assert_eq!(uninstall_plugin(Some(OPENCODE_PLUGIN)).after, After::Remove);
        assert_eq!(uninstall_plugin(Some("mine")).after, After::Unchanged);
        assert_eq!(uninstall_plugin(None).after, After::Unchanged);
    }

    #[test]
    fn the_plugin_cannot_raise_or_push_empty_text() {
        assert!(OPENCODE_PLUGIN.contains("`memoryos hook session-start`.nothrow().text()"));
        assert!(OPENCODE_PLUGIN.contains("if (memories.trim()) output.context.push(memories);"));
    }

    #[test]
    fn uninstall_removes_the_plugin_an_older_install_wrote() {
        assert_eq!(
            uninstall_plugin(Some(OLD_OPENCODE_PLUGIN)).after,
            After::Remove
        );
    }

    // ---- Plans and files ----

    struct Sandbox {
        _root: tempfile::TempDir,
        project: PathBuf,
        home: PathBuf,
    }

    impl Sandbox {
        fn new() -> Self {
            let root = tempfile::tempdir().expect("dir");
            let project = root.path().join("project");
            let home = root.path().join("home");
            fs::create_dir_all(&project).expect("mkdir");
            fs::create_dir_all(&home).expect("mkdir");
            Self {
                _root: root,
                project,
                home,
            }
        }

        fn run(
            &self,
            op: &Op,
            agent: Agent,
            global: bool,
            dry_run: bool,
        ) -> Result<String, String> {
            let home = self.home.display().to_string();
            let env = move |name: &str| (name == "HOME").then(|| home.clone());
            let plans = plan(op, agent, global, &self.project, &env)?;
            apply(&plans, dry_run)
        }
    }

    fn install() -> Op {
        Op::Install {
            base_url: URL.to_owned(),
        }
    }

    /// Every file below `dir`, relative, with its content.
    fn tree(dir: &Path) -> Vec<(String, String)> {
        fn walk(base: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
            for entry in fs::read_dir(dir).expect("list").flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(base, &path, out);
                } else {
                    let name = path
                        .strip_prefix(base)
                        .expect("prefix")
                        .display()
                        .to_string();
                    out.push((name, fs::read_to_string(&path).expect("read")));
                }
            }
        }
        let mut out = Vec::new();
        walk(dir, dir, &mut out);
        out.sort();
        out
    }

    #[test]
    fn claude_project_install_writes_the_project_files_only() {
        let sandbox = Sandbox::new();
        let report = sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect("install");
        assert!(report.contains("created"), "{report}");
        let names: Vec<_> = tree(&sandbox.project).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, [".claude/settings.json", ".mcp.json"]);
        assert!(tree(&sandbox.home).is_empty());
    }

    #[test]
    fn claude_global_install_uses_the_user_files() {
        let sandbox = Sandbox::new();
        sandbox
            .run(&install(), Agent::ClaudeCode, true, false)
            .expect("install");
        let names: Vec<_> = tree(&sandbox.home).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, [".claude.json", ".claude/settings.json"]);
        assert!(tree(&sandbox.project).is_empty());
    }

    #[test]
    fn opencode_project_and_global_paths() {
        let sandbox = Sandbox::new();
        sandbox
            .run(&install(), Agent::Opencode, false, false)
            .expect("project");
        let names: Vec<_> = tree(&sandbox.project).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, [".opencode/plugins/memoryos.ts", "opencode.json"]);

        sandbox
            .run(&install(), Agent::Opencode, true, false)
            .expect("global");
        let names: Vec<_> = tree(&sandbox.home).into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            [
                ".config/opencode/opencode.json",
                ".config/opencode/plugins/memoryos.ts"
            ]
        );
    }

    #[test]
    fn an_existing_jsonc_file_is_the_one_used() {
        let sandbox = Sandbox::new();
        let jsonc = sandbox.project.join("opencode.jsonc");
        fs::write(&jsonc, r#"{"model":"x"}"#).expect("seed");
        sandbox
            .run(&install(), Agent::Opencode, false, false)
            .expect("install");
        assert!(!sandbox.project.join("opencode.json").exists());
        assert!(fs::read_to_string(&jsonc)
            .expect("read")
            .contains("memoryos"));
    }

    #[test]
    fn dry_run_writes_nothing() {
        for agent in [Agent::ClaudeCode, Agent::Opencode] {
            let sandbox = Sandbox::new();
            let report = sandbox
                .run(&install(), agent, false, true)
                .expect("dry run");
            assert!(report.contains("would create"), "{report}");
            assert!(tree(&sandbox.project).is_empty());
        }
    }

    #[test]
    fn dry_run_uninstall_writes_nothing() {
        let sandbox = Sandbox::new();
        sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect("install");
        let before = tree(&sandbox.project);
        let report = sandbox
            .run(&Op::Uninstall, Agent::ClaudeCode, false, true)
            .expect("dry run");
        assert!(report.contains("would remove"), "{report}");
        assert_eq!(tree(&sandbox.project), before);
    }

    #[test]
    fn a_second_install_touches_nothing_and_makes_no_backup() {
        for agent in [Agent::ClaudeCode, Agent::Opencode] {
            let sandbox = Sandbox::new();
            sandbox.run(&install(), agent, false, false).expect("first");
            let before = tree(&sandbox.project);
            let report = sandbox
                .run(&install(), agent, false, false)
                .expect("second");
            assert_eq!(tree(&sandbox.project), before);
            assert_eq!(
                report
                    .lines()
                    .filter(|l| l.starts_with("unchanged"))
                    .count(),
                2
            );
        }
    }

    #[test]
    fn an_unchanged_file_keeps_its_exact_bytes() {
        let sandbox = Sandbox::new();
        sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect("first");
        let settings = sandbox.project.join(".claude/settings.json");
        // Compact, valid and already complete: there is nothing to change, so it is not reformatted.
        let compact = value(&fs::read_to_string(&settings).expect("read")).to_string();
        fs::write(&settings, &compact).expect("compact");
        sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect("second");
        assert_eq!(fs::read_to_string(&settings).expect("read"), compact);
    }

    #[test]
    fn a_changed_file_gets_a_backup_of_the_original() {
        let sandbox = Sandbox::new();
        let settings = sandbox.project.join(".claude/settings.json");
        fs::create_dir_all(settings.parent().expect("parent")).expect("mkdir");
        fs::write(&settings, r#"{"a":1}"#).expect("seed");
        let report = sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect("install");
        let backup = sandbox.project.join(".claude/settings.json.bak");
        assert_eq!(fs::read_to_string(&backup).expect("backup"), r#"{"a":1}"#);
        assert!(report.contains("settings.json.bak"), "{report}");
    }

    #[test]
    fn malformed_json_in_either_file_writes_nothing_anywhere() {
        let sandbox = Sandbox::new();
        fs::write(sandbox.project.join(".mcp.json"), "{ not json").expect("seed");
        let error = sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect_err("bad");
        assert!(error.contains(".mcp.json"), "{error}");
        assert!(error.contains("Nothing was written"), "{error}");
        assert_eq!(
            tree(&sandbox.project),
            [(".mcp.json".to_owned(), "{ not json".to_owned())],
            "no hook file and no change"
        );
    }

    #[test]
    fn comments_in_opencode_json_write_nothing_not_even_the_plugin() {
        let sandbox = Sandbox::new();
        let shape = "{ // mine\n}";
        fs::write(sandbox.project.join("opencode.json"), shape).expect("seed");
        let error = sandbox
            .run(&install(), Agent::Opencode, false, false)
            .expect_err("stop");
        assert!(error.contains("by hand"), "{error}");
        assert!(error.contains("memoryos.ts"), "{error}");
        assert_eq!(
            tree(&sandbox.project),
            [("opencode.json".to_owned(), shape.to_owned())]
        );
    }

    #[test]
    fn install_then_uninstall_in_an_empty_project_leaves_no_file() {
        for agent in [Agent::ClaudeCode, Agent::Opencode] {
            let sandbox = Sandbox::new();
            sandbox
                .run(&install(), agent, false, false)
                .expect("install");
            sandbox
                .run(&Op::Uninstall, agent, false, false)
                .expect("uninstall");
            let left: Vec<_> = tree(&sandbox.project)
                .into_iter()
                .filter(|(name, _)| !name.contains(".bak"))
                .collect();
            assert!(left.is_empty(), "{left:?}");
        }
    }

    #[test]
    fn uninstall_keeps_other_settings_and_backs_up() {
        let sandbox = Sandbox::new();
        let mcp = sandbox.project.join(".mcp.json");
        fs::write(
            &mcp,
            r#"{"mcpServers":{"f":{"type":"http","url":"http://f"}}}"#,
        )
        .expect("seed");
        sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect("install");
        sandbox
            .run(&Op::Uninstall, Agent::ClaudeCode, false, false)
            .expect("uninstall");
        assert_eq!(
            value(&fs::read_to_string(&mcp).expect("read")),
            json!({"mcpServers": {"f": {"type": "http", "url": "http://f"}}})
        );
        assert!(sandbox.project.join(".mcp.json.bak").exists());
        assert!(
            sandbox.project.join(".mcp.json.bak.1").exists(),
            "both states are kept"
        );
    }

    #[test]
    fn uninstall_on_a_clean_project_does_nothing() {
        for agent in [Agent::ClaudeCode, Agent::Opencode] {
            let sandbox = Sandbox::new();
            let report = sandbox
                .run(&Op::Uninstall, agent, false, false)
                .expect("uninstall");
            assert_eq!(
                report
                    .lines()
                    .filter(|l| l.starts_with("unchanged"))
                    .count(),
                2
            );
            assert!(tree(&sandbox.project).is_empty());
        }
    }

    #[test]
    fn no_home_is_an_error_for_global_only() {
        let sandbox = Sandbox::new();
        let env = |_: &str| None;
        assert!(plan(&install(), Agent::ClaudeCode, true, &sandbox.project, &env).is_err());
        assert!(plan(&install(), Agent::ClaudeCode, false, &sandbox.project, &env).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_stays_a_link_and_its_target_is_updated() {
        let sandbox = Sandbox::new();
        let target = sandbox.home.join("dotfiles-mcp.json");
        fs::write(&target, "{}").expect("seed");
        let link = sandbox.project.join(".mcp.json");
        std::os::unix::fs::symlink(&target, &link).expect("link");

        sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .expect("install");

        assert!(fs::symlink_metadata(&link)
            .expect("stat")
            .file_type()
            .is_symlink());
        assert!(fs::read_to_string(&target)
            .expect("read")
            .contains("memoryos"));
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_link_is_an_error() {
        let sandbox = Sandbox::new();
        let link = sandbox.project.join(".mcp.json");
        std::os::unix::fs::symlink(sandbox.home.join("missing"), &link).expect("link");
        assert!(sandbox
            .run(&install(), Agent::ClaudeCode, false, false)
            .is_err());
    }

    #[test]
    fn hooks_are_written_in_the_documented_key_order() {
        let text = written(install_claude_hooks(None).expect("install"));
        let matcher = text.find("\"matcher\"").expect("matcher");
        let hooks = text
            .find("startup")
            .map(|at| text[at..].find("\"hooks\"").map(|n| n + at));
        assert!(hooks.flatten().is_some_and(|at| matcher < at), "{text}");
        assert!(text.find("\"type\"") < text.find("\"command\""), "{text}");
    }

    #[test]
    fn config_holds_a_reference_to_the_key_never_a_value() {
        assert!(claude_entry(URL).contains("${MEMORYOS_API_KEY}"));
        assert!(opencode_entry(URL).contains("{env:MEMORYOS_API_KEY}"));
    }
}
