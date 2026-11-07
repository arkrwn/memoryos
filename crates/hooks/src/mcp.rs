//! `memoryos mcp`: a stdio MCP server for the agent, forwarding each JSON-RPC line to the server's
//! `/mcp` with the agent key. The agent's configuration therefore holds no key and no reference to
//! one. The admin key is never used here.

use std::io::{BufRead, Write};
use std::time::Duration;

use serde_json::{json, Value};

use crate::config::{self, process_env};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Longer than any tool call should take, so a hung server cannot stall the agent forever.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const INTERNAL_ERROR: i64 = -32603;
const PARSE_ERROR: i64 = -32700;

const META: &str = "_meta";
const META_VERSION: &str = "io.modelcontextprotocol/protocolVersion";

pub fn run() -> Result<(), String> {
    let key = config::agent_key(&process_env)?;
    let endpoint = format!("{}/mcp", config::base_url(&process_env)?);
    let http = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| format!("cannot start the HTTP client: {e}"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the runtime: {e}"))?;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| format!("cannot read stdin: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = runtime.block_on(forward(&http, &endpoint, &key, &line));
        if let Some(reply) = reply {
            // The host closed the pipe, so there is nobody left to serve.
            if writeln!(stdout, "{reply}")
                .and_then(|()| stdout.flush())
                .is_err()
            {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// What to write back for one line, if anything.
async fn forward(http: &reqwest::Client, endpoint: &str, key: &str, line: &str) -> Option<String> {
    let Ok(message) = serde_json::from_str::<Value>(line) else {
        return Some(error_reply(&Value::Null, PARSE_ERROR, "invalid JSON"));
    };
    let id = message.get("id").cloned();
    let mut request = http
        .post(endpoint)
        .bearer_auth(key)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .body(line.to_owned());
    for (name, value) in modern_headers(&message) {
        request = request.header(name, value);
    }
    match request.send().await {
        Ok(response) => {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            reply_for(id.as_ref(), status, &body, endpoint)
        }
        Err(error) => {
            let cause = if error.is_timeout() {
                "did not answer in time"
            } else {
                "cannot be reached"
            };
            failure(
                id.as_ref(),
                &format!("the MemoryOS server at {endpoint} {cause}. Is it running?"),
            )
        }
    }
}

/// Revision 2026-07-28 mirrors the method and name in headers, and the server checks them. Older
/// revisions, which send no version in `_meta`, take none.
fn modern_headers(message: &Value) -> Vec<(&'static str, String)> {
    let Some(version) = message["params"][META][META_VERSION].as_str() else {
        return Vec::new();
    };
    let mut headers = vec![("mcp-protocol-version", version.to_owned())];
    if let Some(method) = message["method"].as_str() {
        headers.push(("mcp-method", method.to_owned()));
    }
    let params = &message["params"];
    if let Some(name) = params["name"].as_str().or(params["uri"].as_str()) {
        headers.push(("mcp-name", name.to_owned()));
    }
    headers
}

/// A JSON-RPC body passes through whatever its HTTP status. Anything else on a request becomes a
/// JSON-RPC error, because the host waits for an answer to every id. A notification gets no answer.
fn reply_for(id: Option<&Value>, status: u16, body: &str, endpoint: &str) -> Option<String> {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if value.get("jsonrpc").is_some() {
            return Some(value.to_string());
        }
    }
    if (200..300).contains(&status) && (id.is_none() || body.trim().is_empty() && status == 202) {
        return None;
    }
    let hint = if status == 401 {
        ". The key was refused: run `memoryos setup`"
    } else {
        ""
    };
    failure(
        id,
        &format!("{endpoint} answered HTTP {status} without a JSON-RPC body{hint}"),
    )
}

fn failure(id: Option<&Value>, message: &str) -> Option<String> {
    match id {
        Some(id) => Some(error_reply(id, INTERNAL_ERROR, message)),
        None => {
            eprintln!("memoryos mcp: {message}");
            None
        }
    }
}

fn error_reply(id: &Value, code: i64, message: &str) -> String {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const AT: &str = "http://h/mcp";

    fn parsed(reply: Option<String>) -> Value {
        serde_json::from_str(&reply.expect("a reply")).expect("JSON")
    }

    #[test]
    fn a_jsonrpc_body_passes_through_on_one_line_at_any_status() {
        let body =
            "{\n\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32602,\"message\":\"x\"}\n}";
        let reply = reply_for(Some(&json!(1)), 400, body, AT).expect("reply");
        assert!(!reply.contains('\n'), "{reply}");
        assert_eq!(parsed(Some(reply))["error"]["code"], -32602);
    }

    #[test]
    fn an_accepted_notification_gets_no_reply() {
        assert_eq!(reply_for(None, 202, "", AT), None);
    }

    #[test]
    fn a_request_always_gets_an_answer() {
        let refused = parsed(reply_for(Some(&json!(7)), 401, "{}", AT));
        assert_eq!(refused["id"], 7);
        assert!(refused["error"]["message"]
            .as_str()
            .expect("message")
            .contains("memoryos setup"));
        let empty = parsed(reply_for(Some(&json!("a")), 200, "", AT));
        assert_eq!(empty["id"], "a");
    }

    #[test]
    fn a_failed_notification_is_not_answered() {
        assert_eq!(reply_for(None, 500, "oops", AT), None);
    }

    #[test]
    fn headers_are_mirrored_only_for_the_modern_revision() {
        let legacy = json!({"method": "tools/list", "params": {}});
        assert!(modern_headers(&legacy).is_empty());
        let modern = json!({"method": "tools/call", "params": {
            "name": "recall",
            "_meta": {META_VERSION: "2026-07-28"},
        }});
        assert_eq!(
            modern_headers(&modern),
            [
                ("mcp-protocol-version", "2026-07-28".to_owned()),
                ("mcp-method", "tools/call".to_owned()),
                ("mcp-name", "recall".to_owned()),
            ]
        );
    }
}
