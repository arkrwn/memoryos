//! `POST /mcp` on the wire, in both protocol eras.

use memoryos_core::Trust;
use reqwest::StatusCode;
use serde_json::{json, Map, Value};

use crate::common::{assert_error, random_uuid, statement, uuid_at, Key, Server};

const MODERN: &str = "2026-07-28";

fn rpc(id: i64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

async fn post(server: &Server, key: &Key, body: Value) -> (StatusCode, Value) {
    server.call("POST", "/mcp", Some(key), Some(body)).await
}

/// Calls a tool the way a client on revision 2025-11-25 does, and returns the parsed result.
async fn call_tool(server: &Server, key: &Key, name: &str, arguments: Value) -> (bool, Value) {
    let (status, reply) = post(
        server,
        key,
        rpc(
            7,
            "tools/call",
            json!({"name": name, "arguments": arguments}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["id"], 7, "{reply}");
    let result = &reply["result"];
    assert_eq!(result["content"][0]["type"], "text", "{reply}");
    let text = result["content"][0]["text"].as_str().expect("text");
    let parsed = serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_owned()));
    (result["isError"].as_bool().expect("isError"), parsed)
}

fn tool_names(reply: &Value) -> Vec<String> {
    reply["result"]["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|tool| tool["name"].as_str().expect("name").to_owned())
        .collect()
}

/// The headers and `_meta` a client on revision 2026-07-28 must send.
fn modern_request(
    server: &Server,
    key: &Key,
    method: &str,
    mut params: Map<String, Value>,
) -> reqwest::RequestBuilder {
    params.insert(
        "_meta".to_owned(),
        json!({
            "io.modelcontextprotocol/protocolVersion": MODERN,
            "io.modelcontextprotocol/clientCapabilities": {},
        }),
    );
    let mut request = server
        .raw_client()
        .post(format!("{}/mcp", server.url))
        .bearer_auth(&key.token)
        .header("accept", "application/json, text/event-stream")
        .header("mcp-protocol-version", MODERN)
        .header("mcp-method", method);
    if let Some(name) = params.get("name").and_then(Value::as_str) {
        request = request.header("mcp-name", name);
    }
    request.json(&json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}))
}

async fn read(request: reqwest::RequestBuilder) -> (StatusCode, Value) {
    let response = request.send().await.expect("send");
    let status = response.status();
    let bytes = response.bytes().await.expect("body");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn a_legacy_client_can_initialize_and_ping() {
    let server = Server::start().await;
    let key = server.root("m1").await;
    let (status, reply) = post(
        &server,
        &key,
        rpc(
            1,
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["jsonrpc"], "2.0");
    assert_eq!(reply["id"], 1);
    assert_eq!(reply["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(reply["result"]["serverInfo"]["name"], "memoryos");
    assert!(reply["result"]["capabilities"]["tools"].is_object());
    assert!(reply["result"]["instructions"].is_string());
    assert!(reply["result"].get("resultType").is_none());

    let response = server
        .raw_client()
        .post(format!("{}/mcp", server.url))
        .bearer_auth(&key.token)
        .json(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        .send()
        .await
        .expect("notification");
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert!(response.bytes().await.expect("body").is_empty());

    let (status, reply) = post(&server, &key, rpc(2, "ping", json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply, json!({"jsonrpc": "2.0", "id": 2, "result": {}}));

    // A string id is echoed as a string.
    let (_, reply) = post(
        &server,
        &key,
        json!({"jsonrpc": "2.0", "id": "abc", "method": "ping"}),
    )
    .await;
    assert_eq!(reply["id"], "abc");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_modern_client_uses_discover_and_per_request_metadata() {
    let server = Server::start().await;
    let key = server.root("m2").await;

    let (status, reply) = read(modern_request(&server, &key, "server/discover", Map::new())).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let result = &reply["result"];
    assert_eq!(result["resultType"], "complete");
    assert!(result["supportedVersions"]
        .as_array()
        .expect("versions")
        .contains(&json!(MODERN)));
    assert!(result["capabilities"]["tools"].is_object());
    assert_eq!(
        result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
        "memoryos"
    );
    assert!(result["ttlMs"].is_number());
    assert!(result["cacheScope"].is_string());

    let (status, reply) = read(modern_request(&server, &key, "tools/list", Map::new())).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["result"]["resultType"], "complete");
    assert!(reply["result"]["ttlMs"].is_number());
    assert!(tool_names(&reply).contains(&"recall".to_owned()));

    let mut params = Map::new();
    params.insert("name".to_owned(), json!("block_read"));
    params.insert(
        "arguments".to_owned(),
        json!({"agent_id": "a1", "label": "absent"}),
    );
    let (status, reply) = read(modern_request(&server, &key, "tools/call", params)).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["result"]["resultType"], "complete");
    assert_eq!(reply["result"]["isError"], true);

    // An unknown method on this revision is a 404 carrying the JSON-RPC error.
    let (status, reply) = read(modern_request(&server, &key, "no/such", Map::new())).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(reply["error"]["code"], -32601);

    // A header that disagrees with the body is refused before the method runs.
    let mut params = Map::new();
    params.insert("name".to_owned(), json!("recall"));
    let (status, reply) = read(
        modern_request(&server, &key, "tools/call", params).header("mcp-name", "commit_facts"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(reply["error"]["code"], -32020);

    // An unsupported version names the ones that work.
    let (status, reply) = read(
        server
            .raw_client()
            .post(format!("{}/mcp", server.url))
            .bearer_auth(&key.token)
            .header("mcp-protocol-version", "1900-01-01")
            .json(&rpc(1, "tools/list", json!({}))),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(reply["error"]["code"], -32022);
    assert_eq!(reply["error"]["data"]["requested"], "1900-01-01");
}

#[tokio::test(flavor = "multi_thread")]
async fn tools_list_differs_by_mode() {
    let host = Server::start().await;
    let key = host.root("m3").await;
    let (_, reply) = post(&host, &key, rpc(1, "tools/list", json!({}))).await;
    let mut names = tool_names(&reply);
    names.sort();
    assert_eq!(
        names,
        [
            "block_read",
            "block_replace",
            "commit_facts",
            "graph_neighbors",
            "recall",
            "recall_candidates"
        ]
    );
    for tool in reply["result"]["tools"].as_array().expect("tools") {
        assert!(tool["description"].as_str().expect("description").len() > 100);
        assert_eq!(tool["inputSchema"]["type"], "object");
    }
    // `remember` does not exist here, so it is a protocol error and not a tool error.
    let (status, reply) = post(
        &host,
        &key,
        rpc(
            2,
            "tools/call",
            json!({"name": "remember", "arguments": {"scope": "user:u", "content": "c"}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["error"]["code"], -32602, "{reply}");
}

#[tokio::test(flavor = "multi_thread")]
async fn each_host_tool_works_end_to_end() {
    let server = Server::start().await;
    let key = server.root("m4").await;

    // commit_facts then recall_candidates, recall, and graph_neighbors over the same data.
    let (is_error, committed) = call_tool(
        &server,
        &key,
        "commit_facts",
        json!({
            "scope": "agent:a1",
            "source": "chat",
            "content": "Ada works at Acme",
            "facts": [{
                "statement": "Ada works at Acme",
                "subject": {"name": "Ada", "kind": "person"},
                "predicate": "works_at",
                "object": {"name": "Acme", "kind": "company"},
            }],
        }),
    )
    .await;
    assert!(!is_error, "{committed}");
    assert_eq!(committed["facts"][0]["status"], "active");
    let fact_id = uuid_at(&committed, "/facts/0/id");

    let (is_error, candidates) = call_tool(
        &server,
        &key,
        "recall_candidates",
        json!({"scopes": ["agent:a1"], "statement": "Ada works at Globex"}),
    )
    .await;
    assert!(!is_error, "{candidates}");
    assert_eq!(uuid_at(&candidates, "/facts/0/fact/id"), fact_id);

    let (is_error, recalled) = call_tool(
        &server,
        &key,
        "recall",
        json!({"scopes": ["agent:a1"], "query": "Ada works"}),
    )
    .await;
    assert!(!is_error, "{recalled}");
    assert_eq!(
        recalled["facts"][0]["fact"]["statement"],
        "Ada works at Acme"
    );
    let subject = uuid_at(&recalled, "/facts/0/fact/subject_id");

    let (is_error, graph) = call_tool(
        &server,
        &key,
        "graph_neighbors",
        json!({"entity_id": subject, "scopes": ["agent:a1"], "depth": 1}),
    )
    .await;
    assert!(!is_error, "{graph}");
    assert_eq!(graph["edges"].as_array().expect("edges").len(), 1);
    assert_eq!(graph["entities"].as_array().expect("entities").len(), 2);

    // block_replace creates, block_read reads, and a stale version is a readable tool error.
    let (is_error, block) = call_tool(
        &server,
        &key,
        "block_replace",
        json!({"agent_id": "a1", "label": "notes", "content": "first"}),
    )
    .await;
    assert!(!is_error, "{block}");
    assert_eq!(block["version"], 1);
    let (is_error, block) = call_tool(
        &server,
        &key,
        "block_read",
        json!({"agent_id": "a1", "label": "notes"}),
    )
    .await;
    assert!(!is_error, "{block}");
    assert_eq!(block["content"], "first");
    let (is_error, conflict) = call_tool(
        &server,
        &key,
        "block_replace",
        json!({"agent_id": "a1", "label": "notes", "content": "x", "expected_version": 9}),
    )
    .await;
    assert!(is_error);
    assert_eq!(conflict["error"]["code"], "version_conflict");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_same_authorization_applies_as_on_rest() {
    let server = Server::start().await;
    let root = server.root("m6").await;
    let narrow = server.key("m6", &["user:a"], Trust::Agent, false).await;
    let other_tenant = server.root("m6_other").await;

    // No token and a bad token are HTTP errors, before any JSON-RPC.
    let (status, reply) = server
        .call("POST", "/mcp", None, Some(rpc(1, "ping", json!({}))))
        .await;
    assert_error(status, &reply, 401, "unauthorized");
    let bad = Key {
        id: random_uuid(),
        token: "mos_bad".to_owned(),
    };
    let (status, reply) = post(&server, &bad, rpc(1, "ping", json!({}))).await;
    assert_error(status, &reply, 401, "unauthorized");

    let (fact, _, _) = server
        .commit_one(&root, "agent:a1", "Ada sails boats")
        .await;
    let subject = uuid_at(&server.fact_json(&root, fact).await, "/subject_id");

    // Wrong scope is a tool error the model can read, with the REST code.
    let forbidden = [
        ("recall", json!({"scopes": ["agent:a1"], "query": "sails"})),
        (
            "recall_candidates",
            json!({"scopes": ["agent:a1"], "statement": "sails"}),
        ),
        (
            "commit_facts",
            json!({"scope": "agent:a1", "source": "s", "content": "c",
                   "facts": [statement("x", "Ada")]}),
        ),
        ("block_read", json!({"agent_id": "a1", "label": "l"})),
        (
            "block_replace",
            json!({"agent_id": "a1", "label": "l", "content": "c"}),
        ),
        (
            "graph_neighbors",
            json!({"entity_id": subject, "scopes": ["agent:a1"]}),
        ),
    ];
    for (tool, arguments) in forbidden {
        let (is_error, reply) = call_tool(&server, &narrow, tool, arguments).await;
        assert!(is_error, "{tool}");
        assert_eq!(reply["error"]["code"], "forbidden", "{tool}");
    }
    // A row in a scope outside the key looks absent.
    let (is_error, reply) = call_tool(
        &server,
        &narrow,
        "graph_neighbors",
        json!({"entity_id": subject, "scopes": ["user:a"]}),
    )
    .await;
    assert!(is_error);
    assert_eq!(reply["error"]["code"], "not_found");

    // Another tenant finds nothing, even with `*`.
    let (is_error, recalled) = call_tool(
        &server,
        &other_tenant,
        "recall",
        json!({"scopes": ["agent:a1"], "query": "sails"}),
    )
    .await;
    assert!(!is_error);
    assert_eq!(recalled["facts"], json!([]));

    // Agent trust in a user scope is quarantined through MCP as well.
    let (is_error, committed) = call_tool(
        &server,
        &root,
        "commit_facts",
        json!({"scope": "user:u", "source": "s", "content": "c",
               "facts": [statement("Ada is a spy", "Ada")]}),
    )
    .await;
    assert!(!is_error);
    assert_eq!(committed["facts"][0]["status"], "quarantined");
    let hidden = server.recall_statements(&root, "user:u", "Ada spy").await;
    assert!(hidden.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_take_the_documented_shapes() {
    let server = Server::start().await;
    let key = server.root("m7").await;

    // A malformed argument is a tool error, so the model can correct it.
    let (is_error, reply) = call_tool(&server, &key, "recall", json!({"query": 5})).await;
    assert!(is_error);
    assert_eq!(reply["error"]["code"], "invalid_input");
    let (is_error, reply) = call_tool(
        &server,
        &key,
        "graph_neighbors",
        json!({"entity_id": "not-a-uuid", "scopes": ["agent:a"]}),
    )
    .await;
    assert!(is_error);
    assert_eq!(reply["error"]["code"], "invalid_input");

    // Protocol errors are JSON-RPC errors.
    let (status, reply) = post(&server, &key, rpc(1, "no/such", json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["error"]["code"], -32601);
    assert_eq!(reply["id"], 1);
    let (status, reply) = post(
        &server,
        &key,
        rpc(2, "tools/call", json!({"name": "no_such_tool"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["error"]["code"], -32602);
    let (status, reply) = post(&server, &key, rpc(3, "tools/call", json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["error"]["code"], -32602);

    // Not JSON-RPC at all.
    let (status, reply) = post(&server, &key, json!([rpc(1, "ping", json!({}))])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(reply["error"]["code"], -32600);
    let (status, reply) = post(
        &server,
        &key,
        json!({"jsonrpc": "1.0", "id": 1, "method": "ping"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(reply["error"]["code"], -32600);
    let (status, reply) = post(
        &server,
        &key,
        json!({"jsonrpc": "2.0", "id": null, "method": "ping"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(reply["error"]["code"], -32600);
    let response = server
        .raw_client()
        .post(format!("{}/mcp", server.url))
        .bearer_auth(&key.token)
        .body("{not json")
        .send()
        .await
        .expect("send");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let reply: Value = response.json().await.expect("json");
    assert_eq!(reply["error"]["code"], -32700);
    assert!(reply.get("id").is_none());

    // No stream is offered.
    for method in ["GET", "DELETE"] {
        let (status, reply) = server.call(method, "/mcp", Some(&key), None).await;
        assert_error(status, &reply, 405, "method_not_allowed");
    }

    // A page on another site cannot use a local server through DNS rebinding.
    let (status, reply) = read(
        server
            .raw_client()
            .post(format!("{}/mcp", server.url))
            .bearer_auth(&key.token)
            .header("origin", "http://evil.example")
            .json(&rpc(1, "ping", json!({}))),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(reply["error"]["code"].is_number());
    let (status, _) = read(
        server
            .raw_client()
            .post(format!("{}/mcp", server.url))
            .bearer_auth(&key.token)
            .header("origin", "http://localhost:3000")
            .json(&rpc(1, "ping", json!({}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}
