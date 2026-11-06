//! `docs/openapi.yaml` and the served routes must list exactly the same operations.

use std::collections::BTreeSet;

use memoryos_server::routes::served_routes;
use reqwest::StatusCode;
use serde_json::json;

use crate::common::{random_uuid, Server};

const METHODS: [&str; 7] = ["get", "post", "put", "delete", "patch", "head", "options"];

struct Operation {
    method: String,
    path: String,
    /// `security: []` on the operation.
    public: bool,
}

fn operations() -> Vec<Operation> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/openapi.yaml");
    let text = std::fs::read_to_string(path).expect("read docs/openapi.yaml");
    let spec: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).expect("parse openapi.yaml");
    let paths = spec["paths"].as_mapping().expect("paths");
    let mut found = Vec::new();
    for (path, item) in paths {
        let path = path.as_str().expect("path");
        let item = item.as_mapping().expect("path item");
        for (method, operation) in item {
            let method = method.as_str().expect("method");
            if !METHODS.contains(&method) {
                continue;
            }
            let public = operation
                .get("security")
                .and_then(serde_yaml_ng::Value::as_sequence)
                .is_some_and(Vec::is_empty);
            found.push(Operation {
                method: method.to_uppercase(),
                path: path.to_owned(),
                public,
            });
        }
    }
    assert!(found.len() > 20, "the file should list every operation");
    found
}

#[test]
fn served_routes_match_the_openapi_file_exactly() {
    let documented: BTreeSet<(String, String, bool)> = operations()
        .into_iter()
        .map(|op| (op.method, op.path, op.public))
        .collect();
    let served: BTreeSet<(String, String, bool)> = served_routes()
        .into_iter()
        .map(|(method, path, public)| (method.to_owned(), path.to_owned(), public))
        .collect();
    assert_eq!(
        documented.difference(&served).collect::<Vec<_>>(),
        Vec::<&(String, String, bool)>::new(),
        "documented but not served"
    );
    assert_eq!(
        served.difference(&documented).collect::<Vec<_>>(),
        Vec::<&(String, String, bool)>::new(),
        "served but not documented"
    );
}

/// `served_routes` is only a list. This proves the router really answers each one.
#[tokio::test(flavor = "multi_thread")]
async fn every_documented_operation_reaches_a_handler() {
    let server = Server::host().await;
    let key = server.root("contract").await;
    for operation in operations() {
        let path = operation
            .path
            .replace("{id}", &random_uuid().to_string())
            .replace("{agent_id}", "a1")
            .replace("{user_id}", "u1")
            .replace("{label}", "l1");
        let body = (operation.method != "GET").then(|| json!({}));
        let (status, reply) = server
            .call(&operation.method, &path, Some(&key), body)
            .await;
        let what = format!("{} {}", operation.method, operation.path);
        assert_ne!(status, StatusCode::METHOD_NOT_ALLOWED, "{what}: {reply}");
        let fell_through =
            status == StatusCode::NOT_FOUND && reply["error"]["message"] == "route not found";
        assert!(!fell_through, "{what}: {reply}");
    }
}
