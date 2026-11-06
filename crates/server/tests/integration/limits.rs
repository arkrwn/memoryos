//! Request limits, error shapes and operational endpoints.

use memoryos_core::Trust;
use reqwest::header::{CONTENT_TYPE, RETRY_AFTER};
use reqwest::StatusCode;
use serde_json::{json, Value};

use crate::common::{assert_error, drop_database, random_uuid, Server};

const ONE_MIB: usize = 1024 * 1024;

#[tokio::test(flavor = "multi_thread")]
async fn a_key_over_its_rate_gets_429_with_retry_after_and_others_do_not() {
    let server = Server::host_with(&[("MEMORYOS_RATE_LIMIT_RPS", "3")]).await;
    let busy = server.root("rate").await;
    let quiet = server.root("rate").await;

    let mut limited = None;
    for _ in 0..50 {
        let (status, headers, body) = server
            .call_full("GET", "/v1/agents/a/blocks", Some(&busy), None)
            .await;
        if status == StatusCode::TOO_MANY_REQUESTS {
            limited = Some((status, headers, body));
            break;
        }
    }
    let (status, headers, body) = limited.expect("the limit should apply within 50 requests");
    assert_error(status, &body, 429, "rate_limited");
    let wait: u64 = headers[RETRY_AFTER]
        .to_str()
        .expect("header")
        .parse()
        .expect("whole seconds");
    assert!(wait >= 1);

    let (status, _) = server
        .call("GET", "/v1/agents/a/blocks", Some(&quiet), None)
        .await;
    assert_eq!(status, StatusCode::OK, "keys must not share a bucket");

    // Operational endpoints are not limited, and neither do they count against a key.
    for _ in 0..10 {
        let (status, _) = server.call("GET", "/healthz", None, None).await;
        assert_eq!(status, StatusCode::OK);
    }

    // The bucket refills.
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let (status, _) = server
        .call("GET", "/v1/agents/a/blocks", Some(&busy), None)
        .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn bodies_over_1_mib_are_refused_and_at_1_mib_are_read() {
    let server = Server::host().await;
    let key = server.root("size").await;

    let oversized = "x".repeat(ONE_MIB + 1);
    for path in ["/v1/recall", "/mcp"] {
        let response = server
            .raw_client()
            .post(format!("{}{path}", server.url))
            .bearer_auth(&key.token)
            .body(oversized.clone())
            .send()
            .await
            .expect("send");
        let status = response.status();
        let body: Value = response.json().await.expect("json");
        assert_error(status, &body, 413, "payload_too_large");
    }

    // Exactly at the limit the body is read, and then refused only for what it holds.
    let padding = ONE_MIB
        - json!({"scopes": ["agent:a"], "query": ""})
            .to_string()
            .len();
    let at_limit = json!({"scopes": ["agent:a"], "query": " ".repeat(padding)}).to_string();
    assert_eq!(at_limit.len(), ONE_MIB);
    let response = server
        .raw_client()
        .post(format!("{}/v1/recall", server.url))
        .bearer_auth(&key.token)
        .body(at_limit)
        .send()
        .await
        .expect("send");
    let status = response.status();
    let body: Value = response.json().await.expect("json");
    assert_error(status, &body, 400, "invalid_input");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("query"));
}

#[tokio::test(flavor = "multi_thread")]
async fn content_and_scope_counts_are_limited() {
    let server = Server::worker_without_loop().await;
    let key = server.root("content").await;
    let episode = |content: String| json!({"scope": "agent:a", "source": "s", "trust": "agent", "content": content});
    let (status, reply) = server
        .call(
            "POST",
            "/v1/episodes",
            Some(&key),
            Some(episode("x".repeat(100_001))),
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    // Characters, not bytes: 100000 multi-byte characters are within the limit.
    let (status, reply) = server
        .call(
            "POST",
            "/v1/episodes",
            Some(&key),
            Some(episode("\u{e9}".repeat(100_000))),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{reply}");

    let scopes =
        |count: usize| -> Vec<String> { (0..count).map(|i| format!("user:u{i}")).collect() };
    let (status, reply) = server
        .call(
            "POST",
            "/v1/recall",
            Some(&key),
            Some(json!({"scopes": scopes(51), "query": "q"})),
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    let (status, reply) = server
        .call(
            "POST",
            "/v1/recall",
            Some(&key),
            Some(json!({"scopes": scopes(50), "query": "q"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let (status, reply) = server
        .call(
            "POST",
            "/v1/recall",
            Some(&key),
            Some(json!({"scopes": [], "query": "q"})),
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    let many = scopes(51).join(",");
    let (status, reply) = server
        .call(
            "GET",
            &format!("/v1/quarantine?scopes={many}"),
            Some(&key),
            None,
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    for limit in [0, 201] {
        let (status, reply) = server
            .call(
                "GET",
                &format!("/v1/quarantine?scopes=user:a&limit={limit}"),
                Some(&key),
                None,
            )
            .await;
        assert_error(status, &reply, 400, "invalid_input");
    }
    for limit in [0, 101] {
        let (status, reply) = server
            .call(
                "POST",
                "/v1/recall",
                Some(&key),
                Some(json!({"scopes": ["user:a"], "query": "q", "limit": limit})),
            )
            .await;
        assert_error(status, &reply, 400, "invalid_input");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_input_is_a_400_in_the_error_shape() {
    let server = Server::host().await;
    let key = server.root("bad").await;

    for path in ["/v1/recall", "/v1/candidates", "/v1/commits"] {
        let response = server
            .raw_client()
            .post(format!("{}{path}", server.url))
            .bearer_auth(&key.token)
            .header(CONTENT_TYPE, "application/json")
            .body("{not json")
            .send()
            .await
            .expect("send");
        let status = response.status();
        let body: Value = response.json().await.expect("json");
        assert_error(status, &body, 400, "invalid_input");
    }

    for path in [
        "/v1/episodes/not-a-uuid",
        "/v1/facts/not-a-uuid",
        "/v1/facts/123/history",
        "/v1/entities/not-a-uuid",
        "/v1/entities/not-a-uuid/neighbors?scopes=user:a",
        "/v1/sessions/not-a-uuid/messages",
    ] {
        let (status, body) = server.call("GET", path, Some(&key), None).await;
        assert_error(status, &body, 400, "invalid_input");
    }
    for path in [
        "/v1/episodes/not-a-uuid/purge",
        "/v1/quarantine/not-a-uuid/approve",
        "/v1/quarantine/not-a-uuid/reject",
    ] {
        let (status, body) = server.call("POST", path, Some(&key), None).await;
        assert_error(status, &body, 400, "invalid_input");
    }

    // A bad scope, a bad field type and a missing query parameter.
    let (status, body) = server
        .call(
            "POST",
            "/v1/recall",
            Some(&key),
            Some(json!({"scopes": ["team:a"], "query": "q"})),
        )
        .await;
    assert_error(status, &body, 400, "invalid_input");
    let (status, body) = server
        .call(
            "POST",
            "/v1/recall",
            Some(&key),
            Some(json!({"scopes": ["user:a"], "query": 5})),
        )
        .await;
    assert_error(status, &body, 400, "invalid_input");
    let (status, body) = server.call("GET", "/v1/quarantine", Some(&key), None).await;
    assert_error(status, &body, 400, "invalid_input");
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_routes_and_methods_use_the_error_shape() {
    let server = Server::host().await;
    let key = server.root("routes").await;
    let (status, body) = server.call("GET", "/v1/nope", Some(&key), None).await;
    assert_error(status, &body, 404, "not_found");
    let (status, body) = server.call("DELETE", "/v1/recall", Some(&key), None).await;
    assert_error(status, &body, 405, "method_not_allowed");
}

#[tokio::test(flavor = "multi_thread")]
async fn health_ready_and_metrics_are_public_and_carry_no_tenant_labels() {
    let server = Server::host().await;
    let (status, body) = server.call("GET", "/healthz", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    let (status, body) = server.call("GET", "/readyz", None, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["database"], "ok");

    let tenant = "tenant_for_metrics";
    let key = server.key(tenant, &["*"], Trust::Agent, false).await;
    let id = random_uuid();
    server
        .call("GET", &format!("/v1/facts/{id}"), Some(&key), None)
        .await;
    let recall = json!({"scopes": ["agent:a"], "query": "q"});
    server
        .call("POST", "/v1/recall", Some(&key), Some(recall))
        .await;

    let response = server
        .raw_client()
        .get(format!("{}/metrics", server.url))
        .send()
        .await
        .expect("metrics");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers()[CONTENT_TYPE]
        .to_str()
        .expect("type")
        .starts_with("text/plain"));
    let text = response.text().await.expect("text");
    assert!(
        text.contains(
            r#"memoryos_http_requests_total{method="GET",route="/v1/facts/{id}",status="404"} 1"#
        ),
        "{text}"
    );
    assert!(text.contains("memoryos_http_request_duration_seconds_bucket"));
    assert!(
        text.contains("memoryos_recall_duration_seconds_count 1"),
        "{text}"
    );
    assert!(
        !text.contains(&id.to_string()),
        "a raw path reached a label"
    );
    assert!(!text.contains(tenant), "a tenant reached a label");
    assert!(!text.contains(&key.id.to_string()), "a key reached a label");
    // The SDK covers the same three endpoints.
    let mem = server.sdk(&key);
    mem.health().await.expect("health");
    mem.ready().await.expect("ready");
    assert!(mem
        .metrics()
        .await
        .expect("metrics")
        .contains("memoryos_http_requests_total"));

    // A probe of an unknown path adds no series for that path.
    server.call("GET", "/v1/secret-probe", None, None).await;
    let text = server
        .raw_client()
        .get(format!("{}/metrics", server.url))
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("text");
    assert!(!text.contains("secret-probe"), "{text}");
    assert!(text.contains(r#"route="unmatched""#), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn responses_carry_a_request_id_and_keep_a_clean_one() {
    let server = Server::host().await;
    let (_, headers, _) = server.call_full("GET", "/healthz", None, None).await;
    assert!(!headers["x-request-id"].is_empty());

    let response = server
        .raw_client()
        .get(format!("{}/healthz", server.url))
        .header("x-request-id", "abc-123")
        .send()
        .await
        .expect("send");
    assert_eq!(response.headers()["x-request-id"], "abc-123");
    let response = server
        .raw_client()
        .get(format!("{}/healthz", server.url))
        .header("x-request-id", "bad id with spaces")
        .send()
        .await
        .expect("send");
    assert_ne!(response.headers()["x-request-id"], "bad id with spaces");
}

#[tokio::test(flavor = "multi_thread")]
async fn readyz_fails_and_errors_stay_generic_when_the_database_is_gone() {
    let server = Server::host().await;
    let key = server.root("down").await;
    drop_database(&server.database_url).await;

    let (status, body) = server.call("GET", "/readyz", None, None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["status"], "not_ready");
    assert_eq!(body["database"], "unavailable");
    let (status, _) = server.call("GET", "/healthz", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "liveness does not need the database"
    );

    let (status, body) = server
        .call("GET", "/v1/agents/a/blocks", Some(&key), None)
        .await;
    assert_error(status, &body, 500, "internal");
    assert_eq!(body["error"]["message"], "internal error");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_over_its_time_limit_gets_a_timeout_error() {
    let server = Server::host_with(&[("MEMORYOS_REQUEST_TIMEOUT_SECS", "1")]).await;
    let key = server.root("slow").await;
    server.mock.delay_embeddings(5_000).await;
    let (status, body) = server
        .call(
            "POST",
            "/v1/commits",
            Some(&key),
            Some(json!({
                "scope": "agent:a", "source": "s", "content": "c",
                "facts": [crate::common::statement("Ada waits", "Ada")],
            })),
        )
        .await;
    assert_error(status, &body, 504, "timeout");
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_falls_back_to_text_search_when_embeddings_are_slow() {
    let server = Server::host().await;
    let key = server.root("fallback").await;
    let (_, _, _) = server.commit_one(&key, "agent:a", "Ada sails boats").await;
    server.mock.delay_embeddings(10_000).await;
    let started = std::time::Instant::now();
    let found = server.recall_statements(&key, "agent:a", "sails").await;
    assert_eq!(found, ["Ada sails boats"]);
    assert!(started.elapsed().as_secs() < 8, "{:?}", started.elapsed());
}

#[tokio::test(flavor = "multi_thread")]
async fn shutdown_completes_an_in_flight_request() {
    let server = Server::host().await;
    let key = server.root("drain").await;
    server.mock.delay_embeddings(1_500).await;
    let url = format!("{}/v1/commits", server.url);
    let client = server.raw_client().clone();
    let token = key.token.clone();
    let in_flight = tokio::spawn(async move {
        client
            .post(url)
            .bearer_auth(token)
            .json(&json!({
                "scope": "agent:a", "source": "s", "content": "c",
                "facts": [crate::common::statement("Ada finishes", "Ada")],
            }))
            .send()
            .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let address = server.url.clone();
    let stopped = server.stop();
    let result = stopped.await;
    assert_eq!(result, Ok(()));

    let response = in_flight.await.expect("task").expect("request completes");
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json");
    assert_eq!(body["facts"][0]["status"], "active");

    // New connections are refused from here on.
    assert!(reqwest::get(format!("{address}/healthz")).await.is_err());
}
