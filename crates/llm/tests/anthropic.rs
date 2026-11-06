#![cfg(feature = "integration")]

mod common;

use std::time::{Duration, Instant};

use common::*;
use memoryos_core::{ContentBlock, Llm};
use memoryos_llm::AnthropicClient;
use serde_json::{json, Value};

const URL: &str = "/v1/messages";
const MODEL: &str = "claude-opus-5";

fn client(wiremock: &WireMock, model: &str) -> AnthropicClient {
    AnthropicClient::new(
        API_KEY.into(),
        model.into(),
        Some(wiremock.base_url.clone()),
    )
}

fn text_reply() -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 7},
    })
}

fn tool_reply() -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_1", "name": "extract", "input": {"facts": ["a"]}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    })
}

fn api_error(message: &str) -> Value {
    json!({"type": "error", "error": {"type": "some_error", "message": message}})
}

#[tokio::test]
async fn text_reply_maps_response_and_sends_required_headers() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub(json!({
            "request": {
                "method": "POST",
                "url": URL,
                "headers": {
                    "x-api-key": {"equalTo": API_KEY},
                    "anthropic-version": {"equalTo": "2026-10-21"},
                },
                "bodyPatterns": [{
                    "equalToJson": {
                        "model": MODEL,
                        "max_tokens": 100,
                        "system": "be brief",
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                        ],
                    },
                }],
            },
            "response": {"status": 200, "jsonBody": text_reply()},
        }))
        .await;

    let response = client(&wiremock, MODEL)
        .complete(text_request("hi"))
        .await
        .expect("completion");

    assert_eq!(
        response.content,
        vec![ContentBlock::Text {
            text: "hello".into()
        }]
    );
    assert_eq!((response.input_tokens, response.output_tokens), (11, 7));
}

#[tokio::test]
async fn forced_tool_call_sends_tool_choice_and_returns_tool_use() {
    let wiremock = WireMock::start().await;
    wiremock.stub_post(URL, 200, tool_reply()).await;

    let response = client(&wiremock, MODEL)
        .complete(forced_request())
        .await
        .expect("completion");

    assert_eq!(
        response.content,
        vec![ContentBlock::ToolUse {
            id: "toolu_1".into(),
            name: "extract".into(),
            input: json!({"facts": ["a"]}),
        }]
    );
    let expected = json!({
        "tool_choice": {"type": "tool", "name": "extract"},
        "tools": [{
            "name": "extract",
            "description": "Extract facts",
            "input_schema": {"type": "object", "properties": {"facts": {"type": "array"}}},
        }],
    });
    assert_eq!(wiremock.count_posts_with_body(URL, expected).await, 1);
}

#[tokio::test]
async fn forced_tool_on_default_model_uses_auto_with_one_tool() {
    let wiremock = WireMock::start().await;
    wiremock.stub_post(URL, 200, tool_reply()).await;

    let mut request = forced_request();
    request.tools.push(memoryos_core::ToolDef {
        name: "other".into(),
        description: "must not be offered".into(),
        input_schema: json!({"type": "object"}),
    });
    client(&wiremock, "claude-sonnet-5-5")
        .complete(request)
        .await
        .expect("completion");

    let expected = json!({
        "system": "You must respond by calling the extract tool.",
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
        "tools": [{
            "name": "extract",
            "description": "Extract facts",
            "input_schema": {"type": "object", "properties": {"facts": {"type": "array"}}},
        }],
    });
    assert_eq!(wiremock.count_posts_with_body(URL, expected).await, 1);
    // ignoreExtraElements also tolerates extra array items, so check the array length directly.
    let second_tool = json!({
        "method": "POST",
        "url": URL,
        "bodyPatterns": [{"matchesJsonPath": "$.tools[1]"}],
    });
    assert_eq!(wiremock.count(second_tool).await, 0);
}

#[tokio::test]
async fn forced_tool_on_default_model_fails_when_no_call_is_made() {
    let wiremock = WireMock::start().await;
    wiremock.stub_post(URL, 200, text_reply()).await;

    let message = provider_error(
        client(&wiremock, "claude-sonnet-5-5")
            .complete(forced_request())
            .await,
    );

    assert!(message.contains("extract"), "{message}");
    assert_eq!(wiremock.count_posts(URL).await, 1);
}

#[tokio::test]
async fn tool_result_round_trips_in_the_request_body() {
    let wiremock = WireMock::start().await;
    wiremock.stub_post(URL, 200, text_reply()).await;

    client(&wiremock, MODEL)
        .complete(tool_round_trip_request())
        .await
        .expect("completion");

    let expected = json!({
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "extract this"}]},
            {"role": "assistant", "content": [{
                "type": "tool_use", "id": "call_1", "name": "extract", "input": {"facts": ["a"]},
            }]},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "call_1", "content": "boom", "is_error": true,
            }]},
        ],
        "tool_choice": {"type": "auto"},
    });
    assert_eq!(wiremock.count_posts_with_body(URL, expected).await, 1);
}

#[tokio::test]
async fn rate_limit_is_retried_after_retry_after() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_sequence(
            URL,
            json!({
                "status": 429,
                "headers": {"Retry-After": "1"},
                "jsonBody": api_error("slow down"),
            }),
            json!({"status": 200, "jsonBody": text_reply()}),
        )
        .await;

    let started = Instant::now();
    let response = client(&wiremock, MODEL)
        .complete(text_request("hi"))
        .await
        .expect("completion after retry");

    assert!(started.elapsed() >= Duration::from_secs(1));
    assert_eq!(response.output_tokens, 7);
    assert_eq!(wiremock.count_posts(URL).await, 2);
}

#[tokio::test]
async fn retry_after_above_the_cap_fails_without_waiting() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub(json!({
            "request": {"method": "POST", "url": URL},
            "response": {
                "status": 429,
                "headers": {"Retry-After": "3600"},
                "jsonBody": api_error("slow down"),
            },
        }))
        .await;

    let message = provider_error(client(&wiremock, MODEL).complete(text_request("hi")).await);

    assert!(message.contains("429"), "{message}");
    assert_eq!(wiremock.count_posts(URL).await, 1);
}

#[tokio::test]
async fn bad_request_is_not_retried_and_hides_the_request_body() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(URL, 400, api_error("invalid tool schema"))
        .await;

    let message = provider_error(
        client(&wiremock, MODEL)
            .complete(text_request("private user text"))
            .await,
    );

    assert!(message.contains("400"), "{message}");
    assert!(message.contains("invalid tool schema"), "{message}");
    assert!(!message.contains("private user text"), "{message}");
    assert_eq!(wiremock.count_posts(URL).await, 1);
}

#[tokio::test]
async fn server_errors_exhaust_three_retries() {
    let wiremock = WireMock::start().await;
    wiremock.stub_post(URL, 500, api_error("overloaded")).await;

    let message = provider_error(client(&wiremock, MODEL).complete(text_request("hi")).await);

    assert!(message.contains("500"), "{message}");
    assert!(message.contains("overloaded"), "{message}");
    assert_eq!(wiremock.count_posts(URL).await, 4);
}

#[tokio::test]
async fn malformed_tool_input_is_a_provider_error() {
    let wiremock = WireMock::start().await;
    // A tool_use block with no input, then a body that is not JSON at all.
    wiremock
        .stub_sequence(
            URL,
            json!({"status": 200, "jsonBody": {
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "extract"}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }}),
            json!({"status": 200, "body": "{not json"}),
        )
        .await;
    let client = client(&wiremock, MODEL);

    for _ in 0..2 {
        let message = provider_error(client.complete(forced_request()).await);
        assert!(message.contains("malformed"), "{message}");
    }
    assert_eq!(wiremock.count_posts(URL).await, 2);
}

#[tokio::test]
async fn api_key_echoed_by_the_provider_is_scrubbed() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(
            URL,
            401,
            api_error(&format!("invalid x-api-key: {API_KEY}")),
        )
        .await;

    let client = client(&wiremock, MODEL);
    let message = provider_error(client.complete(text_request("hi")).await);

    assert!(message.contains("401"), "{message}");
    assert!(!message.contains(API_KEY), "{message}");
    assert!(!format!("{client:?}").contains(API_KEY));
    assert_eq!(wiremock.count_posts(URL).await, 1);
}

#[tokio::test]
async fn connect_errors_are_retried_then_mapped() {
    let client = AnthropicClient::new(
        API_KEY.into(),
        MODEL.into(),
        Some("http://127.0.0.1:1".into()),
    );

    let started = Instant::now();
    let message = provider_error(client.complete(text_request("hi")).await);

    assert!(message.contains("after 3 retries"), "{message}");
    assert!(!message.contains("127.0.0.1:1/v1"), "{message}");
    assert!(started.elapsed() >= Duration::from_millis(1750));
}
