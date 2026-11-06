#![cfg(feature = "integration")]

mod common;

use std::time::{Duration, Instant};

use common::*;
use memoryos_core::{ContentBlock, Embedder, Llm, StopReason, EMBEDDING_DIMS};
use memoryos_llm::{OpenAiClient, OpenAiEmbedder};
use serde_json::{json, Value};

const CHAT_URL: &str = "/v1/chat/completions";
const EMBEDDINGS_URL: &str = "/v1/embeddings";
const MODEL: &str = "gpt-5.1";
const EMBEDDING_MODEL: &str = "text-embedding-3-small";

fn client(wiremock: &WireMock) -> OpenAiClient {
    OpenAiClient::new(
        API_KEY.into(),
        MODEL.into(),
        Some(format!("{}/v1", wiremock.base_url)),
    )
}

fn embedder(wiremock: &WireMock) -> OpenAiEmbedder {
    OpenAiEmbedder::new(
        API_KEY.into(),
        EMBEDDING_MODEL.into(),
        Some(format!("{}/v1", wiremock.base_url)),
    )
}

fn chat_reply(message: Value, finish_reason: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    })
}

fn text_reply() -> Value {
    chat_reply(json!({"role": "assistant", "content": "hello"}), "stop")
}

fn tool_reply(arguments: &str) -> Value {
    chat_reply(
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "extract", "arguments": arguments},
            }],
        }),
        "tool_calls",
    )
}

fn api_error(message: &str) -> Value {
    json!({"error": {"message": message, "type": "some_error", "code": null}})
}

/// A vector of the right width whose first element identifies it.
fn vector(first: usize) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIMS];
    vector[0] = first as f32;
    vector
}

fn embeddings_reply(items: Vec<(usize, Vec<f32>)>) -> Value {
    let data: Vec<Value> = items
        .into_iter()
        .map(|(index, embedding)| {
            json!({"object": "embedding", "index": index, "embedding": embedding})
        })
        .collect();
    json!({"object": "list", "model": EMBEDDING_MODEL, "data": data})
}

fn strings(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("input-{i}")).collect()
}

#[tokio::test]
async fn text_reply_maps_response_and_sends_bearer_key() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub(json!({
            "request": {
                "method": "POST",
                "url": CHAT_URL,
                "headers": {"Authorization": {"equalTo": format!("Bearer {API_KEY}")}},
                "bodyPatterns": [{
                    "equalToJson": {
                        "model": MODEL,
                        "max_completion_tokens": 100,
                        "messages": [
                            {"role": "system", "content": "be brief"},
                            {"role": "user", "content": "hi"},
                        ],
                    },
                }],
            },
            "response": {"status": 200, "jsonBody": text_reply()},
        }))
        .await;

    let response = client(&wiremock)
        .complete(text_request("hi"))
        .await
        .expect("completion");

    assert_eq!(
        response.content,
        vec![ContentBlock::Text {
            text: "hello".into()
        }]
    );
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!((response.input_tokens, response.output_tokens), (11, 7));
}

#[tokio::test]
async fn forced_tool_call_parses_string_arguments() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(CHAT_URL, 200, tool_reply("{\"facts\":[\"a\"]}"))
        .await;

    let response = client(&wiremock)
        .complete(forced_request())
        .await
        .expect("completion");

    assert_eq!(
        response.content,
        vec![ContentBlock::ToolUse {
            id: "call_1".into(),
            name: "extract".into(),
            input: json!({"facts": ["a"]}),
        }]
    );
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let expected = json!({
        "tool_choice": {"type": "function", "function": {"name": "extract"}},
        "tools": [{
            "type": "function",
            "function": {
                "name": "extract",
                "description": "Extract facts",
                "parameters": {"type": "object", "properties": {"facts": {"type": "array"}}},
            },
        }],
    });
    assert_eq!(wiremock.count_posts_with_body(CHAT_URL, expected).await, 1);
}

#[tokio::test]
async fn tool_result_round_trips_in_the_request_body() {
    let wiremock = WireMock::start().await;
    wiremock.stub_post(CHAT_URL, 200, text_reply()).await;

    client(&wiremock)
        .complete(tool_round_trip_request())
        .await
        .expect("completion");

    let expected = json!({
        "messages": [
            {"role": "user", "content": "extract this"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "extract", "arguments": "{\"facts\":[\"a\"]}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "[tool error] boom"},
        ],
        "tool_choice": "auto",
    });
    assert_eq!(wiremock.count_posts_with_body(CHAT_URL, expected).await, 1);
}

#[tokio::test]
async fn truncated_text_returns_ok_with_max_tokens() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(
            CHAT_URL,
            200,
            chat_reply(json!({"role": "assistant", "content": "partial"}), "length"),
        )
        .await;

    let response = client(&wiremock)
        .complete(text_request("hi"))
        .await
        .expect("truncated text is not an error");

    assert_eq!(response.stop_reason, StopReason::MaxTokens);
    assert_eq!(
        response.content,
        vec![ContentBlock::Text {
            text: "partial".into()
        }]
    );
    assert_eq!(wiremock.count_posts(CHAT_URL).await, 1);
}

#[tokio::test]
async fn truncated_tool_call_is_a_provider_error() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(
            CHAT_URL,
            200,
            chat_reply(
                json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "extract", "arguments": "{}"},
                    }],
                }),
                "length",
            ),
        )
        .await;

    let message = provider_error(client(&wiremock).complete(forced_request()).await);

    assert!(message.contains("truncated"), "{message}");
    assert_eq!(wiremock.count_posts(CHAT_URL).await, 1);
}

#[tokio::test]
async fn rate_limit_is_retried_after_retry_after() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_sequence(
            CHAT_URL,
            json!({
                "status": 429,
                "headers": {"Retry-After": "1"},
                "jsonBody": api_error("slow down"),
            }),
            json!({"status": 200, "jsonBody": text_reply()}),
        )
        .await;

    let started = Instant::now();
    let response = client(&wiremock)
        .complete(text_request("hi"))
        .await
        .expect("completion after retry");

    assert!(started.elapsed() >= Duration::from_secs(1));
    assert_eq!(response.output_tokens, 7);
    assert_eq!(wiremock.count_posts(CHAT_URL).await, 2);
}

#[tokio::test]
async fn bad_request_is_not_retried_and_hides_the_request_body() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(CHAT_URL, 400, api_error("invalid tool schema"))
        .await;

    let message = provider_error(
        client(&wiremock)
            .complete(text_request("private user text"))
            .await,
    );

    assert!(message.contains("400"), "{message}");
    assert!(message.contains("invalid tool schema"), "{message}");
    assert!(!message.contains("private user text"), "{message}");
    assert_eq!(wiremock.count_posts(CHAT_URL).await, 1);
}

#[tokio::test]
async fn server_errors_exhaust_three_retries() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(CHAT_URL, 500, api_error("overloaded"))
        .await;

    let message = provider_error(client(&wiremock).complete(text_request("hi")).await);

    assert!(message.contains("500"), "{message}");
    assert!(message.contains("overloaded"), "{message}");
    assert_eq!(wiremock.count_posts(CHAT_URL).await, 4);
}

#[tokio::test]
async fn malformed_tool_arguments_are_a_provider_error() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(CHAT_URL, 200, tool_reply("{\"facts\": [\"a\""))
        .await;

    let message = provider_error(client(&wiremock).complete(forced_request()).await);

    assert!(message.contains("malformed"), "{message}");
    assert!(message.contains("extract"), "{message}");
    assert_eq!(wiremock.count_posts(CHAT_URL).await, 1);
}

#[tokio::test]
async fn api_key_echoed_by_the_provider_is_scrubbed() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(
            CHAT_URL,
            401,
            api_error(&format!("Incorrect API key provided: {API_KEY}")),
        )
        .await;

    let client = client(&wiremock);
    let message = provider_error(client.complete(text_request("hi")).await);

    assert!(message.contains("401"), "{message}");
    assert!(!message.contains(API_KEY), "{message}");
    assert!(!format!("{client:?}").contains(API_KEY));
}

#[tokio::test]
async fn embeddings_are_restored_to_input_order() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(
            EMBEDDINGS_URL,
            200,
            embeddings_reply(vec![(2, vector(2)), (0, vector(0)), (1, vector(1))]),
        )
        .await;

    let vectors = embedder(&wiremock)
        .embed(&strings(3))
        .await
        .expect("embeddings");

    let firsts: Vec<f32> = vectors.iter().map(|vector| vector[0]).collect();
    assert_eq!(firsts, [0.0, 1.0, 2.0]);
    let expected = json!({
        "model": EMBEDDING_MODEL,
        "dimensions": memoryos_core::EMBEDDING_DIMS,
        "input": ["input-0", "input-1", "input-2"],
    });
    assert_eq!(
        wiremock
            .count_posts_with_body(EMBEDDINGS_URL, expected)
            .await,
        1
    );
}

#[tokio::test]
async fn wrong_embedding_width_is_a_provider_error() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_post(
            EMBEDDINGS_URL,
            200,
            embeddings_reply(vec![(0, vec![0.0; 3072])]),
        )
        .await;

    let message = provider_error(embedder(&wiremock).embed(&strings(1)).await);

    assert!(message.contains("3072"), "{message}");
}

#[tokio::test]
async fn empty_embedding_input_makes_no_request() {
    let wiremock = WireMock::start().await;

    let vectors = embedder(&wiremock).embed(&[]).await.expect("embeddings");

    assert!(vectors.is_empty());
    assert_eq!(wiremock.count(json!({"urlPattern": ".*"})).await, 0);
}

#[tokio::test]
async fn embeddings_batch_at_256_inputs() {
    let wiremock = WireMock::start().await;
    let first_batch: Vec<(usize, Vec<f32>)> = (0..256).map(|i| (i, vector(i))).collect();
    wiremock
        .stub(json!({
            "request": {
                "method": "POST",
                "url": EMBEDDINGS_URL,
                "bodyPatterns": [
                    {"matchesJsonPath": "$.input[255]"},
                    {"matchesJsonPath": "$[?(@.input.size() == 256)]"},
                ],
            },
            "response": {"status": 200, "jsonBody": embeddings_reply(first_batch)},
        }))
        .await;
    wiremock
        .stub(json!({
            "request": {
                "method": "POST",
                "url": EMBEDDINGS_URL,
                "bodyPatterns": [{"equalToJson": {"input": ["input-256"]}, "ignoreExtraElements": true}],
            },
            // The index restarts in each batch, so results must not be merged by raw index.
            "response": {"status": 200, "jsonBody": embeddings_reply(vec![(0, vector(256))])},
        }))
        .await;

    let vectors = embedder(&wiremock)
        .embed(&strings(257))
        .await
        .expect("embeddings");

    assert_eq!(vectors.len(), 257);
    for (i, vector) in vectors.iter().enumerate() {
        assert_eq!(vector[0], i as f32);
    }
    assert_eq!(wiremock.count_posts(EMBEDDINGS_URL).await, 2);
}

#[tokio::test]
async fn embedding_rate_limit_is_retried_within_a_batch() {
    let wiremock = WireMock::start().await;
    wiremock
        .stub_sequence(
            EMBEDDINGS_URL,
            json!({
                "status": 429,
                "headers": {"Retry-After": "1"},
                "jsonBody": api_error("slow down"),
            }),
            json!({"status": 200, "jsonBody": embeddings_reply(vec![(0, vector(9))])}),
        )
        .await;

    let vectors = embedder(&wiremock)
        .embed(&strings(1))
        .await
        .expect("embeddings");

    assert_eq!(vectors[0][0], 9.0);
    assert_eq!(wiremock.count_posts(EMBEDDINGS_URL).await, 2);
}
