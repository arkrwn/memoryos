//! OpenAI Chat Completions and Embeddings.

use std::fmt;

use async_trait::async_trait;
use memoryos_core::{
    CompletionRequest, CompletionResponse, ContentBlock, Embedder, Error, Llm, Message, Result,
    Role, ToolChoice, EMBEDDING_DIMS,
};
use reqwest::header::HeaderMap;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::require_forced_tool;
use crate::transport::{Auth, Transport};

const PROVIDER: &str = "openai";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const MAX_EMBEDDING_BATCH: usize = 256;
/// Chat Completions has no error flag on a tool message, so the model reads it from the text.
const TOOL_ERROR_PREFIX: &str = "[tool error] ";

fn normalized_base_url(base_url: Option<String>) -> String {
    base_url
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
        .trim_end_matches('/')
        .to_owned()
}

pub struct OpenAiClient {
    transport: Transport,
    model: String,
    base_url: String,
}

impl OpenAiClient {
    /// `base_url` includes `/v1`, like the official SDKs' `OPENAI_BASE_URL`.
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Self {
        Self {
            transport: Transport::new(PROVIDER, api_key, Auth::Bearer, HeaderMap::new()),
            model,
            base_url: normalized_base_url(base_url),
        }
    }
}

impl fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiClient")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[async_trait]
impl Llm for OpenAiClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let body = build_chat_request(&self.model, &request)?;
        let url = format!("{}/chat/completions", self.base_url);
        let response: ChatResponse = self.transport.post_json(&url, &body).await?;
        parse_chat_response(response, &request.tool_choice)
    }
}

fn build_chat_request(model: &str, request: &CompletionRequest) -> Result<Value> {
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    for message in &request.messages {
        append_message(&mut messages, message)?;
    }

    let mut body = json!({
        "model": model,
        // Current models reject the older `max_tokens` name on this endpoint.
        "max_completion_tokens": request.max_tokens,
        "messages": messages,
    });
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                })
            })
            .collect();
        body["tools"] = json!(tools);
        body["tool_choice"] = match &request.tool_choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Tool(name) => json!({"type": "function", "function": {"name": name}}),
        };
    }
    Ok(body)
}

/// One core message can become several: each tool result is its own `tool` message. Those must
/// directly follow the assistant's `tool_calls`, so they go before any text in the same turn.
fn append_message(out: &mut Vec<Value>, message: &Message) -> Result<()> {
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in &message.content {
        match (block, message.role) {
            (ContentBlock::Text { text }, _) => texts.push(text.as_str()),
            (ContentBlock::ToolUse { id, name, input }, Role::Assistant) => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()},
                }));
            }
            (
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                },
                Role::User,
            ) => {
                let content = if *is_error {
                    format!("{TOOL_ERROR_PREFIX}{content}")
                } else {
                    content.clone()
                };
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content,
                }));
            }
            (ContentBlock::ToolUse { .. }, Role::User) => {
                return Err(Error::InvalidInput(
                    "tool_use block in a user message".to_owned(),
                ));
            }
            (ContentBlock::ToolResult { .. }, Role::Assistant) => {
                return Err(Error::InvalidInput(
                    "tool_result block in an assistant message".to_owned(),
                ));
            }
        }
    }

    let text = texts.join("\n");
    match message.role {
        Role::User => {
            if !text.is_empty() {
                out.push(json!({"role": "user", "content": text}));
            }
        }
        Role::Assistant if tool_calls.is_empty() => {
            out.push(json!({"role": "assistant", "content": text}));
        }
        Role::Assistant => {
            let content = if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            };
            out.push(json!({"role": "assistant", "content": content, "tool_calls": tool_calls}));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: Option<String>,
    refusal: Option<String>,
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Deserialize)]
struct ChatToolCall {
    id: String,
    function: ChatFunction,
}

#[derive(Deserialize)]
struct ChatFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct ChatUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

fn parse_chat_response(
    response: ChatResponse,
    tool_choice: &ToolChoice,
) -> Result<CompletionResponse> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Provider(format!("{PROVIDER} returned no choices")))?;
    if let Some(refusal) = choice.message.refusal {
        return Err(Error::Provider(format!(
            "{PROVIDER} refused the request: {refusal}"
        )));
    }

    let tool_calls = choice.message.tool_calls.unwrap_or_default();
    if choice.finish_reason.as_deref() == Some("length") && !tool_calls.is_empty() {
        return Err(Error::Provider(format!(
            "{PROVIDER} tool call was truncated by max_completion_tokens"
        )));
    }

    let mut content = Vec::with_capacity(tool_calls.len() + 1);
    if let Some(text) = choice.message.content.filter(|text| !text.is_empty()) {
        content.push(ContentBlock::Text { text });
    }
    for call in tool_calls {
        content.push(ContentBlock::ToolUse {
            input: parse_tool_arguments(&call.function.name, &call.function.arguments)?,
            id: call.id,
            name: call.function.name,
        });
    }
    require_forced_tool(PROVIDER, tool_choice, &content)?;

    Ok(CompletionResponse {
        content,
        input_tokens: response.usage.prompt_tokens,
        output_tokens: response.usage.completion_tokens,
    })
}

/// The arguments are model output and may hold user data, so errors do not quote them.
fn parse_tool_arguments(name: &str, arguments: &str) -> Result<Value> {
    let value: Value = serde_json::from_str(arguments).map_err(|err| {
        Error::Provider(format!(
            "{PROVIDER} returned malformed JSON arguments for tool {name}: line {} column {}",
            err.line(),
            err.column()
        ))
    })?;
    if !value.is_object() {
        return Err(Error::Provider(format!(
            "{PROVIDER} returned non-object arguments for tool {name}"
        )));
    }
    Ok(value)
}

pub struct OpenAiEmbedder {
    transport: Transport,
    model: String,
    base_url: String,
}

impl OpenAiEmbedder {
    /// `base_url` includes `/v1`, like the official SDKs' `OPENAI_BASE_URL`.
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Self {
        Self {
            transport: Transport::new(PROVIDER, api_key, Auth::Bearer, HeaderMap::new()),
            model,
            base_url: normalized_base_url(base_url),
        }
    }
}

impl fmt::Debug for OpenAiEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiEmbedder")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let mut vectors = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(MAX_EMBEDDING_BATCH) {
            let body = json!({
                "model": self.model,
                "input": batch,
                "dimensions": EMBEDDING_DIMS,
            });
            let response: EmbeddingResponse = self.transport.post_json(&url, &body).await?;
            vectors.extend(order_embeddings(response, batch.len())?);
        }
        Ok(vectors)
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

/// Restores input order by `index` and rejects anything but one correctly sized vector per input.
fn order_embeddings(response: EmbeddingResponse, expected: usize) -> Result<Vec<Vec<f32>>> {
    if response.data.len() != expected {
        return Err(Error::Provider(format!(
            "{PROVIDER} returned {} embeddings for {expected} inputs",
            response.data.len()
        )));
    }
    let mut slots: Vec<Option<Vec<f32>>> = vec![None; expected];
    for item in response.data {
        if item.embedding.len() != EMBEDDING_DIMS {
            return Err(Error::Provider(format!(
                "{PROVIDER} returned an embedding of width {}, expected {EMBEDDING_DIMS}",
                item.embedding.len()
            )));
        }
        let slot = slots.get_mut(item.index).ok_or_else(|| {
            Error::Provider(format!(
                "{PROVIDER} returned embedding index {} for {expected} inputs",
                item.index
            ))
        })?;
        if slot.replace(item.embedding).is_some() {
            return Err(Error::Provider(format!(
                "{PROVIDER} returned embedding index {} twice",
                item.index
            )));
        }
    }
    // The length check and the duplicate check together mean every slot is filled.
    Ok(slots.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use memoryos_core::ToolDef;

    use super::*;

    fn request(messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            system: None,
            messages,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            max_tokens: 50,
        }
    }

    fn wire(body: Value) -> ChatResponse {
        serde_json::from_value(body).expect("valid wire response")
    }

    fn reply(message: Value, finish_reason: &str) -> ChatResponse {
        wire(json!({
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }))
    }

    fn provider_message<T: fmt::Debug>(result: Result<T>) -> String {
        match result {
            Err(Error::Provider(message)) => message,
            other => panic!("expected provider error, got {other:?}"),
        }
    }

    #[test]
    fn system_becomes_first_message() {
        let mut req = request(vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "hi".into() }],
        }]);
        req.system = Some("be brief".into());
        let body = build_chat_request("m", &req).expect("built");
        assert_eq!(
            body["messages"],
            json!([
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ])
        );
        assert_eq!(body["max_completion_tokens"], 50);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn tool_use_arguments_become_a_json_string() {
        let body = build_chat_request(
            "m",
            &request(vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text { text: "t".into() },
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "n".into(),
                        input: json!({"k": 1}),
                    },
                ],
            }]),
        )
        .expect("built");
        assert_eq!(
            body["messages"],
            json!([{"role": "assistant", "content": "t", "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "n", "arguments": "{\"k\":1}"},
            }]}])
        );
    }

    #[test]
    fn assistant_with_only_tool_calls_has_null_content() {
        let body = build_chat_request(
            "m",
            &request(vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "n".into(),
                    input: json!({}),
                }],
            }]),
        )
        .expect("built");
        assert_eq!(body["messages"][0]["content"], Value::Null);
    }

    #[test]
    fn tool_results_become_tool_messages_before_user_text() {
        let body = build_chat_request(
            "m",
            &request(vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "next".into(),
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "c1".into(),
                        content: "ok".into(),
                        is_error: false,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "c2".into(),
                        content: "boom".into(),
                        is_error: true,
                    },
                ],
            }]),
        )
        .expect("built");
        assert_eq!(
            body["messages"],
            json!([
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "tool", "tool_call_id": "c2", "content": "[tool error] boom"},
                {"role": "user", "content": "next"},
            ])
        );
    }

    #[test]
    fn misplaced_blocks_are_invalid_input() {
        let tool_use_in_user = request(vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolUse {
                id: "c".into(),
                name: "n".into(),
                input: json!({}),
            }],
        }]);
        assert!(matches!(
            build_chat_request("m", &tool_use_in_user),
            Err(Error::InvalidInput(_))
        ));
        let result_in_assistant = request(vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "c".into(),
                content: "x".into(),
                is_error: false,
            }],
        }]);
        assert!(matches!(
            build_chat_request("m", &result_in_assistant),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn maps_tools_and_tool_choice() {
        let mut req = request(vec![]);
        req.tools = vec![ToolDef {
            name: "a".into(),
            description: "d".into(),
            input_schema: json!({"type": "object"}),
        }];
        let auto = build_chat_request("m", &req).expect("built");
        assert_eq!(auto["tool_choice"], "auto");
        assert_eq!(
            auto["tools"][0],
            json!({"type": "function", "function": {
                "name": "a", "description": "d", "parameters": {"type": "object"},
            }})
        );
        req.tool_choice = ToolChoice::Tool("a".into());
        let forced = build_chat_request("m", &req).expect("built");
        assert_eq!(
            forced["tool_choice"],
            json!({"type": "function", "function": {"name": "a"}})
        );
    }

    #[test]
    fn parses_text_tool_calls_and_usage() {
        let response = parse_chat_response(
            reply(
                json!({"content": "hello", "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "n", "arguments": "{\"k\":1}"},
                }]}),
                "tool_calls",
            ),
            &ToolChoice::Auto,
        )
        .expect("parsed");
        assert_eq!(
            response.content,
            vec![
                ContentBlock::Text {
                    text: "hello".into()
                },
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "n".into(),
                    input: json!({"k": 1}),
                },
            ]
        );
        assert_eq!((response.input_tokens, response.output_tokens), (3, 5));
    }

    #[test]
    fn malformed_arguments_do_not_leak_into_the_error() {
        let message = provider_message(parse_tool_arguments("n", "{\"ssn\": 123-"));
        assert!(message.contains("malformed"), "{message}");
        assert!(!message.contains("ssn"), "{message}");
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        let message = provider_message(parse_tool_arguments("n", "[1]"));
        assert!(message.contains("non-object"), "{message}");
    }

    #[test]
    fn empty_choices_refusal_and_truncation_are_provider_errors() {
        let empty =
            wire(json!({"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}));
        assert!(
            provider_message(parse_chat_response(empty, &ToolChoice::Auto)).contains("no choices")
        );

        let refused = reply(json!({"content": null, "refusal": "no"}), "stop");
        assert!(
            provider_message(parse_chat_response(refused, &ToolChoice::Auto)).contains("refused")
        );

        let truncated = reply(
            json!({"content": null, "tool_calls": [{
                "id": "c", "type": "function", "function": {"name": "n", "arguments": "{}"},
            }]}),
            "length",
        );
        assert!(
            provider_message(parse_chat_response(truncated, &ToolChoice::Auto))
                .contains("truncated")
        );
    }

    #[test]
    fn missing_forced_call_is_a_provider_error() {
        let message = provider_message(parse_chat_response(
            reply(json!({"content": "no"}), "stop"),
            &ToolChoice::Tool("extract".into()),
        ));
        assert!(message.contains("extract"), "{message}");
    }

    fn vector(first: f32) -> Vec<f32> {
        let mut vector = vec![0.0; EMBEDDING_DIMS];
        vector[0] = first;
        vector
    }

    fn embeddings(items: Vec<(usize, Vec<f32>)>) -> EmbeddingResponse {
        EmbeddingResponse {
            data: items
                .into_iter()
                .map(|(index, embedding)| EmbeddingItem { index, embedding })
                .collect(),
        }
    }

    #[test]
    fn embeddings_are_reordered_by_index() {
        let ordered = order_embeddings(
            embeddings(vec![(2, vector(2.0)), (0, vector(0.0)), (1, vector(1.0))]),
            3,
        )
        .expect("ordered");
        let firsts: Vec<f32> = ordered.iter().map(|v| v[0]).collect();
        assert_eq!(firsts, [0.0, 1.0, 2.0]);
    }

    #[test]
    fn bad_embedding_responses_are_provider_errors() {
        let cases = [
            (embeddings(vec![(0, vector(0.0))]), 2, "1 embeddings"),
            (embeddings(vec![(0, vec![0.0; 3])]), 1, "width 3"),
            (embeddings(vec![(5, vector(0.0))]), 1, "index 5"),
            (
                embeddings(vec![(0, vector(0.0)), (0, vector(1.0))]),
                2,
                "twice",
            ),
        ];
        for (response, expected, needle) in cases {
            let message = provider_message(order_embeddings(response, expected));
            assert!(message.contains(needle), "{message}");
        }
    }

    #[test]
    fn debug_redacts_key() {
        let chat = OpenAiClient::new("sk-secret".into(), "m".into(), None);
        let embedder = OpenAiEmbedder::new("sk-secret".into(), "m".into(), None);
        for debug in [format!("{chat:?}"), format!("{embedder:?}")] {
            assert!(!debug.contains("sk-secret"), "{debug}");
            assert!(debug.contains("[REDACTED]"), "{debug}");
        }
    }

    #[tokio::test]
    async fn empty_embedding_input_makes_no_call() {
        // Nothing listens on this address, so any HTTP call would fail the test.
        let embedder =
            OpenAiEmbedder::new("k".into(), "m".into(), Some("http://127.0.0.1:1/v1".into()));
        assert_eq!(
            embedder.embed(&[]).await.expect("empty"),
            Vec::<Vec<f32>>::new()
        );
    }
}
