//! Claude Messages API, non-streaming.

use std::fmt;

use async_trait::async_trait;
use memoryos_core::{
    CompletionRequest, CompletionResponse, ContentBlock, Error, Llm, Message, Result, Role,
    ToolChoice,
};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::require_forced_tool;
use crate::transport::{Auth, Transport};

const PROVIDER: &str = "anthropic";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2026-10-21";

/// These models answer `tool_choice` `tool` and `any` with a 400. See `build_request`.
const MODELS_REJECTING_FORCED_TOOL_CHOICE: [&str; 4] = [
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
];

pub struct AnthropicClient {
    transport: Transport,
    model: String,
    base_url: String,
}

impl AnthropicClient {
    /// `base_url` excludes `/v1`, like the official SDKs' `ANTHROPIC_BASE_URL`.
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static(API_VERSION));
        let base_url = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        Self {
            transport: Transport::new(PROVIDER, api_key, Auth::XApiKey, headers),
            model,
            base_url: base_url.trim_end_matches('/').to_owned(),
        }
    }
}

impl fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[async_trait]
impl Llm for AnthropicClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let body = build_request(&self.model, &request);
        let url = format!("{}/v1/messages", self.base_url);
        let response: MessagesResponse = self.transport.post_json(&url, &body).await?;
        parse_response(response, &request.tool_choice)
    }
}

/// Maps `ToolChoice::Tool` to `{"type": "tool"}`, except on models that reject it. There the
/// closest available request is `auto` with only that tool offered, at most one call, and an
/// instruction in the system prompt. `auto` does not guarantee a call, so `parse_response`
/// fails the request when the call is missing.
fn build_request(model: &str, request: &CompletionRequest) -> Value {
    let emulated_tool = match &request.tool_choice {
        ToolChoice::Tool(name) if rejects_forced_tool_choice(model) => Some(name.as_str()),
        _ => None,
    };

    let mut body = json!({
        "model": model,
        "max_tokens": request.max_tokens,
        "messages": request.messages.iter().map(message_to_wire).collect::<Vec<_>>(),
    });

    let system = match emulated_tool {
        Some(name) => {
            let instruction = format!("You must respond by calling the {name} tool.");
            Some(match &request.system {
                Some(system) => format!("{system}\n\n{instruction}"),
                None => instruction,
            })
        }
        None => request.system.clone(),
    };
    if let Some(system) = system {
        body["system"] = json!(system);
    }

    let tools: Vec<Value> = request
        .tools
        .iter()
        .filter(|tool| emulated_tool.is_none_or(|name| tool.name == name))
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
        })
        .collect();
    if !tools.is_empty() {
        body["tools"] = json!(tools);
        body["tool_choice"] = match (&request.tool_choice, emulated_tool) {
            (_, Some(_)) => json!({"type": "auto", "disable_parallel_tool_use": true}),
            (ToolChoice::Tool(name), None) => json!({"type": "tool", "name": name}),
            (ToolChoice::Auto, None) => json!({"type": "auto"}),
        };
    }
    body
}

fn rejects_forced_tool_choice(model: &str) -> bool {
    MODELS_REJECTING_FORCED_TOOL_CHOICE
        .iter()
        .any(|prefix| model.starts_with(prefix))
}

fn message_to_wire(message: &Message) -> Value {
    let role = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content: Vec<Value> = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => json!({"type": "text", "text": text}),
            ContentBlock::ToolUse { id, name, input } => {
                json!({"type": "tool_use", "id": id, "name": name, "input": input})
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
            }),
        })
        .collect();
    json!({"role": role, "content": content})
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<WireBlock>,
    stop_reason: Option<String>,
    usage: WireUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// `thinking` and similar blocks. `ContentBlock` cannot carry them.
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: u64,
    output_tokens: u64,
}

fn parse_response(
    response: MessagesResponse,
    tool_choice: &ToolChoice,
) -> Result<CompletionResponse> {
    let content: Vec<ContentBlock> = response
        .content
        .into_iter()
        .filter_map(|block| match block {
            WireBlock::Text { text } => Some(ContentBlock::Text { text }),
            WireBlock::ToolUse { id, name, input } => {
                Some(ContentBlock::ToolUse { id, name, input })
            }
            WireBlock::Other => None,
        })
        .collect();

    match response.stop_reason.as_deref() {
        Some("refusal") => {
            return Err(Error::Provider(format!("{PROVIDER} refused the request")));
        }
        // A truncated tool input usually still parses, as a valid partial object.
        Some("max_tokens")
            if content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. })) =>
        {
            return Err(Error::Provider(format!(
                "{PROVIDER} tool call was truncated by max_tokens"
            )));
        }
        _ => {}
    }
    require_forced_tool(PROVIDER, tool_choice, &content)?;

    Ok(CompletionResponse {
        content,
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
    })
}

#[cfg(test)]
mod tests {
    use memoryos_core::ToolDef;

    use super::*;

    fn tool(name: &str) -> ToolDef {
        ToolDef {
            name: name.into(),
            description: format!("{name} description"),
            input_schema: json!({"type": "object"}),
        }
    }

    fn request(tools: Vec<ToolDef>, tool_choice: ToolChoice) -> CompletionRequest {
        CompletionRequest {
            system: Some("be brief".into()),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: "hi".into() }],
            }],
            tools,
            tool_choice,
            max_tokens: 100,
        }
    }

    fn wire(body: Value) -> MessagesResponse {
        serde_json::from_value(body).expect("valid wire response")
    }

    #[test]
    fn maps_every_block_type_and_role() {
        let mut req = request(vec![], ToolChoice::Auto);
        req.messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text { text: "t".into() },
                    ContentBlock::ToolUse {
                        id: "a".into(),
                        name: "n".into(),
                        input: json!({"k": 1}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "a".into(),
                    content: "boom".into(),
                    is_error: true,
                }],
            },
        ];
        let body = build_request("claude-opus-5", &req);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "assistant", "content": [
                    {"type": "text", "text": "t"},
                    {"type": "tool_use", "id": "a", "name": "n", "input": {"k": 1}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "boom", "is_error": true},
                ]},
            ])
        );
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["max_tokens"], 100);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn omits_absent_system() {
        let mut req = request(vec![], ToolChoice::Auto);
        req.system = None;
        assert!(build_request("m", &req).get("system").is_none());
    }

    #[test]
    fn maps_tools_and_tool_choice() {
        let auto = build_request(
            "claude-opus-5",
            &request(vec![tool("a"), tool("b")], ToolChoice::Auto),
        );
        assert_eq!(auto["tool_choice"], json!({"type": "auto"}));
        assert_eq!(
            auto["tools"][0],
            json!({"name": "a", "description": "a description", "input_schema": {"type": "object"}})
        );

        let forced = build_request(
            "claude-opus-5",
            &request(vec![tool("a"), tool("b")], ToolChoice::Tool("b".into())),
        );
        assert_eq!(forced["tool_choice"], json!({"type": "tool", "name": "b"}));
        assert_eq!(forced["tools"].as_array().map(Vec::len), Some(2));
        assert_eq!(forced["system"], "be brief");
    }

    #[test]
    fn emulates_forcing_on_models_that_reject_it() {
        for model in MODELS_REJECTING_FORCED_TOOL_CHOICE {
            let body = build_request(
                model,
                &request(vec![tool("a"), tool("b")], ToolChoice::Tool("b".into())),
            );
            assert_eq!(
                body["tool_choice"],
                json!({"type": "auto", "disable_parallel_tool_use": true})
            );
            assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));
            assert_eq!(body["tools"][0]["name"], "b");
            assert_eq!(
                body["system"],
                "be brief\n\nYou must respond by calling the b tool."
            );
        }
    }

    #[test]
    fn auto_is_unchanged_on_models_that_reject_forcing() {
        let body = build_request(
            "claude-sonnet-5-5",
            &request(vec![tool("a")], ToolChoice::Auto),
        );
        assert_eq!(body["tool_choice"], json!({"type": "auto"}));
        assert_eq!(body["system"], "be brief");
    }

    #[test]
    fn parses_blocks_usage_and_skips_thinking() {
        let response = parse_response(
            wire(json!({
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": "hello", "citations": null},
                    {"type": "tool_use", "id": "a", "name": "n", "input": {"k": 1}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 3, "output_tokens": 5},
            })),
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
                    id: "a".into(),
                    name: "n".into(),
                    input: json!({"k": 1}),
                },
            ]
        );
        assert_eq!((response.input_tokens, response.output_tokens), (3, 5));
    }

    fn parse_error(body: Value, choice: &ToolChoice) -> String {
        match parse_response(wire(body), choice) {
            Err(Error::Provider(message)) => message,
            other => panic!("expected provider error, got {other:?}"),
        }
    }

    #[test]
    fn refusal_is_a_provider_error() {
        let message = parse_error(
            json!({"content": [], "stop_reason": "refusal",
                   "usage": {"input_tokens": 1, "output_tokens": 0}}),
            &ToolChoice::Auto,
        );
        assert!(message.contains("refused"), "{message}");
    }

    #[test]
    fn truncated_tool_call_is_a_provider_error() {
        let message = parse_error(
            json!({"content": [{"type": "tool_use", "id": "a", "name": "n", "input": {}}],
                   "stop_reason": "max_tokens",
                   "usage": {"input_tokens": 1, "output_tokens": 1}}),
            &ToolChoice::Auto,
        );
        assert!(message.contains("truncated"), "{message}");
    }

    #[test]
    fn truncated_text_is_returned() {
        let parsed = parse_response(
            wire(json!({"content": [{"type": "text", "text": "par"}],
                        "stop_reason": "max_tokens",
                        "usage": {"input_tokens": 1, "output_tokens": 1}})),
            &ToolChoice::Auto,
        );
        assert!(parsed.is_ok());
    }

    #[test]
    fn missing_forced_call_is_a_provider_error() {
        let message = parse_error(
            json!({"content": [{"type": "text", "text": "no"}], "stop_reason": "end_turn",
                   "usage": {"input_tokens": 1, "output_tokens": 1}}),
            &ToolChoice::Tool("extract".into()),
        );
        assert!(message.contains("extract"), "{message}");
    }

    #[test]
    fn debug_redacts_key() {
        let client = AnthropicClient::new("sk-secret".into(), "m".into(), None);
        let debug = format!("{client:?}");
        assert!(!debug.contains("sk-secret"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    #[test]
    fn trims_trailing_slash_from_base_url() {
        let client = AnthropicClient::new("k".into(), "m".into(), Some("http://x/".into()));
        assert_eq!(client.base_url, "http://x");
    }
}
