//! A real WireMock container, driven through its admin API.

use std::time::{Duration, Instant};

use memoryos_core::{CompletionRequest, ContentBlock, Message, Role, ToolChoice, ToolDef};
use serde_json::{json, Value};
use testcontainers::core::{IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};

pub const API_KEY: &str = "test-key-not-a-real-secret";

pub struct WireMock {
    _container: ContainerAsync<GenericImage>,
    pub base_url: String,
    http: reqwest::Client,
}

impl WireMock {
    pub async fn start() -> Self {
        let container = GenericImage::new("wiremock/wiremock", "3.3.1")
            .with_exposed_port(8080.tcp())
            .with_wait_for(WaitFor::message_on_stdout("port:"))
            .start()
            .await
            .expect("wiremock container starts");
        let port = container
            .get_host_port_ipv4(8080)
            .await
            .expect("mapped port");
        let wiremock = Self {
            _container: container,
            base_url: format!("http://127.0.0.1:{port}"),
            http: reqwest::Client::new(),
        };
        wiremock.wait_until_ready().await;
        wiremock
    }

    /// The startup banner can come before the admin API accepts connections.
    async fn wait_until_ready(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        let url = format!("{}/__admin/mappings", self.base_url);
        while Instant::now() < deadline {
            if let Ok(response) = self.http.get(&url).send().await {
                if response.status().is_success() {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("wiremock admin API did not become ready");
    }

    pub async fn stub(&self, mapping: Value) {
        let response = self
            .http
            .post(format!("{}/__admin/mappings", self.base_url))
            .json(&mapping)
            .send()
            .await
            .expect("register stub");
        assert_eq!(response.status().as_u16(), 201, "stub rejected");
    }

    /// Stubs every POST to `url` with one fixed response.
    pub async fn stub_post(&self, url: &str, status: u16, body: Value) {
        self.stub(json!({
            "request": {"method": "POST", "url": url},
            "response": {"status": status, "jsonBody": body},
        }))
        .await;
    }

    /// Two-step scenario: the first POST gets `first`, every later one gets `second`.
    pub async fn stub_sequence(&self, url: &str, first: Value, second: Value) {
        self.stub(json!({
            "scenarioName": "sequence",
            "requiredScenarioState": "Started",
            "newScenarioState": "second",
            "request": {"method": "POST", "url": url},
            "response": first,
        }))
        .await;
        self.stub(json!({
            "scenarioName": "sequence",
            "requiredScenarioState": "second",
            "request": {"method": "POST", "url": url},
            "response": second,
        }))
        .await;
    }

    /// Counts received requests matching a WireMock request pattern.
    pub async fn count(&self, pattern: Value) -> u64 {
        let response: Value = self
            .http
            .post(format!("{}/__admin/requests/count", self.base_url))
            .json(&pattern)
            .send()
            .await
            .expect("count requests")
            .json()
            .await
            .expect("count response");
        response["count"].as_u64().expect("count field")
    }

    pub async fn count_posts(&self, url: &str) -> u64 {
        self.count(json!({"method": "POST", "url": url})).await
    }

    /// Counts POSTs whose JSON body contains `subset`.
    pub async fn count_posts_with_body(&self, url: &str, subset: Value) -> u64 {
        self.count(json!({
            "method": "POST",
            "url": url,
            "bodyPatterns": [{
                "equalToJson": subset,
                "ignoreExtraElements": true,
            }],
        }))
        .await
    }
}

pub fn user_text(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text { text: text.into() }],
    }
}

pub fn extract_tool() -> ToolDef {
    ToolDef {
        name: "extract".into(),
        description: "Extract facts".into(),
        input_schema: json!({"type": "object", "properties": {"facts": {"type": "array"}}}),
    }
}

pub fn text_request(text: &str) -> CompletionRequest {
    CompletionRequest {
        system: Some("be brief".into()),
        messages: vec![user_text(text)],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: 100,
    }
}

pub fn forced_request() -> CompletionRequest {
    CompletionRequest {
        system: None,
        messages: vec![user_text("extract this")],
        tools: vec![extract_tool()],
        tool_choice: ToolChoice::Tool("extract".into()),
        max_tokens: 100,
    }
}

/// An assistant tool call followed by its failed result.
pub fn tool_round_trip_request() -> CompletionRequest {
    CompletionRequest {
        system: None,
        messages: vec![
            user_text("extract this"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "extract".into(),
                    input: json!({"facts": ["a"]}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "boom".into(),
                    is_error: true,
                }],
            },
        ],
        tools: vec![extract_tool()],
        tool_choice: ToolChoice::Auto,
        max_tokens: 100,
    }
}

pub fn provider_error<T>(result: memoryos_core::Result<T>) -> String {
    match result {
        Err(memoryos_core::Error::Provider(message)) => message,
        Err(other) => panic!("expected Error::Provider, got {other:?}"),
        Ok(_) => panic!("expected Error::Provider, got Ok"),
    }
}
