//! Agent loop with sessions and self-edited memory blocks. See docs/SPEC.md sections 2 and 8.

mod context;
mod runtime;
mod tools;
mod transcript;

use async_trait::async_trait;
use memoryos_core::{RequestContext, Result, Scope, ScoredFact};
use serde::{Deserialize, Serialize};

pub use runtime::AgentRuntime;

/// What the `memory_recall` tool needs from retrieval.
#[async_trait]
pub trait MemoryRecall: Send + Sync {
    async fn recall(
        &self,
        ctx: &RequestContext,
        scopes: Vec<Scope>,
        query: String,
        limit: i64,
    ) -> Result<Vec<ScoredFact>>;
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Placed first in the context. The rule that stored data is not instructions always follows
    /// it, so replacing this text cannot remove that rule.
    pub system_prompt: String,
    /// Output limit of one model call.
    pub max_tokens: u32,
    /// Model replies with tool calls that one turn may act on.
    pub max_tool_iterations: u32,
    /// Stored messages above this size are compacted at the start of a turn.
    pub max_context_chars: usize,
    /// Facts returned by one `memory_recall` call.
    pub recall_limit: i64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful assistant with persistent memory.".to_owned(),
            max_tokens: 4096,
            max_tool_iterations: 10,
            max_context_chars: 100_000,
            recall_limit: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnResult {
    /// All text the model produced in this turn, in order.
    pub text: String,
    pub tool_calls: u32,
    /// Sum over every model call of the turn, including compaction and continuation.
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// The text is still cut off after the one continuation attempt.
    pub truncated: bool,
}
