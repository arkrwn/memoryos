//! Provider clients behind the `Llm` and `Embedder` traits from `memoryos-core`.

mod anthropic;
#[cfg(feature = "local")]
mod local;
mod openai;
mod transport;

pub use anthropic::AnthropicClient;
#[cfg(feature = "local")]
pub use local::LocalEmbedder;
pub use openai::{OpenAiClient, OpenAiEmbedder};

use memoryos_core::{ContentBlock, Error, Result, ToolChoice};

/// A forced call that did not happen must not look like a normal reply: extraction parses the
/// tool input and has no other way to tell.
fn require_forced_tool(
    provider: &str,
    choice: &ToolChoice,
    content: &[ContentBlock],
) -> Result<()> {
    let ToolChoice::Tool(forced) = choice else {
        return Ok(());
    };
    let called = content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolUse { name, .. } if name == forced));
    if called {
        return Ok(());
    }
    Err(Error::Provider(format!(
        "{provider} did not call the forced tool {forced}"
    )))
}
