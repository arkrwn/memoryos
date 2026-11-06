//! Shared types, errors and traits for MemoryOS. This crate performs no I/O.

pub mod error;
pub mod llm;
pub mod store;
pub mod types;

pub use error::{Error, Result};
pub use llm::{
    CompletionRequest, CompletionResponse, ContentBlock, Embedder, Llm, Message, Role, ToolChoice,
    ToolDef,
};
pub use store::MemoryStore;
pub use types::*;

/// Embedding width fixed by `migrations/0001_init.sql`. See ADR 0003.
pub const EMBEDDING_DIMS: usize = 1536;
