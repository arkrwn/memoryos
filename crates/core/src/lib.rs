//! Shared types, errors and traits for MemoryOS. This crate performs no I/O.

pub mod embedder;
pub mod error;
pub mod reranker;
pub mod store;
pub mod types;

pub use embedder::Embedder;
pub use error::{Error, Result};
pub use reranker::Reranker;
pub use store::MemoryStore;
pub use types::*;

/// Embedding width fixed by both backend schemas. See ADR 0003.
pub const EMBEDDING_DIMS: usize = 384;
