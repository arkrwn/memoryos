//! Implementations of the `Embedder` and `Reranker` traits from `memoryos-core`. MemoryOS calls no
//! models but these.

#[cfg(feature = "local")]
mod local;
#[cfg(feature = "local")]
mod rerank;
#[cfg(feature = "test-embedder")]
mod test_embedder;

#[cfg(feature = "local")]
pub use local::LocalEmbedder;
#[cfg(feature = "local")]
pub use rerank::LocalReranker;
#[cfg(feature = "test-embedder")]
pub use test_embedder::{vector_for, TestEmbedder};
