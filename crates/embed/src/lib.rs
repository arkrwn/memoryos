//! Implementations of the `Embedder` trait from `memoryos-core`. MemoryOS calls no model but this one.

#[cfg(feature = "local")]
mod local;
#[cfg(feature = "test-embedder")]
mod test_embedder;

#[cfg(feature = "local")]
pub use local::LocalEmbedder;
#[cfg(feature = "test-embedder")]
pub use test_embedder::{vector_for, TestEmbedder};
