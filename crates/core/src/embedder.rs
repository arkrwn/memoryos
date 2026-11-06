use async_trait::async_trait;

use crate::error::Result;

#[async_trait]
pub trait Embedder: Send + Sync {
    /// Returns one vector per input, in order, each `EMBEDDING_DIMS` wide.
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
}
