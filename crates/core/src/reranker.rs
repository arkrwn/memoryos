use async_trait::async_trait;

use crate::error::Result;

/// Scores how well each statement answers a query, reading the two together. Recall uses it to
/// reorder the top of its fused list.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Returns one score per statement, in order. Higher means more relevant. Scores only compare
    /// within one call.
    async fn score(&self, query: &str, statements: &[String]) -> Result<Vec<f32>>;

    /// Gets the reranker ready for `score`, for example by loading a model. Does nothing by default.
    async fn warm_up(&self) -> Result<()> {
        Ok(())
    }
}
