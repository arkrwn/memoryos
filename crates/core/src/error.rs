use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Domain errors. `server` maps each variant to one HTTP status (see docs/SPEC.md).
#[derive(Debug, Error)]
pub enum Error {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("version conflict: expected {expected}, current {current}")]
    VersionConflict { expected: i64, current: i64 },
    #[error("block exceeds max_chars: {len} > {max}")]
    BlockTooLarge { len: usize, max: usize },
    #[error("embedder error: {0}")]
    Provider(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("internal error: {0}")]
    Internal(String),
}
