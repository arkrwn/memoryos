//! Typed HTTP client for the MemoryOS API (`docs/openapi.yaml`).
//!
//! ```no_run
//! use memoryos_sdk::{date, Commit, CommitFact, EntityRef, MemoryOs, Scope};
//!
//! # async fn example() -> Result<(), memoryos_sdk::Error> {
//! let mem = MemoryOs::connect("http://localhost:8080", "mos_...").await?;
//! let user = Scope::user("u_123");
//!
//! let fact = CommitFact::new("Ada moved to Bandung", EntityRef::new("Ada", "person"), "lives_in");
//! mem.commit(&Commit::new(user.clone(), "chat", "I moved to Bandung", vec![fact]))
//!     .await?;
//!
//! let now = mem.recall(user.clone(), "where does the user live?").await?;
//! let then = mem
//!     .recall(user, "where does the user live?")
//!     .as_of(date(2025, 1, 1))
//!     .await?;
//! # let _ = (now, then);
//! # Ok(())
//! # }
//! ```
//!
//! Reads are retried on connection failures, on 429 and on 502, 503 and 504. Writes are never
//! retried, because a lost response does not say whether the write happened. To retry one safely,
//! set an idempotency key and call it again.

mod client;
mod error;
mod types;

pub use client::{MemoryOs, Options, RecallBuilder};
pub use error::{Error, Result};
pub use memoryos_core::{
    Block, BlockVersion, Entity, Episode, Fact, FactStatus, GraphNeighborhood, PurgeReport, Scope,
    ScoredFact, Trust,
};
pub use types::{
    date, Commit, CommitFact, CommitResponse, CommittedFact, EntityRef, PutBlock, RecallResponse,
};
