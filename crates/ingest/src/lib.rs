//! The write path: LLM extraction driven by a job worker (SPEC section 3) and host-decided commits
//! (SPEC section 14). Both build their `FactOp`s through `gate` and resolve entities through
//! `entities`, so the two modes cannot disagree on trust or identity.

mod entities;
mod extract;
mod gate;
mod host;
mod ingestor;
mod validate;
mod worker;

pub use host::{
    CommitFact, CommitRequest, CommitResponse, CommittedFact, EntityRef, HostCommitter,
};
pub use ingestor::Ingestor;
pub use tokio_util::sync::CancellationToken;
pub use worker::{Worker, WorkerConfig};
