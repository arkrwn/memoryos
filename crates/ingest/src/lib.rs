//! The write path (SPEC section 3): the host decides the facts, and `HostCommitter` validates them,
//! resolves entities through `entities` and gates them through `gate`.

mod entities;
mod gate;
mod host;
mod redact;
mod single;
mod validate;

pub use host::{
    CommitFact, CommitRequest, CommitResponse, CommittedFact, EntityRef, HostCommitter, RelatedFact,
};
pub use single::{normalize_predicate, SingleValued, DEFAULT_SINGLE_VALUED_PREDICATES};
