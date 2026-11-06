//! The trust gate of SPEC section 5, and the only place that builds a `FactOp`. The worker and host
//! mode both come through `build_op`.

use chrono::{DateTime, Utc};
use memoryos_core::{source_requires_quarantine, FactOp, FactStatus, NewFact, Scope, Trust};
use uuid::Uuid;

/// The episode a fact derives from. A fact's trust always equals its episode's trust.
pub(crate) struct Source<'a> {
    pub(crate) scope: &'a Scope,
    pub(crate) trust: Trust,
}

/// A fact with everything decided except its status.
pub(crate) struct FactDraft {
    pub(crate) subject_id: Uuid,
    pub(crate) predicate: String,
    pub(crate) object_id: Option<Uuid>,
    pub(crate) statement: String,
    pub(crate) embedding: Vec<f32>,
    pub(crate) valid_from: DateTime<Utc>,
}

/// What a statement does to existing memory, after validation against the candidates.
pub(crate) enum Resolution {
    Add(FactDraft),
    Supersede {
        old_fact_id: Uuid,
        old_trust: Trust,
        draft: FactDraft,
    },
    Corroborate {
        fact_id: Uuid,
    },
}

/// SPEC section 5. `contradicted` is the trust of the fact the statement would supersede.
pub(crate) fn status_for(scope: &Scope, trust: Trust, contradicted: Option<Trust>) -> FactStatus {
    let outranked = contradicted.is_some_and(|old| old > trust);
    if source_requires_quarantine(scope, trust) || outranked {
        FactStatus::Quarantined
    } else {
        FactStatus::Active
    }
}

/// A quarantined `Supersede` stays a `Supersede`: the store then leaves the old fact open and
/// records `pending_supersede_of`, so approval can perform the close. `Corroborate` carries no
/// trust at all, so it can never raise or lower the fact it links to.
pub(crate) fn build_op(source: &Source<'_>, resolution: Resolution) -> FactOp {
    match resolution {
        Resolution::Add(draft) => FactOp::Add(new_fact(source, draft, None)),
        Resolution::Supersede {
            old_fact_id,
            old_trust,
            draft,
        } => FactOp::Supersede {
            old_fact_id,
            new: new_fact(source, draft, Some(old_trust)),
        },
        Resolution::Corroborate { fact_id } => FactOp::Corroborate { fact_id },
    }
}

fn new_fact(source: &Source<'_>, draft: FactDraft, contradicted: Option<Trust>) -> NewFact {
    NewFact {
        scope: source.scope.clone(),
        subject_id: draft.subject_id,
        predicate: draft.predicate,
        object_id: draft.object_id,
        statement: draft.statement,
        embedding: draft.embedding,
        trust: source.trust,
        status: status_for(source.scope, source.trust, contradicted),
        valid_from: draft.valid_from,
        valid_to: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use FactStatus::{Active, Quarantined};
    use Trust::{Agent, External, System, User};

    const ALL_CONTRADICTED: [Option<Trust>; 5] =
        [None, Some(External), Some(Agent), Some(User), Some(System)];

    fn scopes() -> [Scope; 3] {
        [Scope::user("u"), Scope::agent("a"), Scope::shared("s")]
    }

    fn draft() -> FactDraft {
        FactDraft {
            subject_id: Uuid::now_v7(),
            predicate: "p".into(),
            object_id: None,
            statement: "s".into(),
            embedding: vec![1.0],
            valid_from: Utc::now(),
        }
    }

    #[test]
    fn row_external_episode_is_always_quarantined() {
        for scope in scopes() {
            for contradicted in ALL_CONTRADICTED {
                assert_eq!(status_for(&scope, External, contradicted), Quarantined);
            }
        }
    }

    #[test]
    fn row_agent_in_user_or_shared_scope_is_quarantined() {
        for scope in [Scope::user("u"), Scope::shared("s")] {
            for contradicted in ALL_CONTRADICTED {
                assert_eq!(status_for(&scope, Agent, contradicted), Quarantined);
            }
        }
    }

    #[test]
    fn row_contradicting_higher_trust_is_quarantined() {
        let agent_scope = Scope::agent("a");
        assert_eq!(status_for(&agent_scope, Agent, Some(User)), Quarantined);
        assert_eq!(status_for(&agent_scope, Agent, Some(System)), Quarantined);
        for scope in scopes() {
            assert_eq!(status_for(&scope, User, Some(System)), Quarantined);
        }
    }

    #[test]
    fn row_contradicting_equal_or_lower_trust_is_active() {
        let agent_scope = Scope::agent("a");
        assert_eq!(status_for(&agent_scope, Agent, Some(Agent)), Active);
        assert_eq!(status_for(&agent_scope, Agent, Some(External)), Active);
        for scope in scopes() {
            for old in [External, Agent, User] {
                assert_eq!(status_for(&scope, User, Some(old)), Active);
            }
            for old in [External, Agent, User, System] {
                assert_eq!(status_for(&scope, System, Some(old)), Active);
            }
        }
    }

    #[test]
    fn row_otherwise_active() {
        for scope in scopes() {
            assert_eq!(status_for(&scope, User, None), Active);
            assert_eq!(status_for(&scope, System, None), Active);
        }
        assert_eq!(status_for(&Scope::agent("a"), Agent, None), Active);
    }

    #[test]
    fn quarantined_supersede_stays_a_supersede() {
        let scope = Scope::agent("a");
        let source = Source {
            scope: &scope,
            trust: Agent,
        };
        let old_fact_id = Uuid::now_v7();
        let op = build_op(
            &source,
            Resolution::Supersede {
                old_fact_id,
                old_trust: User,
                draft: draft(),
            },
        );
        match op {
            FactOp::Supersede {
                old_fact_id: id,
                new,
            } => {
                assert_eq!(id, old_fact_id);
                assert_eq!(new.status, Quarantined);
                assert_eq!(new.trust, Agent);
            }
            other => panic!("expected Supersede, got {other:?}"),
        }
    }

    #[test]
    fn active_supersede_and_add_take_the_episode_trust() {
        let scope = Scope::user("u");
        let source = Source {
            scope: &scope,
            trust: User,
        };
        let FactOp::Supersede { new, .. } = build_op(
            &source,
            Resolution::Supersede {
                old_fact_id: Uuid::now_v7(),
                old_trust: Agent,
                draft: draft(),
            },
        ) else {
            panic!("expected Supersede");
        };
        assert_eq!((new.status, new.trust), (Active, User));
        let FactOp::Add(new) = build_op(&source, Resolution::Add(draft())) else {
            panic!("expected Add");
        };
        assert_eq!((new.status, new.trust), (Active, User));
    }

    /// A lower-trust restatement only links a source. The op has no trust or status to apply.
    #[test]
    fn corroborate_carries_no_trust() {
        let scope = Scope::user("u");
        let source = Source {
            scope: &scope,
            trust: External,
        };
        let fact_id = Uuid::now_v7();
        let op = build_op(&source, Resolution::Corroborate { fact_id });
        assert!(matches!(op, FactOp::Corroborate { fact_id: id } if id == fact_id));
    }
}
