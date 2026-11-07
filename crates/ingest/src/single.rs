//! Single-valued predicates (SPEC section 3). A subject holds one value of these at a time, so a
//! commit that states a new value replaces the current one even when the host did not name it in
//! `supersedes`. The list is small on purpose: a predicate wrongly listed here would close facts
//! that were both true.

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use memoryos_core::{Fact, FactStatus, Scope};
use uuid::Uuid;

use crate::validate::normalize_statement;

/// Used when `MEMORYOS_SINGLE_VALUED_PREDICATES` is unset.
pub const DEFAULT_SINGLE_VALUED_PREDICATES: &[&str] = &[
    "lives_in",
    "works_at",
    "has_nickname",
    "has_role",
    "prefers_editor",
    "uses_database",
    "has_name",
    "has_email",
    "located_in",
    "current_version",
    "prefers_language",
];

/// The predicates that hold one value at a time per subject, kept normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleValued {
    names: BTreeSet<String>,
}

impl Default for SingleValued {
    fn default() -> Self {
        Self::new(DEFAULT_SINGLE_VALUED_PREDICATES)
    }
}

impl SingleValued {
    /// Blank names are dropped, so an empty list turns the rule off.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let names = names
            .into_iter()
            .map(|name| normalize_predicate(name.as_ref()))
            .filter(|name| !name.is_empty())
            .collect();
        Self { names }
    }

    /// A comma-separated list, as in `MEMORYOS_SINGLE_VALUED_PREDICATES`.
    pub fn parse(list: &str) -> Self {
        Self::new(list.split(','))
    }

    pub fn contains(&self, predicate: &str) -> bool {
        self.names.contains(&normalize_predicate(predicate))
    }

    /// Normalized, in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }
}

/// `livesIn`, `Lives In`, `lives-in` and `LIVES_IN` are all `lives_in`: lowercase snake case, with
/// a word break before an uppercase letter that follows a lowercase letter or digit, and every run
/// of other characters one underscore.
pub fn normalize_predicate(predicate: &str) -> String {
    let mut out = String::with_capacity(predicate.len() + 4);
    let mut previous_lower_or_digit = false;
    for c in predicate.trim().chars() {
        if c.is_alphanumeric() {
            if c.is_uppercase() && previous_lower_or_digit {
                out.push('_');
            }
            out.extend(c.to_lowercase());
            previous_lower_or_digit = c.is_lowercase() || c.is_numeric();
        } else {
            if !out.ends_with('_') {
                out.push('_');
            }
            previous_lower_or_digit = false;
        }
    }
    out.trim_matches('_').to_owned()
}

/// What a new fact on a single-valued predicate is compared with.
pub(crate) struct NewValue<'a> {
    pub(crate) scope: &'a Scope,
    pub(crate) predicate: &'a str,
    pub(crate) object_id: Option<Uuid>,
    pub(crate) statement: &'a str,
}

/// The current fact `new` replaces: among `current`, the subject's current facts oldest first, the
/// latest that has the same normalized predicate and a different value. A different value means
/// another object, or the same object said differently. Facts this request already supersedes are
/// skipped, as are any that are not current, which `current` should not hold anyway.
pub(crate) fn replaced<'a>(
    current: &'a [Fact],
    new: &NewValue<'_>,
    closing: &dyn Fn(Uuid) -> bool,
    now: DateTime<Utc>,
) -> Option<&'a Fact> {
    let predicate = normalize_predicate(new.predicate);
    let statement = normalize_statement(new.statement);
    current.iter().rev().find(|old| {
        old.scope == *new.scope
            && old.status == FactStatus::Active
            && old.superseded_by.is_none()
            && old.invalidated_at.is_none()
            && old.valid_to.is_none_or(|to| to > now)
            && !old.expired_at(now)
            && !closing(old.id)
            && normalize_predicate(&old.predicate) == predicate
            && !(old.object_id == new.object_id && normalize_statement(&old.statement) == statement)
    })
}

#[cfg(test)]
mod tests {
    use chrono::Duration;
    use memoryos_core::Trust;

    use super::*;

    #[test]
    fn predicates_normalize_to_lower_snake_case() {
        for variant in [
            "lives_in",
            "livesIn",
            "LivesIn",
            "Lives In",
            "lives-in",
            " LIVES_IN ",
            "lives__in",
        ] {
            assert_eq!(normalize_predicate(variant), "lives_in", "{variant:?}");
        }
        assert_eq!(normalize_predicate("currentVersion2"), "current_version2");
        assert_eq!(normalize_predicate("v2Name"), "v2_name");
        assert_eq!(normalize_predicate("--"), "");
    }

    #[test]
    fn the_default_list_matches_variants_and_nothing_else() {
        let list = SingleValued::default();
        assert!(list.contains("lives_in"));
        assert!(list.contains("worksAt"));
        assert!(list.contains("Prefers Editor"));
        assert!(!list.contains("likes"));
        assert!(!list.contains("lives"));
        assert_eq!(list.names().count(), DEFAULT_SINGLE_VALUED_PREDICATES.len());
    }

    #[test]
    fn a_configured_list_replaces_the_default_and_can_be_empty() {
        let list = SingleValued::parse(" owns_car , Favourite Color,,");
        assert_eq!(
            list.names().collect::<Vec<_>>(),
            ["favourite_color", "owns_car"]
        );
        assert!(list.contains("ownsCar"));
        assert!(!list.contains("lives_in"));
        let off = SingleValued::parse(",");
        assert_eq!(off.names().count(), 0);
        assert!(!off.contains("lives_in"));
    }

    fn current(predicate: &str, statement: &str, object_id: Option<Uuid>) -> Fact {
        Fact {
            id: Uuid::now_v7(),
            scope: Scope::agent("a"),
            subject_id: Uuid::now_v7(),
            predicate: predicate.into(),
            object_id,
            statement: statement.into(),
            trust: Trust::Agent,
            status: FactStatus::Active,
            valid_from: Utc::now() - Duration::days(1),
            valid_to: None,
            expires_at: None,
            recorded_at: Utc::now(),
            invalidated_at: None,
            superseded_by: None,
            pending_supersede_of: None,
            source_episode_ids: Vec::new(),
        }
    }

    fn new_value<'a>(
        scope: &'a Scope,
        statement: &'a str,
        object_id: Option<Uuid>,
    ) -> NewValue<'a> {
        NewValue {
            scope,
            predicate: "livesIn",
            object_id,
            statement,
        }
    }

    #[test]
    fn the_latest_other_value_of_the_same_predicate_is_replaced() {
        let scope = Scope::agent("a");
        let paris = Uuid::now_v7();
        let facts = vec![
            current("lives_in", "Ada lives in Rome", None),
            current("likes", "Ada likes tea", None),
            current("Lives In", "Ada lives in Paris", Some(paris)),
        ];
        let found = replaced(
            &facts,
            &new_value(&scope, "Ada lives in Berlin", Some(Uuid::now_v7())),
            &|_| false,
            Utc::now(),
        );
        assert_eq!(found.map(|f| f.id), Some(facts[2].id));
    }

    #[test]
    fn the_same_value_or_a_fact_that_is_not_current_is_never_replaced() {
        let scope = Scope::agent("a");
        let paris = Uuid::now_v7();
        let same = current("lives_in", "Ada lives in Paris", Some(paris));
        let mut other_scope = current("lives_in", "Ada lives in Rome", None);
        other_scope.scope = Scope::agent("b");
        let mut held = current("lives_in", "Ada lives in Oslo", None);
        held.status = FactStatus::Quarantined;
        let mut closed = current("lives_in", "Ada lives in Lima", None);
        closed.superseded_by = Some(Uuid::now_v7());
        let mut retracted = current("lives_in", "Ada lives in Kyiv", None);
        retracted.invalidated_at = Some(Utc::now());
        let mut expired = current("lives_in", "Ada lives in Bern", None);
        expired.expires_at = Some(Utc::now() - Duration::minutes(1));
        let closing = current("lives_in", "Ada lives in Riga", None);
        let closing_id = closing.id;
        let facts = vec![other_scope, held, closed, retracted, expired, closing, same];
        let found = replaced(
            &facts,
            &new_value(&scope, "ada lives in paris.", Some(paris)),
            &|id| id == closing_id,
            Utc::now(),
        );
        assert!(found.is_none(), "{:?}", found.map(|f| &f.statement));
    }

    #[test]
    fn the_same_object_said_differently_is_a_new_value() {
        let scope = Scope::agent("a");
        let paris = Uuid::now_v7();
        let facts = vec![current("lives_in", "Ada lives in Paris", Some(paris))];
        let found = replaced(
            &facts,
            &new_value(&scope, "Ada lives in Paris, France", Some(paris)),
            &|_| false,
            Utc::now(),
        );
        assert_eq!(found.map(|f| f.id), Some(facts[0].id));
    }
}
