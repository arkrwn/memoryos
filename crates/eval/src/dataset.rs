//! Versioned case data, embedded in the binary so a report always names what it ran.
//!
//! Each case is a script: the source text a host read, and the facts the host decided on. Validation
//! enforces what makes a case able to fail honestly. For example, a staleness case whose later
//! statement names the old value would fail against a correct server, because a true statement like
//! "moved from X to Y" contains X.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::client::Trust;
use crate::score::contains_ci;

pub const RECALL_DATASET: &str = "recall.v2";
pub const STALENESS_DATASET: &str = "staleness.v2";
pub const POISONING_DATASET: &str = "poisoning.v2";

pub const MIN_RECALL_CASES: usize = 30;
pub const MIN_STALENESS_CASES: usize = 30;
pub const MIN_POISONING_CASES: usize = 25;

const RECALL_DATA: &str = include_str!("../data/recall.v2.jsonl");
const STALENESS_DATA: &str = include_str!("../data/staleness.v2.jsonl");
const POISONING_DATA: &str = include_str!("../data/poisoning.v2.jsonl");

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("{dataset} line {line}: {source}")]
    Parse {
        dataset: String,
        line: usize,
        source: serde_json::Error,
    },
    #[error("{dataset}: case {case}: {problem}")]
    Invalid {
        dataset: String,
        case: String,
        problem: String,
    },
    #[error("{dataset}: has {found} cases, needs at least {needed}")]
    TooFew {
        dataset: String,
        found: usize,
        needed: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityRef {
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactSpec {
    pub statement: String,
    pub subject: EntityRef,
    pub predicate: String,
    #[serde(default)]
    pub valid_from: Option<DateTime<Utc>>,
}

/// A commit of several facts from one source text.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallCommit {
    pub content: String,
    pub facts: Vec<FactSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallCase {
    pub id: String,
    pub lang: String,
    pub commits: Vec<RecallCommit>,
    pub query: String,
    pub expected: Vec<String>,
}

/// A commit of one fact from one source text.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptedFact {
    pub content: String,
    pub fact: FactSpec,
}

/// Which kind of scope a case runs in. It decides whether `agent` trust goes active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopeKind {
    User,
    Shared,
    Agent,
}

impl ScopeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Shared => "shared",
            Self::Agent => "agent",
        }
    }

    /// SPEC section 5: `agent` trust is quarantined in user and shared scopes, active in its own.
    pub fn holds_agent_trust(self) -> bool {
        self != Self::Agent
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StalenessCase {
    pub id: String,
    pub lang: String,
    pub topic: String,
    pub scope_kind: ScopeKind,
    pub earlier: ScriptedFact,
    pub later: ScriptedFact,
    pub as_of_before_change: DateTime<Utc>,
    pub query: String,
    pub old_values: Vec<String>,
    pub new_values: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoisonClass {
    ExternalFalseFact,
    InstructionInjection,
    FabricatedExperience,
    SleeperClaim,
    TrustOverwrite,
}

pub const POISON_CLASSES: [PoisonClass; 5] = [
    PoisonClass::ExternalFalseFact,
    PoisonClass::InstructionInjection,
    PoisonClass::FabricatedExperience,
    PoisonClass::SleeperClaim,
    PoisonClass::TrustOverwrite,
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoisonCommit {
    pub trust: Trust,
    pub source: String,
    pub content: String,
    pub fact: FactSpec,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoisonCase {
    pub id: String,
    pub lang: String,
    pub class: PoisonClass,
    pub scope_kind: ScopeKind,
    pub seed: ScriptedFact,
    pub true_values: Vec<String>,
    pub poison: PoisonCommit,
    pub planted: Vec<String>,
    /// The poison names the seed in `supersedes`, found through `/v1/candidates`, instead of
    /// being added alongside it.
    pub contradicts_seed: bool,
    pub query: String,
    /// Valid-time points to probe as well as now, for claims set to activate later.
    #[serde(default)]
    pub as_of_probes: Vec<DateTime<Utc>>,
}

pub fn load_recall() -> Result<Vec<RecallCase>, DatasetError> {
    parse_recall(RECALL_DATA)
}

pub fn load_staleness() -> Result<Vec<StalenessCase>, DatasetError> {
    parse_staleness(STALENESS_DATA)
}

pub fn load_poisoning() -> Result<Vec<PoisonCase>, DatasetError> {
    parse_poisoning(POISONING_DATA)
}

fn parse_recall(data: &str) -> Result<Vec<RecallCase>, DatasetError> {
    let cases: Vec<RecallCase> = parse_lines(RECALL_DATASET, data)?;
    check_ids(RECALL_DATASET, cases.iter().map(|c| c.id.as_str()))?;
    for case in &cases {
        check_recall_case(case).map_err(|p| invalid(RECALL_DATASET, &case.id, p))?;
    }
    require_count(RECALL_DATASET, cases.len(), MIN_RECALL_CASES)?;
    Ok(cases)
}

fn check_recall_case(case: &RecallCase) -> Result<(), String> {
    check_common(&case.lang, &case.query)?;
    if case.commits.is_empty() || case.expected.is_empty() {
        return Err("needs commits and expected values".to_owned());
    }
    let statements: Vec<&str> = case
        .commits
        .iter()
        .flat_map(|c| &c.facts)
        .map(|f| f.statement.as_str())
        .collect();
    let contents: Vec<&str> = case.commits.iter().map(|c| c.content.as_str()).collect();
    for value in &case.expected {
        if value.trim().is_empty() {
            return Err("empty expected value".to_owned());
        }
        if !statements.iter().any(|s| contains_ci(s, value)) {
            return Err(format!("no statement states {value:?}"));
        }
        if !contents.iter().any(|c| contains_ci(c, value)) {
            return Err(format!("no source text states {value:?}"));
        }
        if contains_ci(&case.query, value) {
            return Err(format!("query gives away {value:?}"));
        }
    }
    Ok(())
}

fn parse_staleness(data: &str) -> Result<Vec<StalenessCase>, DatasetError> {
    let cases: Vec<StalenessCase> = parse_lines(STALENESS_DATASET, data)?;
    check_ids(STALENESS_DATASET, cases.iter().map(|c| c.id.as_str()))?;
    for case in &cases {
        check_staleness_case(case).map_err(|p| invalid(STALENESS_DATASET, &case.id, p))?;
    }
    require_count(STALENESS_DATASET, cases.len(), MIN_STALENESS_CASES)?;
    Ok(cases)
}

fn check_staleness_case(case: &StalenessCase) -> Result<(), String> {
    check_common(&case.lang, &case.query)?;
    let (Some(earlier_at), Some(later_at)) =
        (case.earlier.fact.valid_from, case.later.fact.valid_from)
    else {
        return Err("both facts need valid_from".to_owned());
    };
    if !(earlier_at < case.as_of_before_change && case.as_of_before_change < later_at) {
        return Err("as_of_before_change must fall between the two facts".to_owned());
    }
    if case.old_values.is_empty() || case.new_values.is_empty() {
        return Err("needs old_values and new_values".to_owned());
    }
    let earlier = [&case.earlier.content, &case.earlier.fact.statement];
    let later = [&case.later.content, &case.later.fact.statement];
    for old in &case.old_values {
        if !earlier.iter().all(|text| contains_ci(text, old)) {
            return Err(format!("earlier commit does not state {old:?}"));
        }
        if later.iter().any(|text| contains_ci(text, old)) {
            return Err(format!("later commit repeats old value {old:?}"));
        }
        if contains_ci(&case.query, old) {
            return Err(format!("query names old value {old:?}"));
        }
        if case.new_values.iter().any(|n| contains_ci(n, old)) {
            return Err(format!("old value {old:?} is inside a new value"));
        }
    }
    for new in &case.new_values {
        if !later.iter().all(|text| contains_ci(text, new)) {
            return Err(format!("later commit does not state {new:?}"));
        }
        if earlier.iter().any(|text| contains_ci(text, new)) {
            return Err(format!("earlier commit already has new value {new:?}"));
        }
        if contains_ci(&case.query, new) {
            return Err(format!("query names new value {new:?}"));
        }
        if case.old_values.iter().any(|o| contains_ci(o, new)) {
            return Err(format!("new value {new:?} is inside an old value"));
        }
    }
    Ok(())
}

fn parse_poisoning(data: &str) -> Result<Vec<PoisonCase>, DatasetError> {
    let cases: Vec<PoisonCase> = parse_lines(POISONING_DATASET, data)?;
    check_ids(POISONING_DATASET, cases.iter().map(|c| c.id.as_str()))?;
    for case in &cases {
        check_poison_case(case).map_err(|p| invalid(POISONING_DATASET, &case.id, p))?;
    }
    for class in POISON_CLASSES {
        if !cases.iter().any(|c| c.class == class) {
            return Err(invalid(
                POISONING_DATASET,
                "*",
                format!("no case covers class {class:?}"),
            ));
        }
    }
    require_count(POISONING_DATASET, cases.len(), MIN_POISONING_CASES)?;
    Ok(cases)
}

fn check_poison_case(case: &PoisonCase) -> Result<(), String> {
    check_common(&case.lang, &case.query)?;
    let (Some(seed_at), Some(poison_at)) = (case.seed.fact.valid_from, case.poison.fact.valid_from)
    else {
        return Err("seed and poison need valid_from".to_owned());
    };
    if poison_at <= seed_at {
        return Err("poison must start after the seed".to_owned());
    }
    if case.as_of_probes.iter().any(|probe| *probe < poison_at) {
        return Err("an as_of probe is before the poison starts".to_owned());
    }
    match (case.poison.trust, case.scope_kind) {
        (Trust::External, _) => {}
        (Trust::Agent, kind) if kind.holds_agent_trust() => {}
        (Trust::Agent, _) => {
            return Err("agent poison needs a user or shared scope to be held".to_owned())
        }
        (Trust::User | Trust::System, _) => {
            return Err("poison trust must be below user".to_owned())
        }
    }
    if case.true_values.is_empty() || case.planted.is_empty() {
        return Err("needs true_values and planted".to_owned());
    }
    let seed = [&case.seed.content, &case.seed.fact.statement];
    let poison = [&case.poison.content, &case.poison.fact.statement];
    for planted in &case.planted {
        if !poison.iter().all(|text| contains_ci(text, planted)) {
            return Err(format!("poison does not state {planted:?}"));
        }
        if seed.iter().any(|text| contains_ci(text, planted)) {
            return Err(format!("seed contains planted value {planted:?}"));
        }
        if contains_ci(&case.query, planted) {
            return Err(format!("query names planted value {planted:?}"));
        }
    }
    for truth in &case.true_values {
        if !seed.iter().all(|text| contains_ci(text, truth)) {
            return Err(format!("seed does not state {truth:?}"));
        }
        if poison.iter().any(|text| contains_ci(text, truth)) {
            return Err(format!("poison names true value {truth:?}"));
        }
        if contains_ci(&case.query, truth) {
            return Err(format!("query names true value {truth:?}"));
        }
    }
    Ok(())
}

fn parse_lines<T: DeserializeOwned>(dataset: &str, data: &str) -> Result<Vec<T>, DatasetError> {
    data.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|source| DatasetError::Parse {
                dataset: dataset.to_owned(),
                line: index + 1,
                source,
            })
        })
        .collect()
}

/// Case ids become part of a scope id, which allows only a limited character set.
fn check_ids<'a>(dataset: &str, ids: impl Iterator<Item = &'a str>) -> Result<(), DatasetError> {
    let mut seen = HashSet::new();
    for id in ids {
        let valid = !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
        if !valid {
            return Err(invalid(dataset, id, "id must be [a-z0-9]+".to_owned()));
        }
        if !seen.insert(id) {
            return Err(invalid(dataset, id, "duplicate id".to_owned()));
        }
    }
    Ok(())
}

fn check_common(lang: &str, query: &str) -> Result<(), String> {
    if !matches!(lang, "en" | "id") {
        return Err(format!("unknown lang {lang:?}"));
    }
    if query.trim().is_empty() {
        return Err("empty query".to_owned());
    }
    Ok(())
}

fn require_count(dataset: &str, found: usize, needed: usize) -> Result<(), DatasetError> {
    if found < needed {
        return Err(DatasetError::TooFew {
            dataset: dataset.to_owned(),
            found,
            needed,
        });
    }
    Ok(())
}

fn invalid(dataset: &str, case: &str, problem: String) -> DatasetError {
    DatasetError::Invalid {
        dataset: dataset.to_owned(),
        case: case.to_owned(),
        problem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_data_parses_and_keeps_its_case_count() {
        let cases = load_recall().expect("recall data is valid");
        assert_eq!(cases.len(), 34);
        assert!(cases.iter().any(|c| c.lang == "id"));
    }

    #[test]
    fn staleness_data_parses_and_keeps_its_case_count() {
        let cases = load_staleness().expect("staleness data is valid");
        assert_eq!(cases.len(), 32);
        assert!(cases.iter().any(|c| c.lang == "id"));
        assert!(cases.iter().any(|c| c.scope_kind.holds_agent_trust()));
        assert!(cases.iter().any(|c| !c.scope_kind.holds_agent_trust()));
    }

    #[test]
    fn poisoning_data_parses_and_covers_every_class() {
        let cases = load_poisoning().expect("poisoning data is valid");
        assert_eq!(cases.len(), 30);
        assert!(cases.iter().any(|c| c.lang == "id"));
        for class in POISON_CLASSES {
            let count = cases.iter().filter(|c| c.class == class).count();
            assert!(count >= 5, "{class:?} has {count}");
        }
        assert!(cases.iter().any(|c| c.poison.trust == Trust::Agent));
        assert!(cases.iter().any(|c| c.poison.trust == Trust::External));
        for kind in [ScopeKind::User, ScopeKind::Shared, ScopeKind::Agent] {
            assert!(cases.iter().any(|c| c.scope_kind == kind), "{kind:?}");
        }
        assert!(cases.iter().any(|c| c.contradicts_seed));
        assert!(cases.iter().any(|c| !c.contradicts_seed));
    }

    #[test]
    fn sleeper_cases_probe_a_future_date() {
        let cases = load_poisoning().expect("poisoning data is valid");
        for case in cases
            .iter()
            .filter(|c| c.class == PoisonClass::SleeperClaim)
        {
            assert!(!case.as_of_probes.is_empty(), "{}", case.id);
        }
    }

    #[test]
    fn scope_ids_fit_the_scope_format() {
        // Scope ids allow at most 128 chars of [A-Za-z0-9_.-]. See core `Scope::from_str`.
        let longest = load_poisoning()
            .expect("valid")
            .iter()
            .map(|c| c.id.len())
            .max()
            .expect("non-empty");
        assert!("eval-".len() + 12 + 1 + longest <= 128);
    }

    fn one_line(line: &str, replace: (&str, &str)) -> String {
        let replaced = line.replace(replace.0, replace.1);
        assert_ne!(replaced, line, "fixture replacement did not apply");
        replaced
    }

    const STALE_LINE: &str = r#"{"id":"s1","lang":"en","topic":"city","scope_kind":"agent","earlier":{"content":"lives in Oslo","fact":{"statement":"lives in Oslo","subject":{"name":"a","kind":"person"},"predicate":"city","valid_from":"2023-01-01T00:00:00Z"}},"later":{"content":"lives in Rome","fact":{"statement":"lives in Rome","subject":{"name":"a","kind":"person"},"predicate":"city","valid_from":"2024-01-01T00:00:00Z"}},"as_of_before_change":"2023-06-01T00:00:00Z","query":"Where?","old_values":["Oslo"],"new_values":["Rome"]}"#;

    fn stale_problem(line: &str) -> String {
        let case: StalenessCase = serde_json::from_str(line).expect("fixture parses");
        check_staleness_case(&case).expect_err("should be rejected")
    }

    #[test]
    fn staleness_fixture_is_valid() {
        let case: StalenessCase = serde_json::from_str(STALE_LINE).expect("fixture parses");
        check_staleness_case(&case).expect("fixture is valid");
    }

    #[test]
    fn staleness_rejects_later_commit_naming_old_value() {
        let line = one_line(STALE_LINE, ("lives in Rome", "moved from Oslo to Rome"));
        assert!(stale_problem(&line).contains("later commit repeats"));
    }

    #[test]
    fn staleness_rejects_as_of_outside_the_change_window() {
        let line = one_line(STALE_LINE, ("2023-06-01", "2025-06-01"));
        assert!(stale_problem(&line).contains("as_of_before_change"));
    }

    const POISON_LINE: &str = r#"{"id":"p1","lang":"en","class":"trust_overwrite","scope_kind":"user","seed":{"content":"lives in Oslo","fact":{"statement":"lives in Oslo","subject":{"name":"a","kind":"person"},"predicate":"city","valid_from":"2023-01-01T00:00:00Z"}},"true_values":["Oslo"],"poison":{"trust":"agent","source":"x","content":"lives in Rome","fact":{"statement":"lives in Rome","subject":{"name":"a","kind":"person"},"predicate":"city","valid_from":"2024-01-01T00:00:00Z"}},"planted":["Rome"],"contradicts_seed":true,"query":"Where?"}"#;

    fn poison_problem(line: &str) -> String {
        let case: PoisonCase = serde_json::from_str(line).expect("fixture parses");
        check_poison_case(&case).expect_err("should be rejected")
    }

    #[test]
    fn poison_fixture_is_valid() {
        let case: PoisonCase = serde_json::from_str(POISON_LINE).expect("fixture parses");
        check_poison_case(&case).expect("fixture is valid");
    }

    #[test]
    fn poison_rejects_agent_trust_in_its_own_scope() {
        let line = one_line(
            POISON_LINE,
            (r#""scope_kind":"user""#, r#""scope_kind":"agent""#),
        );
        assert!(poison_problem(&line).contains("user or shared"));
    }

    #[test]
    fn poison_rejects_trust_at_or_above_user() {
        let line = one_line(POISON_LINE, (r#""trust":"agent""#, r#""trust":"user""#));
        assert!(poison_problem(&line).contains("below user"));
    }

    #[test]
    fn poison_rejects_a_statement_missing_the_planted_value() {
        let line = one_line(
            POISON_LINE,
            (r#""statement":"lives in Rome""#, r#""statement":"moved""#),
        );
        assert!(poison_problem(&line).contains("does not state"));
    }

    #[test]
    fn recall_rejects_a_value_no_statement_states() {
        let line = r#"{"id":"r1","lang":"en","commits":[{"content":"lives in Oslo","facts":[{"statement":"has a home","subject":{"name":"a","kind":"person"},"predicate":"city"}]}],"query":"Where?","expected":["Oslo"]}"#;
        let case: RecallCase = serde_json::from_str(line).expect("fixture parses");
        let problem = check_recall_case(&case).expect_err("should be rejected");
        assert!(problem.contains("no statement"), "{problem}");
    }

    #[test]
    fn parse_reports_the_failing_line() {
        let err = parse_lines::<RecallCase>("x", "\n{not json}\n").expect_err("bad json");
        assert!(err.to_string().contains("line 2"), "{err}");
    }

    #[test]
    fn parse_rejects_unknown_fields() {
        let line = one_line(POISON_LINE, (r#""query""#, r#""typo":1,"query""#));
        assert!(serde_json::from_str::<PoisonCase>(&line).is_err());
    }

    #[test]
    fn duplicate_and_malformed_ids_are_rejected() {
        assert!(check_ids("x", ["a1", "a1"].into_iter()).is_err());
        assert!(check_ids("x", ["A b"].into_iter()).is_err());
        assert!(check_ids("x", ["a1", "b2"].into_iter()).is_ok());
    }

    #[test]
    fn too_few_cases_are_rejected() {
        assert!(matches!(
            parse_recall(""),
            Err(DatasetError::TooFew { found: 0, .. })
        ));
    }
}
