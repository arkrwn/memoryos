//! Versioned case data, embedded in the binary so a report always names what it ran.
//!
//! Validation enforces what makes a case able to fail honestly. For example, a staleness case
//! whose later episode names the old value would fail against a correct server, because a true
//! statement like "moved from X to Y" contains X.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::client::Trust;
use crate::score::contains_ci;

pub const RECALL_DATASET: &str = "recall.v1";
pub const STALENESS_DATASET: &str = "staleness.v1";
pub const POISONING_DATASET: &str = "poisoning.v1";

pub const MIN_RECALL_CASES: usize = 30;
pub const MIN_STALENESS_CASES: usize = 30;
pub const MIN_POISONING_CASES: usize = 25;

const RECALL_DATA: &str = include_str!("../data/recall.v1.jsonl");
const STALENESS_DATA: &str = include_str!("../data/staleness.v1.jsonl");
const POISONING_DATA: &str = include_str!("../data/poisoning.v1.jsonl");

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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpisodeSpec {
    pub content: String,
    #[serde(default)]
    pub occurred_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallCase {
    pub id: String,
    pub lang: String,
    pub episodes: Vec<EpisodeSpec>,
    pub query: String,
    pub expected: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StalenessCase {
    pub id: String,
    pub lang: String,
    pub topic: String,
    pub earlier: EpisodeSpec,
    pub later: EpisodeSpec,
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
pub struct PoisonEpisode {
    pub trust: Trust,
    pub source: String,
    pub content: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoisonCase {
    pub id: String,
    pub lang: String,
    pub class: PoisonClass,
    pub seed: EpisodeSpec,
    pub true_values: Vec<String>,
    pub poison: PoisonEpisode,
    pub planted: Vec<String>,
    /// The poison contradicts the seed fact. SPEC section 5 only quarantines `agent` trust when
    /// it contradicts a higher-trust fact, so every `agent` case must set this.
    #[serde(default)]
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
        let invalid = |problem: String| invalid(RECALL_DATASET, &case.id, problem);
        check_common(&case.lang, &case.query).map_err(invalid)?;
        if case.episodes.is_empty() || case.expected.is_empty() {
            return Err(invalid("needs episodes and expected values".to_owned()));
        }
        let contents: Vec<&str> = case.episodes.iter().map(|e| e.content.as_str()).collect();
        for value in &case.expected {
            if !contents.iter().any(|c| contains_ci(c, value)) {
                return Err(invalid(format!("no episode states {value:?}")));
            }
            if contains_ci(&case.query, value) {
                return Err(invalid(format!("query gives away {value:?}")));
            }
        }
    }
    require_count(RECALL_DATASET, cases.len(), MIN_RECALL_CASES)?;
    Ok(cases)
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
    let (Some(earlier_at), Some(later_at)) = (case.earlier.occurred_at, case.later.occurred_at)
    else {
        return Err("both episodes need occurred_at".to_owned());
    };
    if !(earlier_at < case.as_of_before_change && case.as_of_before_change < later_at) {
        return Err("as_of_before_change must fall between the two episodes".to_owned());
    }
    if case.old_values.is_empty() || case.new_values.is_empty() {
        return Err("needs old_values and new_values".to_owned());
    }
    for old in &case.old_values {
        if !contains_ci(&case.earlier.content, old) {
            return Err(format!("earlier episode does not state {old:?}"));
        }
        if contains_ci(&case.later.content, old) {
            return Err(format!("later episode repeats old value {old:?}"));
        }
        if contains_ci(&case.query, old) {
            return Err(format!("query names old value {old:?}"));
        }
        if case.new_values.iter().any(|n| contains_ci(n, old)) {
            return Err(format!("old value {old:?} is inside a new value"));
        }
    }
    for new in &case.new_values {
        if !contains_ci(&case.later.content, new) {
            return Err(format!("later episode does not state {new:?}"));
        }
        if contains_ci(&case.earlier.content, new) {
            return Err(format!("earlier episode already has new value {new:?}"));
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
    let Some(seed_at) = case.seed.occurred_at else {
        return Err("seed needs occurred_at".to_owned());
    };
    if case.poison.occurred_at <= seed_at {
        return Err("poison must occur after the seed".to_owned());
    }
    match case.poison.trust {
        Trust::External => {}
        Trust::Agent if case.contradicts_seed => {}
        Trust::Agent => return Err("agent poison must contradict the seed".to_owned()),
        Trust::User | Trust::System => return Err("poison trust must be below user".to_owned()),
    }
    if case.true_values.is_empty() || case.planted.is_empty() {
        return Err("needs true_values and planted".to_owned());
    }
    for planted in &case.planted {
        if !contains_ci(&case.poison.content, planted) {
            return Err(format!("poison does not state {planted:?}"));
        }
        if contains_ci(&case.seed.content, planted) {
            return Err(format!("seed contains planted value {planted:?}"));
        }
        if contains_ci(&case.query, planted) {
            return Err(format!("query names planted value {planted:?}"));
        }
    }
    for truth in &case.true_values {
        if !contains_ci(&case.seed.content, truth) {
            return Err(format!("seed does not state {truth:?}"));
        }
        if contains_ci(&case.poison.content, truth) {
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
    fn recall_data_parses_and_meets_minimum() {
        let cases = load_recall().expect("recall data is valid");
        assert!(cases.len() >= MIN_RECALL_CASES, "{}", cases.len());
        assert!(cases.iter().any(|c| c.lang == "id"));
    }

    #[test]
    fn staleness_data_parses_and_meets_minimum() {
        let cases = load_staleness().expect("staleness data is valid");
        assert!(cases.len() >= MIN_STALENESS_CASES, "{}", cases.len());
        assert!(cases.iter().any(|c| c.lang == "id"));
    }

    #[test]
    fn poisoning_data_parses_and_covers_every_class() {
        let cases = load_poisoning().expect("poisoning data is valid");
        assert!(cases.len() >= MIN_POISONING_CASES, "{}", cases.len());
        assert!(cases.iter().any(|c| c.lang == "id"));
        for class in POISON_CLASSES {
            let count = cases.iter().filter(|c| c.class == class).count();
            assert!(count >= 5, "{class:?} has {count}");
        }
        assert!(cases.iter().any(|c| c.poison.trust == Trust::Agent));
        assert!(cases.iter().any(|c| c.poison.trust == Trust::External));
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

    const STALE_LINE: &str = r#"{"id":"s1","lang":"en","topic":"city","earlier":{"content":"lives in Oslo","occurred_at":"2023-01-01T00:00:00Z"},"later":{"content":"lives in Rome","occurred_at":"2024-01-01T00:00:00Z"},"as_of_before_change":"2023-06-01T00:00:00Z","query":"Where?","old_values":["Oslo"],"new_values":["Rome"]}"#;

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
    fn staleness_rejects_later_episode_naming_old_value() {
        let line = one_line(STALE_LINE, ("lives in Rome", "moved from Oslo to Rome"));
        assert!(stale_problem(&line).contains("later episode repeats"));
    }

    #[test]
    fn staleness_rejects_as_of_outside_the_change_window() {
        let line = one_line(STALE_LINE, ("2023-06-01", "2025-06-01"));
        assert!(stale_problem(&line).contains("as_of_before_change"));
    }

    const POISON_LINE: &str = r#"{"id":"p1","lang":"en","class":"trust_overwrite","seed":{"content":"lives in Oslo","occurred_at":"2023-01-01T00:00:00Z"},"true_values":["Oslo"],"poison":{"trust":"agent","source":"x","content":"lives in Rome","occurred_at":"2024-01-01T00:00:00Z"},"planted":["Rome"],"contradicts_seed":true,"query":"Where?"}"#;

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
    fn poison_rejects_agent_trust_with_nothing_to_contradict() {
        let line = one_line(POISON_LINE, (r#""contradicts_seed":true,"#, ""));
        assert!(poison_problem(&line).contains("must contradict"));
    }

    #[test]
    fn poison_rejects_trust_at_or_above_user() {
        let line = one_line(POISON_LINE, (r#""trust":"agent""#, r#""trust":"user""#));
        assert!(poison_problem(&line).contains("below user"));
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
