//! The three suites. Each case runs in its own fresh scope, so runs and cases cannot interfere.
//!
//! A case body records what it observes into a detail struct as it goes, and any detail recorded
//! before a timeout or HTTP failure is kept, so a leak seen early is still reported. The detail
//! fields default to "no credit": nothing counts as good until a check has actually run.

use std::future::Future;
use std::time::Instant;

use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use serde::Serialize;
use serde_json::json;
use uuid::Uuid;

use crate::client::{
    CaseError, Client, Episode, Fact, NewEpisode, PurgeReport, RecallResponse, Trust,
};
use crate::dataset::{EpisodeSpec, PoisonCase, RecallCase, StalenessCase};
use crate::score::{contains_ci, find_leaks, rank_of_set, Stage, TOP_K};

/// Requested so that MRR can see answers ranked just below `TOP_K`.
const RECALL_LIMIT: u32 = 10;
/// The API maximum. A stale or planted fact counts anywhere in the results, not only near the top.
const PROBE_LIMIT: u32 = 100;
const USER_SOURCE: &str = "chat";

pub struct RunContext {
    pub client: Client,
    pub run_id: String,
}

impl RunContext {
    pub fn new(client: Client) -> Self {
        let mut run_id = Uuid::new_v4().simple().to_string();
        run_id.truncate(12);
        Self { client, run_id }
    }

    fn scope(&self, case_id: &str) -> String {
        format!("user:eval-{}-{case_id}", self.run_id)
    }

    /// Writes one episode. The idempotency key is unique per run, case and `label`.
    async fn post_episode(
        &self,
        (case_id, label): (&str, &str),
        scope: &str,
        (source, trust): (&str, Trust),
        content: &str,
        occurred_at: Option<DateTime<Utc>>,
    ) -> Result<Episode, CaseError> {
        self.client
            .post_episode(&NewEpisode {
                scope,
                source,
                trust,
                content,
                occurred_at,
                metadata: json!({ "eval_run": self.run_id, "eval_case": case_id }),
                idempotency_key: format!("eval-{}-{case_id}-{label}", self.run_id),
            })
            .await
    }

    async fn write_user_episode(
        &self,
        case_id: &str,
        label: &str,
        scope: &str,
        spec: &EpisodeSpec,
    ) -> Result<Episode, CaseError> {
        let created = self
            .post_episode(
                (case_id, label),
                scope,
                (USER_SOURCE, Trust::User),
                &spec.content,
                spec.occurred_at,
            )
            .await?;
        self.client.wait_processed(created.id).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CaseStatus {
    /// The server gave the right answer.
    Passed,
    /// The server gave a wrong answer.
    Failed,
    /// No verdict: timeout or HTTP failure. Fails the run and never counts as a pass.
    Error,
}

#[derive(Debug, Serialize)]
pub struct CaseResult<D> {
    pub id: String,
    pub lang: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    pub scope: String,
    pub status: CaseStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub duration_ms: u128,
    pub detail: D,
}

/// Why a finished case is wrong. Empty means it passed.
pub trait Verdict {
    fn problems(&self) -> Vec<String>;
}

struct CaseMeta<'a> {
    id: &'a str,
    lang: &'a str,
    class: Option<String>,
    scope: String,
    started: Instant,
}

fn finish<D: Verdict>(
    meta: CaseMeta<'_>,
    outcome: Result<(), CaseError>,
    detail: D,
) -> CaseResult<D> {
    let (status, reason) = match outcome {
        Err(e) => (CaseStatus::Error, Some(e.to_string())),
        Ok(()) => {
            let problems = detail.problems();
            if problems.is_empty() {
                (CaseStatus::Passed, None)
            } else {
                (CaseStatus::Failed, Some(problems.join("; ")))
            }
        }
    };
    CaseResult {
        id: meta.id.to_owned(),
        lang: meta.lang.to_owned(),
        class: meta.class,
        scope: meta.scope,
        status,
        reason,
        duration_ms: meta.started.elapsed().as_millis(),
        detail,
    }
}

/// Runs cases with bounded concurrency and returns results in dataset order.
async fn run_all<'a, C, D, F, Fut>(
    cases: &'a [C],
    concurrency: usize,
    run_case: F,
) -> Vec<CaseResult<D>>
where
    F: Fn(&'a C) -> Fut,
    Fut: Future<Output = CaseResult<D>>,
{
    let in_flight = cases.iter().enumerate().map(|(index, case)| {
        let future = run_case(case);
        async move { (index, future.await) }
    });
    let mut results: Vec<(usize, CaseResult<D>)> = stream::iter(in_flight)
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

// Recall

#[derive(Debug, Default, Serialize)]
pub struct RecallDetail {
    /// 1-based rank at which the top statements first cover every expected value.
    pub rank: Option<usize>,
    pub missing: Vec<String>,
    pub top_statements: Vec<String>,
}

impl Verdict for RecallDetail {
    fn problems(&self) -> Vec<String> {
        match self.rank {
            Some(rank) if rank <= TOP_K => Vec::new(),
            Some(rank) => vec![format!("expected values first covered at rank {rank}")],
            None => vec![format!(
                "expected values not recalled: {}",
                self.missing.join(", ")
            )],
        }
    }
}

pub async fn run_recall(
    ctx: &RunContext,
    cases: &[RecallCase],
    concurrency: usize,
) -> Vec<CaseResult<RecallDetail>> {
    run_all(cases, concurrency, |case| async move {
        let meta = CaseMeta {
            id: &case.id,
            lang: &case.lang,
            class: None,
            scope: ctx.scope(&case.id),
            started: Instant::now(),
        };
        let mut detail = RecallDetail::default();
        let outcome = recall_steps(ctx, case, &meta.scope, &mut detail).await;
        finish(meta, outcome, detail)
    })
    .await
}

async fn recall_steps(
    ctx: &RunContext,
    case: &RecallCase,
    scope: &str,
    detail: &mut RecallDetail,
) -> Result<(), CaseError> {
    for (index, spec) in case.episodes.iter().enumerate() {
        ctx.write_user_episode(&case.id, &index.to_string(), scope, spec)
            .await?;
    }
    let response = ctx
        .client
        .recall(scope, &case.query, RECALL_LIMIT, None)
        .await?;
    let statements = statements(&response);
    detail.rank = rank_of_set(&statements, &case.expected);
    detail.missing = case
        .expected
        .iter()
        .filter(|value| !statements.iter().any(|s| contains_ci(s, value)))
        .cloned()
        .collect();
    detail.top_statements = statements.into_iter().take(TOP_K).collect();
    Ok(())
}

// Staleness

#[derive(Debug, Default, Serialize)]
pub struct StalenessDetail {
    pub new_in_current: bool,
    pub old_absent_from_current: bool,
    pub old_in_as_of: bool,
    pub new_absent_from_as_of: bool,
    pub current_statements: Vec<String>,
    pub as_of_statements: Vec<String>,
}

impl Verdict for StalenessDetail {
    fn problems(&self) -> Vec<String> {
        [
            (self.new_in_current, "new value missing from current recall"),
            (
                self.old_absent_from_current,
                "old value still in current recall",
            ),
            (self.old_in_as_of, "old value missing from as_of recall"),
            (
                self.new_absent_from_as_of,
                "new value leaked into as_of recall",
            ),
        ]
        .into_iter()
        .filter(|(ok, _)| !ok)
        .map(|(_, problem)| problem.to_owned())
        .collect()
    }
}

pub async fn run_staleness(
    ctx: &RunContext,
    cases: &[StalenessCase],
    concurrency: usize,
) -> Vec<CaseResult<StalenessDetail>> {
    run_all(cases, concurrency, |case| async move {
        let meta = CaseMeta {
            id: &case.id,
            lang: &case.lang,
            class: Some(case.topic.clone()),
            scope: ctx.scope(&case.id),
            started: Instant::now(),
        };
        let mut detail = StalenessDetail::default();
        let outcome = staleness_steps(ctx, case, &meta.scope, &mut detail).await;
        finish(meta, outcome, detail)
    })
    .await
}

async fn staleness_steps(
    ctx: &RunContext,
    case: &StalenessCase,
    scope: &str,
    detail: &mut StalenessDetail,
) -> Result<(), CaseError> {
    // In order, and each fully processed, so the later episode reconciles against the earlier.
    ctx.write_user_episode(&case.id, "earlier", scope, &case.earlier)
        .await?;
    ctx.write_user_episode(&case.id, "later", scope, &case.later)
        .await?;

    let current = statements(
        &ctx.client
            .recall(scope, &case.query, PROBE_LIMIT, None)
            .await?,
    );
    detail.new_in_current = covers_all(&current, &case.new_values);
    detail.old_absent_from_current = covers_none(&current, &case.old_values);
    detail.current_statements = current;

    let past = statements(
        &ctx.client
            .recall(
                scope,
                &case.query,
                PROBE_LIMIT,
                Some(case.as_of_before_change),
            )
            .await?,
    );
    detail.old_in_as_of = covers_all(&past, &case.old_values);
    detail.new_absent_from_as_of = covers_none(&past, &case.new_values);
    detail.as_of_statements = past;
    Ok(())
}

// Poisoning

#[derive(Debug, Default, Serialize)]
pub struct PoisoningDetail {
    /// Planted content reachable through recall. Any entry fails the zero-leak gate.
    pub leaks: Vec<String>,
    pub true_fact_problems: Vec<String>,
    pub purge_problems: Vec<String>,
    /// Quarantined facts traced to the poison. Zero is legitimate when extraction dropped it.
    pub quarantined_found: usize,
    pub purge_completed: bool,
    pub purge_report: Option<PurgeReport>,
}

impl Verdict for PoisoningDetail {
    fn problems(&self) -> Vec<String> {
        let mut problems = self.leaks.clone();
        problems.extend(self.true_fact_problems.iter().cloned());
        problems.extend(self.purge_problems.iter().cloned());
        problems
    }
}

pub async fn run_poisoning(
    ctx: &RunContext,
    cases: &[PoisonCase],
    concurrency: usize,
) -> Vec<CaseResult<PoisoningDetail>> {
    run_all(cases, concurrency, |case| async move {
        let meta = CaseMeta {
            id: &case.id,
            lang: &case.lang,
            class: serde_json::to_value(case.class)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned)),
            scope: ctx.scope(&case.id),
            started: Instant::now(),
        };
        let mut detail = PoisoningDetail::default();
        let outcome = poisoning_steps(ctx, case, &meta.scope, &mut detail).await;
        finish(meta, outcome, detail)
    })
    .await
}

async fn poisoning_steps(
    ctx: &RunContext,
    case: &PoisonCase,
    scope: &str,
    detail: &mut PoisoningDetail,
) -> Result<(), CaseError> {
    let seed = ctx
        .write_user_episode(&case.id, "seed", scope, &case.seed)
        .await?;

    let poison = ctx
        .post_episode(
            (&case.id, "poison"),
            scope,
            (&case.poison.source, case.poison.trust),
            &case.poison.content,
            Some(case.poison.occurred_at),
        )
        .await?;

    // Before the worker has finished: SPEC section 3 hides external episodes from recall even
    // while pending, and no trust level may reach facts yet.
    let response = probe(ctx, scope, case, None).await?;
    detail.leaks.extend(find_leaks(
        &response,
        &case.planted,
        poison.id,
        case.poison.trust,
        Stage::BeforeProcessing,
    ));

    ctx.client.wait_processed(poison.id).await?;

    let response = probe(ctx, scope, case, None).await?;
    detail.leaks.extend(find_leaks(
        &response,
        &case.planted,
        poison.id,
        case.poison.trust,
        Stage::AfterProcessing,
    ));
    detail.true_fact_problems.extend(true_fact_problems(
        &response,
        case,
        seed.id,
        "after processing",
    ));

    for &as_of in &case.as_of_probes {
        let response = probe(ctx, scope, case, Some(as_of)).await?;
        detail.leaks.extend(find_leaks(
            &response,
            &case.planted,
            poison.id,
            case.poison.trust,
            Stage::AsOf(as_of),
        ));
    }

    purge_and_verify(ctx, case, scope, seed.id, poison.id, detail).await
}

async fn purge_and_verify(
    ctx: &RunContext,
    case: &PoisonCase,
    scope: &str,
    seed_id: Uuid,
    poison_id: Uuid,
    detail: &mut PoisoningDetail,
) -> Result<(), CaseError> {
    let quarantined = ctx.client.quarantine(scope).await?;
    let planted_facts: Vec<&Fact> = quarantined
        .iter()
        .filter(|fact| traces_to_poison(fact, case, poison_id))
        .collect();
    detail.quarantined_found = planted_facts.len();

    let report = ctx.client.purge_episode(poison_id).await?;
    if report.episodes_deleted < 1 {
        detail
            .purge_problems
            .push("purge reported no deleted episode".to_owned());
    }
    if (report.facts_deleted as usize) < planted_facts.len() {
        detail.purge_problems.push(format!(
            "purge deleted {} facts, {} quarantined facts came from the poison",
            report.facts_deleted,
            planted_facts.len()
        ));
    }
    detail.purge_report = Some(report);

    if ctx.client.episode_exists(poison_id).await? {
        detail
            .purge_problems
            .push(format!("episode {poison_id} still readable after purge"));
    }
    for fact in &planted_facts {
        if ctx.client.fact_exists(fact.id).await? {
            detail
                .purge_problems
                .push(format!("fact {} still readable after purge", fact.id));
        }
    }
    let remaining = ctx.client.quarantine(scope).await?;
    let still_listed = remaining
        .iter()
        .filter(|fact| traces_to_poison(fact, case, poison_id))
        .count();
    if still_listed > 0 {
        detail.purge_problems.push(format!(
            "{still_listed} planted facts still quarantined after purge"
        ));
    }

    let response = probe(ctx, scope, case, None).await?;
    detail.leaks.extend(find_leaks(
        &response,
        &case.planted,
        poison_id,
        case.poison.trust,
        Stage::AfterPurge,
    ));
    detail
        .true_fact_problems
        .extend(true_fact_problems(&response, case, seed_id, "after purge"));
    detail.purge_completed = true;
    Ok(())
}

async fn probe(
    ctx: &RunContext,
    scope: &str,
    case: &PoisonCase,
    as_of: Option<DateTime<Utc>>,
) -> Result<RecallResponse, CaseError> {
    ctx.client
        .recall(scope, &case.query, PROBE_LIMIT, as_of)
        .await
}

fn traces_to_poison(fact: &Fact, case: &PoisonCase, poison_id: Uuid) -> bool {
    fact.source_episode_ids.contains(&poison_id)
        || case.planted.iter().any(|p| contains_ci(&fact.statement, p))
}

/// The true fact must be recalled, active, and still derive from the seed episode, so a poison
/// fact that merely repeats the value does not count.
fn true_fact_problems(
    response: &RecallResponse,
    case: &PoisonCase,
    seed_id: Uuid,
    stage: &str,
) -> Vec<String> {
    case.true_values
        .iter()
        .filter(|value| {
            !response.facts.iter().any(|scored| {
                scored.fact.status == "active"
                    && scored.fact.source_episode_ids.contains(&seed_id)
                    && contains_ci(&scored.fact.statement, value)
            })
        })
        .map(|value| format!("true value {value:?} missing from recall {stage}"))
        .collect()
}

fn statements(response: &RecallResponse) -> Vec<String> {
    response
        .facts
        .iter()
        .map(|scored| scored.fact.statement.clone())
        .collect()
}

fn covers_all(statements: &[String], values: &[String]) -> bool {
    values
        .iter()
        .all(|value| statements.iter().any(|s| contains_ci(s, value)))
}

fn covers_none(statements: &[String], values: &[String]) -> bool {
    !values
        .iter()
        .any(|value| statements.iter().any(|s| contains_ci(s, value)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staleness_detail_gives_no_credit_by_default() {
        assert_eq!(StalenessDetail::default().problems().len(), 4);
    }

    #[test]
    fn recall_detail_fails_without_a_rank() {
        assert_eq!(RecallDetail::default().problems().len(), 1);
    }

    #[test]
    fn recall_detail_fails_beyond_top_k() {
        let detail = RecallDetail {
            rank: Some(TOP_K + 1),
            ..RecallDetail::default()
        };
        assert_eq!(detail.problems().len(), 1);
    }

    fn meta() -> CaseMeta<'static> {
        CaseMeta {
            id: "x",
            lang: "en",
            class: None,
            scope: "user:x".to_owned(),
            started: Instant::now(),
        }
    }

    #[test]
    fn an_error_outcome_is_never_a_pass_even_with_good_detail() {
        let good = RecallDetail {
            rank: Some(1),
            ..RecallDetail::default()
        };
        let error = CaseError::NotProcessed {
            episode: Uuid::nil(),
            secs: 1,
        };
        let result = finish(meta(), Err(error), good);
        assert_eq!(result.status, CaseStatus::Error);
        assert!(result.reason.expect("reason").contains("timeout"));
    }

    #[test]
    fn a_wrong_answer_is_failed_not_error() {
        let result = finish(meta(), Ok(()), RecallDetail::default());
        assert_eq!(result.status, CaseStatus::Failed);
    }

    #[test]
    fn a_right_answer_passes() {
        let good = RecallDetail {
            rank: Some(TOP_K),
            ..RecallDetail::default()
        };
        assert_eq!(finish(meta(), Ok(()), good).status, CaseStatus::Passed);
    }

    #[test]
    fn covers_helpers_are_case_insensitive() {
        let statements = vec!["Lives in DENVER".to_owned()];
        assert!(covers_all(&statements, &["denver".to_owned()]));
        assert!(!covers_none(&statements, &["denver".to_owned()]));
        assert!(covers_none(&statements, &["portland".to_owned()]));
    }

    #[tokio::test]
    async fn run_all_keeps_dataset_order_and_bounds_concurrency() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let cases: Vec<u64> = (0..12).collect();
        let running = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let results = run_all(&cases, 3, |case| {
            let (running, peak) = (&running, &peak);
            async move {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                // Later cases finish first, so ordering has to come from the sort.
                tokio::time::sleep(std::time::Duration::from_millis(12 - case)).await;
                running.fetch_sub(1, Ordering::SeqCst);
                finish(
                    CaseMeta {
                        id: "x",
                        lang: "en",
                        class: Some(case.to_string()),
                        scope: String::new(),
                        started: Instant::now(),
                    },
                    Ok(()),
                    RecallDetail::default(),
                )
            }
        })
        .await;
        let order: Vec<String> = results.into_iter().filter_map(|r| r.class).collect();
        let expected: Vec<String> = (0..12).map(|n: u64| n.to_string()).collect();
        assert_eq!(order, expected);
        assert!(peak.load(Ordering::SeqCst) <= 3);
    }
}
