//! The three suites, each a scripted host: it reads candidates, decides, and commits.
//!
//! Staleness and poisoning cases each run in their own fresh scope, so they cannot interfere. Recall
//! cases share one scope, because in a scope holding five facts every fact is in the top five.
//!
//! A case body records what it observes into a detail struct as it goes, and any detail recorded
//! before an HTTP failure is kept, so a leak seen early is still reported. The detail fields default
//! to "no credit": nothing counts as good until a check has actually run.

use std::collections::HashSet;
use std::future::Future;
use std::time::Instant;

use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use serde::Serialize;
use uuid::Uuid;

use crate::client::{
    CaseError, Client, CommitFact, CommitRequest, CommitResponse, CommittedFact, Fact, PurgeReport,
    RecallResponse, Trust,
};
use crate::dataset::{PoisonCase, RecallCase, ScopeKind, StalenessCase};
use crate::score::{contains_ci, find_leaks, rank_of_set, Stage, TOP_K};

/// Requested so that MRR can see answers ranked just below `TOP_K`.
const RECALL_LIMIT: u32 = 10;
/// The API maximum. A stale or planted fact counts anywhere in the results, not only near the top.
const PROBE_LIMIT: u32 = 100;
/// The API default for candidates, which is what a host would ask for.
const CANDIDATE_LIMIT: u32 = 10;
const HOST_SOURCE: &str = "eval-host";
const ACTIVE: &str = "active";
const QUARANTINED: &str = "quarantined";

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

    fn scope(&self, kind: ScopeKind, name: &str) -> String {
        format!("{}:eval-{}-{name}", kind.as_str(), self.run_id)
    }

    /// One commit. The idempotency key is unique per run, case and `label`.
    async fn commit(
        &self,
        trust: Trust,
        (case_id, label): (&str, &str),
        (scope, source, content): (&str, &str, &str),
        facts: Vec<CommitFact<'_>>,
    ) -> Result<CommitResponse, CaseError> {
        self.client
            .commit(
                trust,
                &CommitRequest {
                    scope,
                    source,
                    content,
                    idempotency_key: format!("eval-{}-{case_id}-{label}", self.run_id),
                    facts,
                },
            )
            .await
    }

    /// Commits as the host and gets the facts to `active`, approving them where the scope holds
    /// `agent` trust for review. A status the trust rules do not predict is a problem to report.
    async fn establish(
        &self,
        ids: (&str, &str),
        kind: ScopeKind,
        (scope, content): (&str, &str),
        facts: Vec<CommitFact<'_>>,
        problems: &mut Vec<String>,
    ) -> Result<CommitResponse, CaseError> {
        let committed = self
            .commit(Trust::Agent, ids, (scope, HOST_SOURCE, content), facts)
            .await?;
        let expected = agent_commit_status(kind);
        for fact in &committed.facts {
            check_status(fact, expected, ids.1, problems);
            if fact.status == QUARANTINED {
                self.client.approve(fact.id).await?;
            }
        }
        Ok(committed)
    }
}

fn agent_commit_status(kind: ScopeKind) -> &'static str {
    if kind.holds_agent_trust() {
        QUARANTINED
    } else {
        ACTIVE
    }
}

fn check_status(fact: &CommittedFact, expected: &str, label: &str, problems: &mut Vec<String>) {
    if fact.status != expected {
        problems.push(format!(
            "{label} commit: fact {} has status {}, expected {expected}",
            fact.id, fact.status
        ));
    }
}

/// The one fact of a one-fact commit. `Client::commit` has already checked the length.
fn only_fact(committed: &CommitResponse) -> Result<&CommittedFact, CaseError> {
    committed
        .facts
        .first()
        .ok_or_else(|| CaseError::Earlier("commit returned no facts".to_owned()))
}

/// The fact a host would choose from candidates: the one stating every given value.
fn choose_candidate(response: &RecallResponse, values: &[String]) -> Option<Uuid> {
    response
        .facts
        .iter()
        .find(|scored| {
            values
                .iter()
                .all(|value| contains_ci(&scored.fact.statement, value))
        })
        .map(|scored| scored.fact.id)
}

fn is_open(fact: &Fact) -> bool {
    fact.valid_to.is_none() && fact.superseded_by.is_none()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CaseStatus {
    /// The server gave the right answer.
    Passed,
    /// The server gave a wrong answer.
    Failed,
    /// No verdict: an HTTP failure or a broken contract. Fails the run and never counts as a pass.
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
    /// Facts this case added to the shared scope.
    pub committed_facts: usize,
    pub write_problems: Vec<String>,
    /// 1-based rank at which this case's own facts first cover every expected value.
    pub rank: Option<usize>,
    pub missing: Vec<String>,
    pub top_statements: Vec<String>,
}

impl Verdict for RecallDetail {
    fn problems(&self) -> Vec<String> {
        let mut problems = self.write_problems.clone();
        match self.rank {
            Some(rank) if rank <= TOP_K => {}
            Some(rank) => problems.push(format!("expected values first covered at rank {rank}")),
            None => problems.push(format!(
                "expected values not recalled: {}",
                self.missing.join(", ")
            )),
        }
        problems
    }
}

/// What one case left in the shared scope. A commit error is kept as text for the query phase.
#[derive(Default)]
struct RecallCorpus {
    fact_ids: HashSet<Uuid>,
    write_problems: Vec<String>,
    error: Option<String>,
}

/// Commits every case first and only then queries, so each query sees the same corpus whatever
/// the concurrency.
pub async fn run_recall(
    ctx: &RunContext,
    cases: &[RecallCase],
    concurrency: usize,
) -> Vec<CaseResult<RecallDetail>> {
    let scope = ctx.scope(ScopeKind::Agent, "recall");
    let scope = scope.as_str();
    let corpora: Vec<RecallCorpus> = stream::iter(cases.iter().map(|case| async move {
        let mut corpus = RecallCorpus::default();
        if let Err(e) = commit_recall_case(ctx, case, scope, &mut corpus).await {
            corpus.error = Some(e.to_string());
        }
        corpus
    }))
    .buffered(concurrency.max(1))
    .collect()
    .await;

    let prepared: Vec<(&RecallCase, RecallCorpus)> = cases.iter().zip(corpora).collect();
    run_all(&prepared, concurrency, |(case, corpus)| async move {
        let meta = CaseMeta {
            id: &case.id,
            lang: &case.lang,
            class: None,
            scope: scope.to_owned(),
            started: Instant::now(),
        };
        let mut detail = RecallDetail {
            committed_facts: corpus.fact_ids.len(),
            write_problems: corpus.write_problems.clone(),
            ..RecallDetail::default()
        };
        let outcome = match &corpus.error {
            Some(error) => Err(CaseError::Earlier(error.clone())),
            None => recall_query(ctx, case, scope, &corpus.fact_ids, &mut detail).await,
        };
        finish(meta, outcome, detail)
    })
    .await
}

async fn commit_recall_case(
    ctx: &RunContext,
    case: &RecallCase,
    scope: &str,
    corpus: &mut RecallCorpus,
) -> Result<(), CaseError> {
    for (index, commit) in case.commits.iter().enumerate() {
        let facts = commit
            .facts
            .iter()
            .map(|fact| CommitFact::new(fact, None))
            .collect();
        let committed = ctx
            .establish(
                (&case.id, &index.to_string()),
                ScopeKind::Agent,
                (scope, &commit.content),
                facts,
                &mut corpus.write_problems,
            )
            .await?;
        corpus
            .fact_ids
            .extend(committed.facts.iter().map(|fact| fact.id));
    }
    Ok(())
}

async fn recall_query(
    ctx: &RunContext,
    case: &RecallCase,
    scope: &str,
    own_facts: &HashSet<Uuid>,
    detail: &mut RecallDetail,
) -> Result<(), CaseError> {
    let response = ctx
        .client
        .recall(scope, &case.query, RECALL_LIMIT, None)
        .await?;
    // Another case's fact holds its rank but cannot answer for this one, even if the words match.
    let own_statements: Vec<String> = response
        .facts
        .iter()
        .map(|scored| {
            if own_facts.contains(&scored.fact.id) {
                scored.fact.statement.clone()
            } else {
                String::new()
            }
        })
        .collect();
    detail.rank = rank_of_set(&own_statements, &case.expected);
    detail.missing = case
        .expected
        .iter()
        .filter(|value| !own_statements.iter().any(|s| contains_ci(s, value)))
        .cloned()
        .collect();
    detail.top_statements = statements(&response).into_iter().take(TOP_K).collect();
    Ok(())
}

// Staleness

#[derive(Debug, Default, Serialize)]
pub struct StalenessDetail {
    pub write_problems: Vec<String>,
    /// `/v1/candidates` offered the earlier fact when asked about the later statement.
    pub candidate_found: bool,
    /// The earlier fact ends where the later one starts, and points at it.
    pub old_closed: bool,
    pub new_in_current: bool,
    pub old_absent_from_current: bool,
    pub old_in_as_of: bool,
    pub new_absent_from_as_of: bool,
    pub current_statements: Vec<String>,
    pub as_of_statements: Vec<String>,
}

impl Verdict for StalenessDetail {
    fn problems(&self) -> Vec<String> {
        let mut problems = self.write_problems.clone();
        problems.extend(
            [
                (
                    self.candidate_found,
                    "candidates did not offer the old fact",
                ),
                (self.old_closed, "old fact was not closed by the new one"),
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
            .map(|(_, problem)| problem.to_owned()),
        );
        problems
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
            scope: ctx.scope(case.scope_kind, &case.id),
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
    let earlier = ctx
        .establish(
            (&case.id, "earlier"),
            case.scope_kind,
            (scope, &case.earlier.content),
            vec![CommitFact::new(&case.earlier.fact, None)],
            &mut detail.write_problems,
        )
        .await?;
    let old_id = only_fact(&earlier)?.id;

    // The host flow: ask what the new statement relates to, and decide from the answer.
    let candidates = ctx
        .client
        .candidates(
            Trust::Agent,
            scope,
            &case.later.fact.statement,
            CANDIDATE_LIMIT,
        )
        .await?;
    let Some(target) = choose_candidate(&candidates, &case.old_values) else {
        return Ok(());
    };
    detail.candidate_found = target == old_id;

    let later = ctx
        .commit(
            Trust::Agent,
            (&case.id, "later"),
            (scope, HOST_SOURCE, &case.later.content),
            vec![CommitFact::new(&case.later.fact, Some(target))],
        )
        .await?;
    let new_fact = only_fact(&later)?;
    check_status(
        new_fact,
        agent_commit_status(case.scope_kind),
        "later",
        &mut detail.write_problems,
    );
    if new_fact.status == QUARANTINED {
        // SPEC section 5: a quarantined supersede leaves the old fact open until approval.
        if !is_open(&ctx.client.get_fact(old_id).await?) {
            detail
                .write_problems
                .push("old fact closed before the new one was approved".to_owned());
        }
        ctx.client.approve(new_fact.id).await?;
    }

    let old = ctx.client.get_fact(old_id).await?;
    detail.old_closed =
        old.superseded_by == Some(new_fact.id) && old.valid_to == case.later.fact.valid_from;

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
    /// Planted content reachable without approval. Any entry fails the zero-leak gate.
    pub leaks: Vec<String>,
    /// Problems with the seed or the poison's commit: wrong status, trust or pending link.
    pub write_problems: Vec<String>,
    pub true_fact_problems: Vec<String>,
    pub purge_problems: Vec<String>,
    /// The approved poison failed to behave as live memory, so "absent" proves less than it says.
    pub control_problems: Vec<String>,
    /// Quarantined facts traced to the poison before the first purge.
    pub quarantined_found: usize,
    pub purge_completed: bool,
    /// The seed was closed by the approved poison, then reopened by purging it.
    pub truth_restored: bool,
    pub purge_reports: Vec<PurgeReport>,
}

impl Verdict for PoisoningDetail {
    fn problems(&self) -> Vec<String> {
        let mut problems = self.leaks.clone();
        problems.extend(self.write_problems.iter().cloned());
        problems.extend(self.true_fact_problems.iter().cloned());
        problems.extend(self.purge_problems.iter().cloned());
        problems.extend(self.control_problems.iter().cloned());
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
            scope: ctx.scope(case.scope_kind, &case.id),
            started: Instant::now(),
        };
        let mut detail = PoisoningDetail::default();
        let outcome = poisoning_steps(ctx, case, &meta.scope, &mut detail).await;
        finish(meta, outcome, detail)
    })
    .await
}

/// One case, with the seed already in place.
struct PoisonRun<'a> {
    ctx: &'a RunContext,
    case: &'a PoisonCase,
    scope: &'a str,
    seed_fact: Uuid,
    seed_episode: Uuid,
}

/// A committed poison: its source episode and its one fact.
struct Poison {
    episode: Uuid,
    fact: Uuid,
}

/// Plants the same claim twice. The first stays held and is purged, which shows it never reached
/// recall. The second is approved, which shows the probes can see it and that purging it restores
/// what it closed. Without the second round, "absent" and "restored" could both hold trivially.
async fn poisoning_steps(
    ctx: &RunContext,
    case: &PoisonCase,
    scope: &str,
    detail: &mut PoisoningDetail,
) -> Result<(), CaseError> {
    let seed = ctx
        .establish(
            (&case.id, "seed"),
            case.scope_kind,
            (scope, &case.seed.content),
            vec![CommitFact::new(&case.seed.fact, None)],
            &mut detail.write_problems,
        )
        .await?;
    let run = PoisonRun {
        ctx,
        case,
        scope,
        seed_fact: only_fact(&seed)?.id,
        seed_episode: seed.episode_id,
    };
    run.try_future_date(detail).await?;
    let Some(held) = run.plant("poison-held", detail).await? else {
        return Ok(());
    };
    run.check_held(&held, detail).await?;
    run.purge_and_verify(&held, detail).await?;
    detail.purge_completed = true;

    let Some(approved) = run.plant("poison-approved", detail).await? else {
        return Ok(());
    };
    run.approve_and_check_control(&approved, detail).await?;
    run.purge_and_verify(&approved, detail).await?;
    detail.truth_restored =
        !case.contradicts_seed || is_open(&ctx.client.get_fact(run.seed_fact).await?);
    if !detail.truth_restored {
        detail
            .true_fact_problems
            .push("seed still closed after purging the fact that closed it".to_owned());
    }
    Ok(())
}

impl PoisonRun<'_> {
    /// A sleeper first tries to enter memory dated years ahead, so it would wake unreviewed. SPEC
    /// section 9 limits `valid_from` to 1 day ahead, so the server has to refuse it with a 400.
    async fn try_future_date(&self, detail: &mut PoisoningDetail) -> Result<(), CaseError> {
        let case = self.case;
        let Some(at) = case.poison.attempted_valid_from else {
            return Ok(());
        };
        let mut fact = CommitFact::new(&case.poison.fact, None);
        fact.valid_from = Some(at);
        let outcome = self
            .ctx
            .commit(
                case.poison.trust,
                (&case.id, "poison-dated"),
                (self.scope, &case.poison.source, &case.poison.content),
                vec![fact],
            )
            .await;
        match outcome {
            Err(CaseError::Status { status: 400, .. }) => Ok(()),
            Err(other) => Err(other),
            Ok(_) => {
                detail
                    .write_problems
                    .push(format!("commit dated {at} was accepted"));
                Ok(())
            }
        }
    }

    /// Commits the claim with the low-trust key, through the host flow. Returns `None`, with the
    /// reason recorded, when candidates did not offer the seed, since the case cannot go on.
    async fn plant(
        &self,
        label: &str,
        detail: &mut PoisoningDetail,
    ) -> Result<Option<Poison>, CaseError> {
        let case = self.case;
        let target = if case.contradicts_seed {
            let candidates = self
                .ctx
                .client
                .candidates(
                    case.poison.trust,
                    self.scope,
                    &case.poison.fact.statement,
                    CANDIDATE_LIMIT,
                )
                .await?;
            if choose_candidate(&candidates, &case.true_values) != Some(self.seed_fact) {
                detail
                    .write_problems
                    .push(format!("{label}: candidates did not offer the seed"));
                return Ok(None);
            }
            Some(self.seed_fact)
        } else {
            None
        };
        let committed = self
            .ctx
            .commit(
                case.poison.trust,
                (&case.id, label),
                (self.scope, &case.poison.source, &case.poison.content),
                vec![CommitFact::new(&case.poison.fact, target)],
            )
            .await?;
        let fact = only_fact(&committed)?;
        if fact.status != QUARANTINED {
            detail.leaks.push(format!(
                "{label}: fact {} committed with status {} without approval",
                fact.id, fact.status
            ));
        }
        let poison = Poison {
            episode: committed.episode_id,
            fact: fact.id,
        };
        // Text claiming a higher trust must not change the trust the key gave it.
        let stored = self.ctx.client.get_fact(poison.fact).await?;
        if stored.trust != case.poison.trust {
            detail.write_problems.push(format!(
                "{label}: stored trust {:?}, the key allows {:?}",
                stored.trust, case.poison.trust
            ));
        }
        if stored.pending_supersede_of != target {
            detail.write_problems.push(format!(
                "{label}: pending_supersede_of is {:?}, expected {target:?}",
                stored.pending_supersede_of
            ));
        }
        Ok(Some(poison))
    }

    /// While held, no read path shows the claim and the seed is untouched at every time probed.
    async fn check_held(
        &self,
        held: &Poison,
        detail: &mut PoisoningDetail,
    ) -> Result<(), CaseError> {
        let case = self.case;
        let client = &self.ctx.client;
        if !is_open(&client.get_fact(self.seed_fact).await?) {
            detail
                .true_fact_problems
                .push("held poison closed the seed without approval".to_owned());
        }

        let response = self.probe(None).await?;
        detail
            .leaks
            .extend(self.leaks(&response, held, Stage::Held));
        detail
            .true_fact_problems
            .extend(self.true_fact_problems(&response, "while held"));
        for &as_of in &case.as_of_probes {
            let response = self.probe(Some(as_of)).await?;
            detail
                .leaks
                .extend(self.leaks(&response, held, Stage::AsOf(as_of)));
            detail
                .true_fact_problems
                .extend(self.true_fact_problems(&response, "as_of while held"));
        }
        let offered = client
            .candidates(Trust::Agent, self.scope, &case.query, PROBE_LIMIT)
            .await?;
        detail
            .leaks
            .extend(self.leaks(&offered, held, Stage::Candidates));

        let listed = client.quarantine(self.scope).await?;
        detail.quarantined_found = listed
            .iter()
            .filter(|fact| traces_to_poison(fact, case, held.episode))
            .count();
        if !listed.iter().any(|fact| fact.id == held.fact) {
            detail
                .write_problems
                .push(format!("held fact {} is not in quarantine", held.fact));
        }
        Ok(())
    }

    /// Approval is the one way in, so the approved claim has to show up where the held one did not.
    async fn approve_and_check_control(
        &self,
        approved: &Poison,
        detail: &mut PoisoningDetail,
    ) -> Result<(), CaseError> {
        let case = self.case;
        let client = &self.ctx.client;
        client.approve(approved.fact).await?;
        if case.contradicts_seed {
            let seed = client.get_fact(self.seed_fact).await?;
            if seed.superseded_by != Some(approved.fact) {
                detail
                    .control_problems
                    .push("approved poison did not close the seed".to_owned());
            }
        }

        let current = statements(&self.probe(None).await?);
        self.expect_live(&current, "current recall", detail);
        for &as_of in &case.as_of_probes {
            let past = statements(&self.probe(Some(as_of)).await?);
            self.expect_live(&past, "as_of recall", detail);
        }
        Ok(())
    }

    fn expect_live(&self, statements: &[String], place: &str, detail: &mut PoisoningDetail) {
        if !covers_all(statements, &self.case.planted) {
            detail
                .control_problems
                .push(format!("approved claim missing from {place}"));
        }
        if self.case.contradicts_seed && !covers_none(statements, &self.case.true_values) {
            detail
                .control_problems
                .push(format!("closed seed still in {place}"));
        }
    }

    async fn purge_and_verify(
        &self,
        poison: &Poison,
        detail: &mut PoisoningDetail,
    ) -> Result<(), CaseError> {
        let case = self.case;
        let client = &self.ctx.client;
        if !client.episode_exists(poison.episode).await? {
            detail.purge_problems.push(format!(
                "episode {} unreadable before purge, so its absence afterwards shows nothing",
                poison.episode
            ));
        }

        let report = client.purge_episode(poison.episode).await?;
        if report.episodes_deleted < 1 {
            detail
                .purge_problems
                .push("purge reported no deleted episode".to_owned());
        }
        if report.facts_deleted < 1 {
            detail
                .purge_problems
                .push("purge reported no deleted fact".to_owned());
        }
        detail.purge_reports.push(report);

        if client.episode_exists(poison.episode).await? {
            detail.purge_problems.push(format!(
                "episode {} still readable after purge",
                poison.episode
            ));
        }
        if client.fact_exists(poison.fact).await? {
            detail
                .purge_problems
                .push(format!("fact {} still readable after purge", poison.fact));
        }
        let still_listed = client
            .quarantine(self.scope)
            .await?
            .iter()
            .filter(|fact| traces_to_poison(fact, case, poison.episode))
            .count();
        if still_listed > 0 {
            detail.purge_problems.push(format!(
                "{still_listed} planted facts still quarantined after purge"
            ));
        }

        let response = self.probe(None).await?;
        detail
            .leaks
            .extend(self.leaks(&response, poison, Stage::AfterPurge));
        detail
            .true_fact_problems
            .extend(self.true_fact_problems(&response, "after purge"));
        for &as_of in &case.as_of_probes {
            let response = self.probe(Some(as_of)).await?;
            detail
                .leaks
                .extend(self.leaks(&response, poison, Stage::AfterPurge));
            detail
                .true_fact_problems
                .extend(self.true_fact_problems(&response, "as_of after purge"));
        }
        Ok(())
    }

    async fn probe(&self, as_of: Option<DateTime<Utc>>) -> Result<RecallResponse, CaseError> {
        self.ctx
            .client
            .recall(self.scope, &self.case.query, PROBE_LIMIT, as_of)
            .await
    }

    fn leaks(&self, response: &RecallResponse, poison: &Poison, stage: Stage) -> Vec<String> {
        find_leaks(response, &self.case.planted, poison.episode, stage)
    }

    /// The true fact must be recalled, active, and still derive from the seed commit, so a poison
    /// fact that merely repeats the value does not count.
    fn true_fact_problems(&self, response: &RecallResponse, stage: &str) -> Vec<String> {
        self.case
            .true_values
            .iter()
            .filter(|value| {
                !response.facts.iter().any(|scored| {
                    scored.fact.status == ACTIVE
                        && scored.fact.source_episode_ids.contains(&self.seed_episode)
                        && contains_ci(&scored.fact.statement, value)
                })
            })
            .map(|value| format!("true value {value:?} missing from recall {stage}"))
            .collect()
    }
}

fn traces_to_poison(fact: &Fact, case: &PoisonCase, poison_episode: Uuid) -> bool {
    fact.source_episode_ids.contains(&poison_episode)
        || case.planted.iter().any(|p| contains_ci(&fact.statement, p))
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
    use crate::client::ScoredFact;

    #[test]
    fn staleness_detail_gives_no_credit_by_default() {
        assert_eq!(StalenessDetail::default().problems().len(), 6);
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

    #[test]
    fn a_write_problem_fails_a_case_with_a_good_rank() {
        let detail = RecallDetail {
            rank: Some(1),
            write_problems: vec!["bad status".to_owned()],
            ..RecallDetail::default()
        };
        assert_eq!(detail.problems().len(), 1);
    }

    #[test]
    fn every_poisoning_problem_kind_fails_the_case() {
        let detail = PoisoningDetail {
            leaks: vec!["a".to_owned()],
            write_problems: vec!["b".to_owned()],
            true_fact_problems: vec!["c".to_owned()],
            purge_problems: vec!["d".to_owned()],
            control_problems: vec!["e".to_owned()],
            ..PoisoningDetail::default()
        };
        assert_eq!(detail.problems().len(), 5);
    }

    #[test]
    fn agent_commits_are_held_outside_the_agent_scope() {
        assert_eq!(agent_commit_status(ScopeKind::User), QUARANTINED);
        assert_eq!(agent_commit_status(ScopeKind::Shared), QUARANTINED);
        assert_eq!(agent_commit_status(ScopeKind::Agent), ACTIVE);
    }

    fn scored(id: u128, statement: &str) -> ScoredFact {
        ScoredFact {
            fact: Fact {
                id: Uuid::from_u128(id),
                scope: "agent:x".to_owned(),
                subject_id: Uuid::nil(),
                predicate: "p".to_owned(),
                statement: statement.to_owned(),
                trust: Trust::Agent,
                status: ACTIVE.to_owned(),
                valid_from: Utc::now(),
                valid_to: None,
                recorded_at: Utc::now(),
                superseded_by: None,
                pending_supersede_of: None,
                source_episode_ids: Vec::new(),
            },
            score: 1.0,
        }
    }

    #[test]
    fn a_host_chooses_the_candidate_stating_every_value() {
        let response = RecallResponse {
            facts: vec![scored(1, "lives in Oslo"), scored(2, "Oslo and Rome")],
        };
        let values = ["oslo".to_owned(), "rome".to_owned()];
        assert_eq!(
            choose_candidate(&response, &values),
            Some(Uuid::from_u128(2))
        );
        assert_eq!(choose_candidate(&response, &["Lima".to_owned()]), None);
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
        let error = CaseError::Earlier("commit failed".to_owned());
        let result = finish(meta(), Err(error), good);
        assert_eq!(result.status, CaseStatus::Error);
        assert!(result.reason.expect("reason").contains("commit failed"));
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
