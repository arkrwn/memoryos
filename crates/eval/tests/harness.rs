//! Runs the real `memoryos-eval` binary against WireMock stubs generated from the real datasets.
//!
//! Stubs require the correct bearer token, so a call with the wrong key gets WireMock's 404 and
//! the case errors. Poisoning stubs use a scenario per case that follows its two rounds, so a read
//! answers as a real server would at that point: held, purged, approved, purged again.
#![cfg(feature = "integration")]

use std::collections::HashMap;
use std::process::Output;
use std::time::Duration;

use memoryos_eval::client::Trust;
use memoryos_eval::dataset::{self, PoisonCase, RecallCase, ScopeKind, StalenessCase};
use serde_json::{json, Value};
use testcontainers::core::{IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, ImageExt};
use tokio::sync::{Semaphore, SemaphorePermit};
use uuid::Uuid;

const AGENT_TOKEN: &str = "mos_agent_token";
const EXTERNAL_TOKEN: &str = "mos_external_token";
const ADMIN_TOKEN: &str = "mos_admin_token";
const OVERRIDE_PRIORITY: u32 = 1;
const AS_OF_PRIORITY: u32 = 2;
const DEFAULT_PRIORITY: u32 = 5;
const NOW: &str = "2026-01-01T00:00:00Z";

const HELD: &str = "Started";
const FIRST_PURGED: &str = "first purged";
const APPROVED: &str = "approved";
const SECOND_PURGED: &str = "second purged";

/// Each test sends thousands of requests to its own WireMock. Too many at once make WireMock reset
/// connections, which the harness rightly reports as errors, so only this many tests run at a time.
static WIREMOCK_SLOTS: Semaphore = Semaphore::const_new(2);

struct Wiremock {
    _container: ContainerAsync<GenericImage>,
    _slot: SemaphorePermit<'static>,
    url: String,
}

async fn start_wiremock() -> Wiremock {
    let slot = WIREMOCK_SLOTS.acquire().await.expect("never closed");
    let container = GenericImage::new("wiremock/wiremock", "3.3.1")
        .with_exposed_port(8080.tcp())
        .with_wait_for(WaitFor::message_on_stdout("port:"))
        .with_startup_timeout(Duration::from_secs(120))
        // Without a limit the JVM takes a quarter of the Docker VM, and two of these grew to about
        // 860 MB each. That pushed a VM other containers share into swap, WireMock stalled for
        // over 30 s, and the harness's requests timed out as case errors.
        .with_env_var("JAVA_OPTS", "-Xmx256m -XX:+UseSerialGC")
        .start()
        .await
        .expect("wiremock starts");
    let port = container
        .get_host_port_ipv4(8080.tcp())
        .await
        .expect("mapped port");
    let url = format!("http://127.0.0.1:{port}");
    let http = reqwest::Client::new();
    for _ in 0..100 {
        let health = http.get(format!("{url}/__admin/health")).send().await;
        if health.is_ok_and(|r| r.status().is_success()) {
            return Wiremock {
                _container: container,
                _slot: slot,
                url,
            };
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("wiremock never became healthy");
}

impl Wiremock {
    async fn import(&self, mappings: Vec<Value>) {
        let response = reqwest::Client::new()
            .post(format!("{}/__admin/mappings/import", self.url))
            .json(&json!({ "mappings": mappings }))
            .send()
            .await
            .expect("import request");
        assert!(
            response.status().is_success(),
            "import failed: {}",
            response.text().await.unwrap_or_default()
        );
    }

    /// How many requests reached this method and path.
    async fn count(&self, method: &str, path: &str) -> u64 {
        let response: Value = reqwest::Client::new()
            .post(format!("{}/__admin/requests/count", self.url))
            .json(&json!({ "method": method, "urlPath": path }))
            .send()
            .await
            .expect("count request")
            .json()
            .await
            .expect("count body");
        response["count"].as_u64().expect("count")
    }

    async fn unmatched(&self) -> Vec<Value> {
        let response: Value = reqwest::Client::new()
            .get(format!("{}/__admin/requests/unmatched", self.url))
            .send()
            .await
            .expect("unmatched request")
            .json()
            .await
            .expect("unmatched body");
        response["requests"].as_array().expect("requests").clone()
    }
}

/// Deterministic ids, so each commit's stubs can reference one another.
#[derive(Default)]
struct Ids {
    next: u128,
    named: HashMap<String, Uuid>,
}

impl Ids {
    fn of(&mut self, name: &str) -> Uuid {
        let next = &mut self.next;
        *self.named.entry(name.to_owned()).or_insert_with(|| {
            *next += 1;
            Uuid::from_u128(*next)
        })
    }
}

fn auth(token: &str) -> Value {
    json!({ "Authorization": { "equalTo": format!("Bearer {token}") } })
}

fn token_for(trust: Trust) -> &'static str {
    if trust == Trust::External {
        EXTERNAL_TOKEN
    } else {
        AGENT_TOKEN
    }
}

fn fact_json(id: Uuid, statement: &str, status: &str, trust: &str, sources: &[Uuid]) -> Value {
    json!({
        "id": id,
        "scope": "agent:stub",
        "subject_id": Uuid::from_u128(u128::MAX),
        "predicate": "stub",
        "statement": statement,
        "trust": trust,
        "status": status,
        "valid_from": NOW,
        "recorded_at": NOW,
        "source_episode_ids": sources,
    })
}

fn scored(facts: Vec<Value>) -> Value {
    let facts: Vec<Value> = facts
        .into_iter()
        .map(|fact| json!({ "fact": fact, "score": 0.5 }))
        .collect();
    json!({ "facts": facts })
}

fn json_response(status: u16, body: &Value) -> Value {
    json!({
        "status": status,
        "jsonBody": body,
        "headers": { "Content-Type": "application/json" },
    })
}

/// Adds the WireMock scenario fields, if the stub belongs to one state of a scenario.
fn in_state(mut stub: Value, scenario: &str, state: &str, next: Option<&str>) -> Value {
    stub["scenarioName"] = json!(scenario);
    stub["requiredScenarioState"] = json!(state);
    if let Some(next) = next {
        stub["newScenarioState"] = json!(next);
    }
    stub
}

/// `POST /v1/commits` for one commit, selected by its idempotency key suffix.
fn commit_stub(
    token: &str,
    case_id: &str,
    label: &str,
    episode: Uuid,
    facts: &[(Uuid, &str)],
) -> Value {
    let facts: Vec<Value> = facts
        .iter()
        .map(|(id, status)| json!({ "id": id, "status": status }))
        .collect();
    json!({
        "priority": DEFAULT_PRIORITY,
        "request": {
            "method": "POST",
            "urlPath": "/v1/commits",
            "headers": auth(token),
            "bodyPatterns": [{ "matchesJsonPath": {
                "expression": "$.idempotency_key",
                "matches": format!(".*-{case_id}-{label}$"),
            }}],
        },
        "response": json_response(200, &json!({ "episode_id": episode, "facts": facts })),
    })
}

fn approve_stub(id: Uuid) -> Value {
    json!({
        "priority": DEFAULT_PRIORITY,
        "request": {
            "method": "POST",
            "urlPath": format!("/v1/quarantine/{id}/approve"),
            "headers": auth(ADMIN_TOKEN),
        },
        "response": json_response(200, &fact_json(id, "approved", "active", "agent", &[])),
    })
}

fn get_fact_stub(id: Uuid, body: &Value) -> Value {
    json!({
        "priority": DEFAULT_PRIORITY,
        "request": {
            "method": "GET",
            "urlPath": format!("/v1/facts/{id}"),
            "headers": auth(AGENT_TOKEN),
        },
        "response": json_response(200, body),
    })
}

fn scope_pattern(case_id: &str) -> Value {
    json!({ "matchesJsonPath": {
        "expression": "$.scopes[0]",
        "matches": format!(".*-{case_id}$"),
    }})
}

fn recall_stub(case_id: &str, priority: u32, as_of: bool, body: &Value) -> Value {
    let mut patterns = vec![scope_pattern(case_id)];
    if as_of {
        patterns.push(json!({ "matchesJsonPath": "$.as_of" }));
    }
    json!({
        "priority": priority,
        "request": {
            "method": "POST",
            "urlPath": "/v1/recall",
            "headers": auth(AGENT_TOKEN),
            "bodyPatterns": patterns,
        },
        "response": json_response(200, body),
    })
}

fn candidates_stub(token: &str, case_id: &str, body: &Value) -> Value {
    json!({
        "priority": DEFAULT_PRIORITY,
        "request": {
            "method": "POST",
            "urlPath": "/v1/candidates",
            "headers": auth(token),
            "bodyPatterns": [scope_pattern(case_id)],
        },
        "response": json_response(200, body),
    })
}

/// All recall cases share a scope, so their queries select the stub.
fn recall_query_stub(priority: u32, query: &str, body: &Value) -> Value {
    json!({
        "priority": priority,
        "request": {
            "method": "POST",
            "urlPath": "/v1/recall",
            "headers": auth(AGENT_TOKEN),
            "bodyPatterns": [{ "matchesJsonPath": {
                "expression": "$.query",
                "equalTo": query,
            }}],
        },
        "response": json_response(200, body),
    })
}

fn recall_case_stubs(case: &RecallCase, ids: &mut Ids) -> Vec<Value> {
    let mut stubs = Vec::new();
    let mut committed: Vec<(Uuid, &str)> = Vec::new();
    for (index, commit) in case.commits.iter().enumerate() {
        let label = index.to_string();
        let episode = ids.of(&format!("{}/{label}", case.id));
        let facts: Vec<(Uuid, &str)> = commit
            .facts
            .iter()
            .map(|fact| {
                let id = ids.of(&format!("{}/fact/{}", case.id, fact.statement));
                (id, "active")
            })
            .collect();
        stubs.push(commit_stub(AGENT_TOKEN, &case.id, &label, episode, &facts));
        committed.extend(
            commit
                .facts
                .iter()
                .zip(&facts)
                .map(|(fact, (id, _))| (*id, fact.statement.as_str())),
        );
    }
    // One fact per expected value, in order, so a set is covered at rank `expected.len()`.
    let recalled = case
        .expected
        .iter()
        .map(|value| {
            let (id, statement) = committed
                .iter()
                .find(|(_, statement)| statement.to_lowercase().contains(&value.to_lowercase()))
                .expect("dataset validation found a statement");
            fact_json(*id, statement, "active", "agent", &[])
        })
        .collect();
    stubs.push(recall_query_stub(
        DEFAULT_PRIORITY,
        &case.query,
        &scored(recalled),
    ));
    stubs
}

fn status_for(kind: ScopeKind) -> &'static str {
    if kind.holds_agent_trust() {
        "quarantined"
    } else {
        "active"
    }
}

fn staleness_case_stubs(case: &StalenessCase, ids: &mut Ids) -> Vec<Value> {
    let old = ids.of(&format!("{}/old", case.id));
    let new = ids.of(&format!("{}/new", case.id));
    let status = status_for(case.scope_kind);
    let old_statement = &case.earlier.fact.statement;
    let old_json = fact_json(old, old_statement, "active", "agent", &[]);
    let new_json = fact_json(new, &case.later.fact.statement, "active", "agent", &[]);

    let mut closed = old_json.clone();
    closed["superseded_by"] = json!(new);
    closed["valid_to"] = json!(case.later.fact.valid_from);

    let mut stubs = vec![
        commit_stub(
            AGENT_TOKEN,
            &case.id,
            "earlier",
            ids.of(&format!("{}/earlier", case.id)),
            &[(old, status)],
        ),
        commit_stub(
            AGENT_TOKEN,
            &case.id,
            "later",
            ids.of(&format!("{}/later", case.id)),
            &[(new, status)],
        ),
        candidates_stub(AGENT_TOKEN, &case.id, &scored(vec![old_json.clone()])),
        recall_stub(
            &case.id,
            DEFAULT_PRIORITY,
            false,
            &scored(vec![new_json.clone()]),
        ),
        recall_stub(
            &case.id,
            AS_OF_PRIORITY,
            true,
            &scored(vec![old_json.clone()]),
        ),
    ];
    if case.scope_kind.holds_agent_trust() {
        // The old fact stays open until the new one is approved.
        let scenario = format!("close-{}", case.id);
        stubs.push(approve_stub(old));
        stubs.push(in_state(approve_stub(new), &scenario, HELD, Some(APPROVED)));
        stubs.push(in_state(
            get_fact_stub(old, &old_json),
            &scenario,
            HELD,
            None,
        ));
        stubs.push(in_state(
            get_fact_stub(old, &closed),
            &scenario,
            APPROVED,
            None,
        ));
    } else {
        stubs.push(get_fact_stub(old, &closed));
    }
    stubs
}

/// Ids of the two rounds of one poison case.
struct PoisonIds {
    seed_episode: Uuid,
    seed_fact: Uuid,
    held_episode: Uuid,
    held_fact: Uuid,
    approved_episode: Uuid,
    approved_fact: Uuid,
}

fn poison_ids(case: &PoisonCase, ids: &mut Ids) -> PoisonIds {
    let mut of = |name: &str| ids.of(&format!("{}/{name}", case.id));
    PoisonIds {
        seed_episode: of("seed"),
        seed_fact: of("seed-fact"),
        held_episode: of("held"),
        held_fact: of("held-fact"),
        approved_episode: of("approved"),
        approved_fact: of("approved-fact"),
    }
}

fn trust_name(trust: Trust) -> &'static str {
    if trust == Trust::External {
        "external"
    } else {
        "agent"
    }
}

fn poison_case_stubs(case: &PoisonCase, ids: &mut Ids) -> Vec<Value> {
    let p = poison_ids(case, ids);
    let scenario = format!("rounds-{}", case.id);
    let trust = trust_name(case.poison.trust);
    let poison_token = token_for(case.poison.trust);
    let planted_statement = &case.poison.fact.statement;
    let true_fact = fact_json(
        p.seed_fact,
        &case.seed.fact.statement,
        "active",
        "agent",
        &[p.seed_episode],
    );
    let held = |fact: Uuid, episode: Uuid| {
        let mut json = fact_json(fact, planted_statement, "quarantined", trust, &[episode]);
        if case.contradicts_seed {
            json["pending_supersede_of"] = json!(p.seed_fact);
        }
        json
    };
    let live = fact_json(
        p.approved_fact,
        planted_statement,
        "active",
        trust,
        &[p.approved_episode],
    );
    let mut closed_seed = true_fact.clone();
    if case.contradicts_seed {
        closed_seed["superseded_by"] = json!(p.approved_fact);
        closed_seed["valid_to"] = json!(case.poison.fact.valid_from);
    }

    let quarantine = |facts: Vec<Value>, state: &str| {
        in_state(
            json!({
                "request": {
                    "method": "GET",
                    "urlPath": "/v1/quarantine",
                    "headers": auth(ADMIN_TOKEN),
                    "queryParameters": { "scopes": { "matches": format!(".*-{}$", case.id) } },
                },
                "response": json_response(200, &json!({ "facts": facts })),
            }),
            &scenario,
            state,
            None,
        )
    };
    let episode_get = |id: Uuid, state: &str| {
        in_state(
            json!({
                "request": {
                    "method": "GET",
                    "urlPath": format!("/v1/episodes/{id}"),
                    "headers": auth(AGENT_TOKEN),
                },
                "response": json_response(200, &json!({ "id": id })),
            }),
            &scenario,
            state,
            None,
        )
    };
    let purge = |id: Uuid, state: &str, next: &str| {
        in_state(
            json!({
                "request": {
                    "method": "POST",
                    "urlPath": format!("/v1/episodes/{id}/purge"),
                    "headers": auth(ADMIN_TOKEN),
                },
                "response": json_response(200, &json!({
                    "facts_deleted": 1,
                    "episodes_deleted": 1,
                    "entities_deleted": 0,
                })),
            }),
            &scenario,
            state,
            Some(next),
        )
    };

    let mut stubs = vec![
        commit_stub(
            AGENT_TOKEN,
            &case.id,
            "seed",
            p.seed_episode,
            &[(p.seed_fact, status_for(case.scope_kind))],
        ),
        commit_stub(
            poison_token,
            &case.id,
            "poison-held",
            p.held_episode,
            &[(p.held_fact, "quarantined")],
        ),
        commit_stub(
            poison_token,
            &case.id,
            "poison-approved",
            p.approved_episode,
            &[(p.approved_fact, "quarantined")],
        ),
        // Reads of a poison fact answer only while it exists; afterwards WireMock returns 404.
        in_state(
            get_fact_stub(p.held_fact, &held(p.held_fact, p.held_episode)),
            &scenario,
            HELD,
            None,
        ),
        in_state(
            get_fact_stub(p.approved_fact, &held(p.approved_fact, p.approved_episode)),
            &scenario,
            FIRST_PURGED,
            None,
        ),
        in_state(
            approve_stub(p.approved_fact),
            &scenario,
            FIRST_PURGED,
            Some(APPROVED),
        ),
        episode_get(p.held_episode, HELD),
        episode_get(p.approved_episode, APPROVED),
        purge(p.held_episode, HELD, FIRST_PURGED),
        purge(p.approved_episode, APPROVED, SECOND_PURGED),
        quarantine(vec![held(p.held_fact, p.held_episode)], HELD),
        quarantine(Vec::new(), FIRST_PURGED),
        quarantine(Vec::new(), SECOND_PURGED),
        candidates_stub(AGENT_TOKEN, &case.id, &scored(vec![true_fact.clone()])),
    ];
    if case.scope_kind.holds_agent_trust() {
        stubs.push(approve_stub(p.seed_fact));
    }
    if case.contradicts_seed && poison_token != AGENT_TOKEN {
        stubs.push(candidates_stub(
            poison_token,
            &case.id,
            &scored(vec![true_fact.clone()]),
        ));
    }

    // The seed, and recall, while the poison is not live.
    for state in [HELD, FIRST_PURGED, SECOND_PURGED] {
        stubs.push(in_state(
            get_fact_stub(p.seed_fact, &true_fact),
            &scenario,
            state,
            None,
        ));
        stubs.push(in_state(
            recall_stub(
                &case.id,
                DEFAULT_PRIORITY,
                false,
                &scored(vec![true_fact.clone()]),
            ),
            &scenario,
            state,
            None,
        ));
    }

    // A far-future date is refused. Without a stub the harness would see 404 and report an error.
    if case.poison.attempted_valid_from.is_some() {
        let mut refused = commit_stub(poison_token, &case.id, "poison-dated", Uuid::nil(), &[]);
        refused["response"] = json_response(
            400,
            &json!({ "error": { "code": "invalid_input", "message": "valid_from" } }),
        );
        stubs.push(refused);
    }

    // While approved, the seed is closed and the claim is live.
    stubs.push(in_state(
        get_fact_stub(p.seed_fact, &closed_seed),
        &scenario,
        APPROVED,
        None,
    ));
    let with_claim = if case.contradicts_seed {
        scored(vec![live.clone()])
    } else {
        scored(vec![live.clone(), true_fact.clone()])
    };
    stubs.push(in_state(
        recall_stub(&case.id, DEFAULT_PRIORITY, false, &with_claim),
        &scenario,
        APPROVED,
        None,
    ));
    if !case.as_of_probes.is_empty() {
        stubs.push(in_state(
            recall_stub(&case.id, AS_OF_PRIORITY, true, &with_claim),
            &scenario,
            APPROVED,
            None,
        ));
    }
    stubs
}

/// Stubs under which every case of every suite passes.
fn healthy_server(ids: &mut Ids) -> Vec<Value> {
    let mut stubs = Vec::new();
    for case in dataset::load_recall().expect("recall data") {
        stubs.extend(recall_case_stubs(&case, ids));
    }
    for case in dataset::load_staleness().expect("staleness data") {
        stubs.extend(staleness_case_stubs(&case, ids));
    }
    for case in dataset::load_poisoning().expect("poisoning data") {
        stubs.extend(poison_case_stubs(&case, ids));
    }
    stubs
}

struct Run {
    output: Output,
    report: Value,
}

async fn run_harness(url: &str, extra: &[&str]) -> Run {
    let dir = tempfile::tempdir().expect("temp dir");
    let out = dir.path().join("report.json");
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_memoryos-eval"))
        .args(["--base-url", url])
        .args(["--api-key", AGENT_TOKEN])
        .args(["--external-api-key", EXTERNAL_TOKEN])
        .args(["--admin-api-key", ADMIN_TOKEN])
        .arg("--out")
        .arg(&out)
        .args(extra)
        .output()
        .await
        .expect("binary runs");
    let text = std::fs::read_to_string(&out).unwrap_or_else(|e| {
        panic!(
            "no report ({e}): {}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    let report = serde_json::from_str(&text).expect("report is json");
    Run { output, report }
}

fn suite<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["suites"]
        .as_array()
        .expect("suites")
        .iter()
        .find(|s| s["suite"] == name)
        .expect("suite present")
}

fn case<'a>(suite: &'a Value, id: &str) -> &'a Value {
    suite["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["id"] == id)
        .expect("case present")
}

fn stdout(run: &Run) -> String {
    String::from_utf8_lossy(&run.output.stdout).into_owned()
}

#[tokio::test]
async fn a_correct_server_passes_every_suite_through_the_host_flow_only() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    wiremock.import(healthy_server(&mut ids)).await;

    let run = run_harness(&wiremock.url, &["--suite", "all"]).await;

    assert_eq!(
        run.output.status.code(),
        Some(0),
        "{} unmatched: {:#?}",
        stdout(&run),
        wiremock.unmatched().await
    );
    assert_eq!(run.report["passed"], json!(true));

    let recall = suite(&run.report, "recall");
    let recall_cases = dataset::load_recall().expect("data");
    assert_eq!(recall["total"], json!(34));
    assert_eq!(recall["cases_passed"], json!(34));
    assert_eq!(recall["metrics"]["recall_at_5"], json!(1.0));
    // The stubs return one fact per expected value, so a set is covered at rank `expected.len()`.
    let expected_mrr = recall_cases
        .iter()
        .map(|c| 1.0 / c.expected.len() as f64)
        .sum::<f64>()
        / recall_cases.len() as f64;
    let mrr = recall["metrics"]["mrr"].as_f64().expect("mrr");
    assert!((mrr - expected_mrr).abs() < 1e-9, "{mrr} vs {expected_mrr}");
    let corpus: usize = recall_cases
        .iter()
        .flat_map(|c| &c.commits)
        .map(|c| c.facts.len())
        .sum();
    assert_eq!(recall["metrics"]["corpus_facts"], json!(corpus));
    let scopes: Vec<&Value> = recall["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| &c["scope"])
        .collect();
    assert!(scopes.iter().all(|s| *s == scopes[0]), "one shared scope");

    let staleness = suite(&run.report, "staleness");
    assert_eq!(staleness["total"], json!(32));
    assert_eq!(staleness["metrics"]["pass_rate"], json!(1.0));
    assert_eq!(staleness["metrics"]["candidate_found_rate"], json!(1.0));
    assert_eq!(staleness["metrics"]["old_closed_rate"], json!(1.0));
    assert_eq!(staleness["cases_errored"], json!(0));

    let poisoning = suite(&run.report, "poisoning");
    assert_eq!(poisoning["total"], json!(30));
    assert_eq!(poisoning["metrics"]["leaks"], json!(0));
    assert_eq!(poisoning["metrics"]["purge_exercised_cases"], json!(30));
    assert_eq!(poisoning["metrics"]["truth_restored_cases"], json!(30));
    assert_eq!(poisoning["metrics"]["control_failure_cases"], json!(0));

    let table = stdout(&run);
    assert!(table.contains("result: PASS"), "{table}");
    let report_text = run.report.to_string();
    for token in [AGENT_TOKEN, EXTERNAL_TOKEN, ADMIN_TOKEN] {
        assert!(!report_text.contains(token));
    }

    // Nothing went anywhere but the host-mode endpoints the stubs describe.
    assert_eq!(wiremock.count("POST", "/v1/episodes").await, 0);
    assert!(wiremock.count("POST", "/v1/commits").await > 0);
    assert!(wiremock.count("POST", "/v1/candidates").await > 0);
    // The only unmatched requests are the reads after each purge, answered 404 on purpose.
    let unmatched = wiremock.unmatched().await;
    assert_eq!(unmatched.len(), 30 * 2 * 2, "{unmatched:#?}");
    assert!(
        unmatched.iter().all(|r| r["method"] == "GET"),
        "{unmatched:#?}"
    );
}

#[tokio::test]
async fn a_poisoned_recall_response_fails_and_exits_nonzero() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_poisoning().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(poison_case_stubs(case, &mut ids));
    }
    let target = &cases[0];
    let leaked = fact_json(
        Uuid::from_u128(u128::MAX - 1),
        &format!("The user is linked to {}", target.planted[0]),
        "active",
        "agent",
        &[ids.of(&format!("{}/seed", target.id))],
    );
    stubs.push(recall_stub(
        &target.id,
        OVERRIDE_PRIORITY,
        false,
        &scored(vec![leaked]),
    ));
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "poisoning"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    assert_eq!(run.report["passed"], json!(false));
    let poisoning = suite(&run.report, "poisoning");
    let poisoned = case(poisoning, &target.id);
    assert_eq!(poisoned["status"], json!("failed"));
    assert!(
        poisoned["reason"]
            .as_str()
            .expect("reason")
            .contains(&target.planted[0]),
        "{poisoned}"
    );
    assert!(poisoning["metrics"]["leaks"].as_u64().expect("leaks") >= 1);
    assert_eq!(poisoning["cases_errored"], json!(0));
    assert_eq!(poisoning["cases_passed"], json!(cases.len() - 1));
}

#[tokio::test]
async fn a_poison_committed_active_is_a_leak() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_poisoning().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(poison_case_stubs(case, &mut ids));
    }
    let target = &cases[0];
    let p = poison_ids(target, &mut ids);
    let mut active = commit_stub(
        token_for(target.poison.trust),
        &target.id,
        "poison-held",
        p.held_episode,
        &[(p.held_fact, "active")],
    );
    active["priority"] = json!(OVERRIDE_PRIORITY);
    stubs.push(active);
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "poisoning"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let poisoning = suite(&run.report, "poisoning");
    let failed = case(poisoning, &target.id);
    assert_eq!(failed["status"], json!("failed"));
    assert!(
        failed["reason"]
            .as_str()
            .expect("reason")
            .contains("without approval"),
        "{failed}"
    );
    assert_eq!(poisoning["metrics"]["leaks"], json!(1));
}

#[tokio::test]
async fn a_far_future_date_that_is_accepted_fails_the_case() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_poisoning().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(poison_case_stubs(case, &mut ids));
    }
    let target = cases
        .iter()
        .find(|c| c.poison.attempted_valid_from.is_some())
        .expect("a sleeper case");
    let mut accepted = commit_stub(
        token_for(target.poison.trust),
        &target.id,
        "poison-dated",
        Uuid::from_u128(u128::MAX - 2),
        &[(Uuid::from_u128(u128::MAX - 3), "quarantined")],
    );
    accepted["priority"] = json!(OVERRIDE_PRIORITY);
    stubs.push(accepted);
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "poisoning"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let failed = case(suite(&run.report, "poisoning"), &target.id);
    assert_eq!(failed["status"], json!("failed"));
    assert!(
        failed["reason"]
            .as_str()
            .expect("reason")
            .contains("was accepted"),
        "{failed}"
    );
}

#[tokio::test]
async fn a_purge_that_does_not_reopen_the_seed_fails() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_poisoning().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(poison_case_stubs(case, &mut ids));
    }
    let target = cases
        .iter()
        .find(|c| c.contradicts_seed)
        .expect("a contradicting case");
    let p = poison_ids(target, &mut ids);
    let mut closed = fact_json(p.seed_fact, "closed", "active", "agent", &[p.seed_episode]);
    closed["superseded_by"] = json!(p.approved_fact);
    let mut still_closed = in_state(
        get_fact_stub(p.seed_fact, &closed),
        &format!("rounds-{}", target.id),
        SECOND_PURGED,
        None,
    );
    still_closed["priority"] = json!(OVERRIDE_PRIORITY);
    stubs.push(still_closed);
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "poisoning"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let poisoning = suite(&run.report, "poisoning");
    let failed = case(poisoning, &target.id);
    assert_eq!(failed["status"], json!("failed"));
    assert!(
        failed["reason"]
            .as_str()
            .expect("reason")
            .contains("still closed"),
        "{failed}"
    );
    assert_eq!(poisoning["cases_errored"], json!(0));
}

#[tokio::test]
async fn an_approved_poison_that_recall_cannot_see_fails_the_control() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_poisoning().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(poison_case_stubs(case, &mut ids));
    }
    let target = cases
        .iter()
        .find(|c| c.as_of_probes.is_empty())
        .expect("a case with no dormant claim");
    let p = poison_ids(target, &mut ids);
    let seed = fact_json(
        p.seed_fact,
        &target.seed.fact.statement,
        "active",
        "agent",
        &[p.seed_episode],
    );
    let mut blind = in_state(
        recall_stub(&target.id, OVERRIDE_PRIORITY, false, &scored(vec![seed])),
        &format!("rounds-{}", target.id),
        APPROVED,
        None,
    );
    blind["priority"] = json!(OVERRIDE_PRIORITY);
    stubs.push(blind);
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "poisoning"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let poisoning = suite(&run.report, "poisoning");
    assert_eq!(case(poisoning, &target.id)["status"], json!("failed"));
    assert_eq!(poisoning["metrics"]["control_failure_cases"], json!(1));
    assert_eq!(poisoning["metrics"]["leaks"], json!(0));
}

#[tokio::test]
async fn a_commit_that_fails_is_an_error_not_a_pass() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_recall().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(recall_case_stubs(case, &mut ids));
    }
    stubs.push(json!({
        "priority": OVERRIDE_PRIORITY,
        "request": {
            "method": "POST",
            "urlPath": "/v1/commits",
            "bodyPatterns": [{ "matchesJsonPath": {
                "expression": "$.idempotency_key",
                "matches": format!(".*-{}-0$", cases[0].id),
            }}],
        },
        "response": json_response(500, &json!({
            "error": { "code": "internal", "message": "boom" }
        })),
    }));
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "recall", "--concurrency", "16"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let recall = suite(&run.report, "recall");
    let errored = case(recall, &cases[0].id);
    assert_eq!(errored["status"], json!("error"));
    assert!(
        errored["reason"].as_str().expect("reason").contains("500"),
        "{errored}"
    );
    assert_eq!(recall["cases_errored"], json!(1));
    assert_eq!(recall["cases_passed"], json!(cases.len() - 1));
    let expected = (cases.len() - 1) as f64 / cases.len() as f64;
    assert_eq!(recall["metrics"]["recall_at_5"], json!(expected));
}

#[tokio::test]
async fn a_commit_response_of_the_wrong_length_is_an_error() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_recall().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(recall_case_stubs(case, &mut ids));
    }
    let mut short = commit_stub(AGENT_TOKEN, &cases[0].id, "0", Uuid::nil(), &[]);
    short["priority"] = json!(OVERRIDE_PRIORITY);
    stubs.push(short);
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "recall"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let errored = case(suite(&run.report, "recall"), &cases[0].id);
    assert_eq!(errored["status"], json!("error"));
    assert!(
        errored["reason"]
            .as_str()
            .expect("reason")
            .contains("contract"),
        "{errored}"
    );
}

#[tokio::test]
async fn an_http_failure_is_an_error_and_fails_a_zero_threshold_gate() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_staleness().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(staleness_case_stubs(case, &mut ids));
    }
    stubs.push(json!({
        "priority": OVERRIDE_PRIORITY,
        "request": {
            "method": "POST",
            "urlPath": "/v1/recall",
            "bodyPatterns": [scope_pattern(&cases[0].id)],
        },
        "response": json_response(500, &json!({
            "error": { "code": "internal", "message": "boom" }
        })),
    }));
    wiremock.import(stubs).await;

    let run = run_harness(
        &wiremock.url,
        &["--suite", "staleness", "--min-staleness", "0"],
    )
    .await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let staleness = suite(&run.report, "staleness");
    let errored = case(staleness, &cases[0].id);
    assert_eq!(errored["status"], json!("error"));
    assert!(
        errored["reason"].as_str().expect("reason").contains("500"),
        "{errored}"
    );
}

#[tokio::test]
async fn a_stale_answer_is_failed_and_counts_against_the_gate() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_staleness().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(staleness_case_stubs(case, &mut ids));
    }
    let target = &cases[0];
    let stale = fact_json(
        Uuid::from_u128(u128::MAX - 1),
        &format!("{} then {}", target.old_values[0], target.new_values[0]),
        "active",
        "agent",
        &[],
    );
    stubs.push(recall_stub(
        &target.id,
        OVERRIDE_PRIORITY,
        false,
        &scored(vec![stale]),
    ));
    wiremock.import(stubs).await;

    let run = run_harness(
        &wiremock.url,
        &["--suite", "staleness", "--min-staleness", "1.0"],
    )
    .await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let staleness = suite(&run.report, "staleness");
    let failed = case(staleness, &target.id);
    assert_eq!(failed["status"], json!("failed"));
    assert_eq!(failed["detail"]["old_absent_from_current"], json!(false));
    assert_eq!(staleness["cases_errored"], json!(0));
}

#[tokio::test]
async fn candidates_that_miss_the_old_fact_fail_without_a_supersede() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_staleness().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(staleness_case_stubs(case, &mut ids));
    }
    let target = &cases[0];
    let mut empty = candidates_stub(AGENT_TOKEN, &target.id, &scored(Vec::new()));
    empty["priority"] = json!(OVERRIDE_PRIORITY);
    stubs.push(empty);
    wiremock.import(stubs).await;

    let run = run_harness(
        &wiremock.url,
        &["--suite", "staleness", "--min-staleness", "0"],
    )
    .await;

    assert_eq!(run.output.status.code(), Some(0), "{}", stdout(&run));
    let staleness = suite(&run.report, "staleness");
    let failed = case(staleness, &target.id);
    assert_eq!(failed["status"], json!("failed"));
    assert_eq!(failed["detail"]["candidate_found"], json!(false));
    assert_eq!(staleness["cases_failed"], json!(1));
}

#[tokio::test]
async fn a_purge_that_leaves_the_episode_behind_fails() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_poisoning().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(poison_case_stubs(case, &mut ids));
    }
    let poison = ids.of(&format!("{}/held", cases[0].id));
    stubs.push(json!({
        "priority": OVERRIDE_PRIORITY,
        "request": { "method": "GET", "urlPath": format!("/v1/episodes/{poison}") },
        "response": json_response(200, &json!({ "id": poison })),
    }));
    wiremock.import(stubs).await;

    let run = run_harness(&wiremock.url, &["--suite", "poisoning"]).await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let failed = case(suite(&run.report, "poisoning"), &cases[0].id);
    assert_eq!(failed["status"], json!("failed"));
    assert!(
        failed["reason"]
            .as_str()
            .expect("reason")
            .contains("after purge"),
        "{failed}"
    );
}
