//! Runs the real `memoryos-eval` binary against WireMock stubs generated from the real datasets.
//!
//! Stubs require the correct bearer token, so a call with the wrong key gets WireMock's 404 and
//! the case errors. Poisoning stubs use a scenario per case, so after the purge the episode and
//! fact return 404 and the quarantine list empties, as on a real server.
#![cfg(feature = "integration")]

use std::collections::HashMap;
use std::process::Output;
use std::time::Duration;

use memoryos_eval::dataset::{self, PoisonCase, RecallCase, StalenessCase};
use serde_json::{json, Value};
use testcontainers::core::{IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, ImageExt};
use uuid::Uuid;

const USER_TOKEN: &str = "mos_user_token";
const ADMIN_TOKEN: &str = "mos_admin_token";
const OVERRIDE_PRIORITY: u32 = 1;
const AS_OF_PRIORITY: u32 = 2;
const DEFAULT_PRIORITY: u32 = 5;
const NOW: &str = "2026-01-01T00:00:00Z";

struct Wiremock {
    _container: ContainerAsync<GenericImage>,
    url: String,
}

async fn start_wiremock() -> Wiremock {
    let container = GenericImage::new("wiremock/wiremock", "3.3.1")
        .with_exposed_port(8080.tcp())
        .with_wait_for(WaitFor::message_on_stdout("port:"))
        .with_startup_timeout(Duration::from_secs(120))
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
}

/// Deterministic ids, so each episode's stubs can reference one another.
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

fn episode_json(id: Uuid, processed: bool) -> Value {
    json!({
        "id": id,
        "scope": "user:stub",
        "source": "stub",
        "trust": "user",
        "content": "stub",
        "metadata": {},
        "occurred_at": NOW,
        "created_at": NOW,
        "processed_at": if processed { json!(NOW) } else { Value::Null },
    })
}

fn fact_json(id: Uuid, statement: &str, status: &str, trust: &str, sources: &[Uuid]) -> Value {
    json!({
        "id": id,
        "scope": "user:stub",
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
    json!({ "facts": facts, "pending_episodes": [] })
}

fn json_response(status: u16, body: &Value) -> Value {
    json!({
        "status": status,
        "jsonBody": body,
        "headers": { "Content-Type": "application/json" },
    })
}

/// `POST /v1/episodes` for one episode, selected by its idempotency key suffix.
fn post_episode_stub(case_id: &str, label: &str, id: Uuid) -> Value {
    json!({
        "priority": DEFAULT_PRIORITY,
        "request": {
            "method": "POST",
            "urlPath": "/v1/episodes",
            "headers": auth(USER_TOKEN),
            "bodyPatterns": [{ "matchesJsonPath": {
                "expression": "$.idempotency_key",
                "matches": format!(".*-{case_id}-{label}$"),
            }}],
        },
        "response": json_response(202, &episode_json(id, false)),
    })
}

fn get_episode_stub(id: Uuid) -> Value {
    json!({
        "priority": DEFAULT_PRIORITY,
        "request": {
            "method": "GET",
            "urlPath": format!("/v1/episodes/{id}"),
            "headers": auth(USER_TOKEN),
        },
        "response": json_response(200, &episode_json(id, true)),
    })
}

fn recall_stub(case_id: &str, priority: u32, as_of: bool, body: &Value) -> Value {
    let mut patterns = vec![json!({ "matchesJsonPath": {
        "expression": "$.scopes[0]",
        "matches": format!(".*-{case_id}$"),
    }})];
    if as_of {
        patterns.push(json!({ "matchesJsonPath": "$.as_of" }));
    }
    json!({
        "priority": priority,
        "request": {
            "method": "POST",
            "urlPath": "/v1/recall",
            "headers": auth(USER_TOKEN),
            "bodyPatterns": patterns,
        },
        "response": json_response(200, body),
    })
}

fn recall_case_stubs(case: &RecallCase, ids: &mut Ids) -> Vec<Value> {
    let mut stubs = Vec::new();
    for index in 0..case.episodes.len() {
        let label = index.to_string();
        let id = ids.of(&format!("{}/{label}", case.id));
        stubs.push(post_episode_stub(&case.id, &label, id));
        stubs.push(get_episode_stub(id));
    }
    let facts = case
        .expected
        .iter()
        .map(|value| {
            let id = ids.of(&format!("{}/fact/{value}", case.id));
            fact_json(id, &format!("Recalled: {value}"), "active", "user", &[])
        })
        .collect();
    stubs.push(recall_stub(
        &case.id,
        DEFAULT_PRIORITY,
        false,
        &scored(facts),
    ));
    stubs
}

fn staleness_case_stubs(case: &StalenessCase, ids: &mut Ids) -> Vec<Value> {
    let mut stubs = Vec::new();
    for label in ["earlier", "later"] {
        let id = ids.of(&format!("{}/{label}", case.id));
        stubs.push(post_episode_stub(&case.id, label, id));
        stubs.push(get_episode_stub(id));
    }
    let statement = |values: &[String]| format!("Value is {}", values.join(" and "));
    let current = fact_json(
        ids.of(&format!("{}/new", case.id)),
        &statement(&case.new_values),
        "active",
        "user",
        &[],
    );
    let past = fact_json(
        ids.of(&format!("{}/old", case.id)),
        &statement(&case.old_values),
        "active",
        "user",
        &[],
    );
    stubs.push(recall_stub(
        &case.id,
        DEFAULT_PRIORITY,
        false,
        &scored(vec![current]),
    ));
    stubs.push(recall_stub(
        &case.id,
        AS_OF_PRIORITY,
        true,
        &scored(vec![past]),
    ));
    stubs
}

fn poison_case_stubs(case: &PoisonCase, ids: &mut Ids) -> Vec<Value> {
    let seed = ids.of(&format!("{}/seed", case.id));
    let poison = ids.of(&format!("{}/poison", case.id));
    let scenario = format!("purge-{}", case.id);

    let true_fact = fact_json(
        ids.of(&format!("{}/true", case.id)),
        &format!("True value is {}", case.true_values.join(" and ")),
        "active",
        "user",
        &[seed],
    );
    let quarantined = fact_json(
        ids.of(&format!("{}/quarantined", case.id)),
        &format!("Claim: {}", case.planted.join(" and ")),
        "quarantined",
        "external",
        &[poison],
    );

    vec![
        post_episode_stub(&case.id, "seed", seed),
        get_episode_stub(seed),
        post_episode_stub(&case.id, "poison", poison),
        recall_stub(&case.id, DEFAULT_PRIORITY, false, &scored(vec![true_fact])),
        // Reachable only until the purge. Afterwards WireMock answers 404, as a server would.
        json!({
            "scenarioName": scenario,
            "requiredScenarioState": "Started",
            "request": {
                "method": "GET",
                "urlPath": format!("/v1/episodes/{poison}"),
                "headers": auth(USER_TOKEN),
            },
            "response": json_response(200, &episode_json(poison, true)),
        }),
        json!({
            "scenarioName": scenario,
            "requiredScenarioState": "Started",
            "request": {
                "method": "GET",
                "urlPath": "/v1/quarantine",
                "headers": auth(ADMIN_TOKEN),
                "queryParameters": { "scopes": { "matches": format!(".*-{}$", case.id) } },
            },
            "response": json_response(200, &json!({ "facts": [quarantined] })),
        }),
        json!({
            "scenarioName": scenario,
            "requiredScenarioState": "Started",
            "newScenarioState": "purged",
            "request": {
                "method": "POST",
                "urlPath": format!("/v1/episodes/{poison}/purge"),
                "headers": auth(ADMIN_TOKEN),
            },
            "response": json_response(200, &json!({
                "facts_deleted": 1,
                "episodes_deleted": 1,
                "entities_deleted": 0,
            })),
        }),
        json!({
            "scenarioName": scenario,
            "requiredScenarioState": "purged",
            "request": {
                "method": "GET",
                "urlPath": "/v1/quarantine",
                "headers": auth(ADMIN_TOKEN),
                "queryParameters": { "scopes": { "matches": format!(".*-{}$", case.id) } },
            },
            "response": json_response(200, &json!({ "facts": [] })),
        }),
    ]
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

async fn run_eval(url: &str, extra: &[&str]) -> Run {
    let dir = tempfile::tempdir().expect("temp dir");
    let out = dir.path().join("report.json");
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_memoryos-eval"))
        .args(["--base-url", url])
        .args(["--api-key", USER_TOKEN])
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
async fn a_correct_server_passes_every_suite_and_exits_zero() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    wiremock.import(healthy_server(&mut ids)).await;

    let run = run_eval(&wiremock.url, &["--suite", "all", "--concurrency", "16"]).await;

    assert_eq!(run.output.status.code(), Some(0), "{}", stdout(&run));
    assert_eq!(run.report["passed"], json!(true));

    let recall = suite(&run.report, "recall");
    let recall_cases = dataset::load_recall().expect("data");
    let recall_total = recall_cases.len();
    assert_eq!(recall["total"], json!(recall_total));
    assert_eq!(recall["cases_passed"], json!(recall_total));
    assert_eq!(recall["metrics"]["recall_at_5"], json!(1.0));
    // The stubs return one fact per expected value, so a set is covered at rank `expected.len()`.
    let expected_mrr = recall_cases
        .iter()
        .map(|c| 1.0 / c.expected.len() as f64)
        .sum::<f64>()
        / recall_total as f64;
    let mrr = recall["metrics"]["mrr"].as_f64().expect("mrr");
    assert!((mrr - expected_mrr).abs() < 1e-9, "{mrr} vs {expected_mrr}");

    let staleness = suite(&run.report, "staleness");
    assert_eq!(staleness["metrics"]["pass_rate"], json!(1.0));
    assert_eq!(staleness["cases_errored"], json!(0));

    let poisoning = suite(&run.report, "poisoning");
    let poison_total = dataset::load_poisoning().expect("data").len();
    assert_eq!(poisoning["metrics"]["leaks"], json!(0));
    assert_eq!(
        poisoning["metrics"]["purge_exercised_cases"],
        json!(poison_total)
    );

    let table = stdout(&run);
    assert!(table.contains("result: PASS"), "{table}");
    let report_text = run.report.to_string();
    assert!(!report_text.contains(USER_TOKEN) && !report_text.contains(ADMIN_TOKEN));
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
        "user",
        &[ids.of(&format!("{}/seed", target.id))],
    );
    stubs.push(recall_stub(
        &target.id,
        OVERRIDE_PRIORITY,
        false,
        &scored(vec![leaked]),
    ));
    wiremock.import(stubs).await;

    let run = run_eval(&wiremock.url, &["--suite", "poisoning"]).await;

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
async fn an_episode_that_never_processes_is_an_error_not_a_pass() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_recall().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(recall_case_stubs(case, &mut ids));
    }
    let stuck = ids.of(&format!("{}/0", cases[0].id));
    stubs.push(json!({
        "priority": OVERRIDE_PRIORITY,
        "request": { "method": "GET", "urlPath": format!("/v1/episodes/{stuck}") },
        "response": json_response(200, &episode_json(stuck, false)),
    }));
    wiremock.import(stubs).await;

    let run = run_eval(
        &wiremock.url,
        &[
            "--suite",
            "recall",
            "--timeout-secs",
            "2",
            "--concurrency",
            "16",
        ],
    )
    .await;

    assert_eq!(run.output.status.code(), Some(1), "{}", stdout(&run));
    let recall = suite(&run.report, "recall");
    let timed_out = case(recall, &cases[0].id);
    assert_eq!(timed_out["status"], json!("error"));
    assert!(
        timed_out["reason"]
            .as_str()
            .expect("reason")
            .contains("timeout"),
        "{timed_out}"
    );
    assert_eq!(recall["cases_errored"], json!(1));
    assert_eq!(recall["cases_passed"], json!(cases.len() - 1));
    let expected = (cases.len() - 1) as f64 / cases.len() as f64;
    assert_eq!(recall["metrics"]["recall_at_5"], json!(expected));
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
            "bodyPatterns": [{ "matchesJsonPath": {
                "expression": "$.scopes[0]",
                "matches": format!(".*-{}$", cases[0].id),
            }}],
        },
        "response": json_response(500, &json!({
            "error": { "code": "internal", "message": "boom" }
        })),
    }));
    wiremock.import(stubs).await;

    let run = run_eval(
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
        "user",
        &[],
    );
    stubs.push(recall_stub(
        &target.id,
        OVERRIDE_PRIORITY,
        false,
        &scored(vec![stale]),
    ));
    wiremock.import(stubs).await;

    let run = run_eval(
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
async fn a_purge_that_leaves_the_episode_behind_fails() {
    let wiremock = start_wiremock().await;
    let mut ids = Ids::default();
    let cases = dataset::load_poisoning().expect("data");
    let mut stubs = Vec::new();
    for case in &cases {
        stubs.extend(poison_case_stubs(case, &mut ids));
    }
    let poison = ids.of(&format!("{}/poison", cases[0].id));
    stubs.push(json!({
        "priority": OVERRIDE_PRIORITY,
        "request": { "method": "GET", "urlPath": format!("/v1/episodes/{poison}") },
        "response": json_response(200, &episode_json(poison, true)),
    }));
    wiremock.import(stubs).await;

    let run = run_eval(&wiremock.url, &["--suite", "poisoning"]).await;

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
