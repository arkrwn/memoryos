//! Times the real `Retriever::recall`, with an embedder that returns a stored vector at once, so
//! embedding time is left out as the gate states. Vector and text search are timed alone as well,
//! through the same `PgStore`, to show where the time goes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Duration as Days, Utc};
use memoryos_core::{Embedder, FactFilter, MemoryStore, RequestContext, Result as CoreResult, Scope};
use memoryos_retrieve::{RecallRequest, Retriever};
use memoryos_store::PgStore;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::data::{self, Vocab, HUB_SCOPE, HUB_TENANT};
use crate::seed::{database_url, timeline_end};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const SAMPLED_FACTS: i64 = 300;
const WARM_OPS: usize = 30;

pub struct Options {
    pub admin_url: String,
    pub database: String,
    pub shape: String,
    pub ops: usize,
    pub seconds: u64,
    pub pool: u32,
    pub concurrency: Vec<usize>,
    pub only: Option<String>,
    pub label: String,
    pub out: Option<String>,
}

/// Returns stored vectors in turn. The recorded latency excludes any model.
struct StoredVectors {
    vectors: Vec<Vec<f32>>,
    next: AtomicUsize,
}

#[async_trait]
impl Embedder for StoredVectors {
    async fn embed(&self, inputs: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        let at = self.next.fetch_add(1, Ordering::Relaxed) % self.vectors.len();
        Ok(inputs.iter().map(|_| self.vectors[at].clone()).collect())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Recall,
    Vector,
    Text,
    Neighbors,
}

#[derive(Clone)]
struct Query {
    text: String,
    vector: Vec<f32>,
    /// For `Op::Neighbors`.
    root: Option<Uuid>,
}

struct Scenario {
    name: String,
    op: Op,
    tenant: String,
    scopes: Vec<Scope>,
    queries: Vec<Query>,
    as_of: Option<DateTime<Utc>>,
    known_at: Option<DateTime<Utc>>,
    /// Run at every concurrency level, not only at 1.
    full: bool,
}

/// A statement and vector of a fact in a scope.
struct Sample {
    statement: String,
    vector: Vec<f32>,
}

async fn sample_scope(pool: &PgPool, scope: &str, count: i64) -> Res<Vec<Sample>> {
    let rows = sqlx::query(
        "SELECT statement, embedding::text AS vector FROM facts \
         WHERE scope = $1 AND status = 'active' ORDER BY md5(id::text) LIMIT $2",
    )
    .bind(scope)
    .bind(count)
    .fetch_all(pool)
    .await?;
    let mut samples = Vec::new();
    for row in rows {
        let text: String = row.try_get("vector")?;
        let vector = text
            .trim_matches(['[', ']'])
            .split(',')
            .map(str::parse::<f32>)
            .collect::<Result<Vec<_>, _>>()?;
        samples.push(Sample {
            statement: row.try_get("statement")?,
            vector,
        });
    }
    if samples.is_empty() {
        return Err(format!("no facts in {scope}").into());
    }
    Ok(samples)
}

/// A fact's vector with noise: a question close to that fact, not identical to it.
fn query_vector(sample: &Sample, rng: &mut SmallRng) -> Vec<f32> {
    let sigma = 0.5 / (data::DIMS as f32).sqrt();
    let mut vector: Vec<f32> = sample
        .vector
        .iter()
        .map(|value| value + sigma * data::gaussian(rng))
        .collect();
    data::normalize(&mut vector);
    vector
}

/// Text queries by word frequency. `common1` is a top-5 word in up to 70% of facts.
fn queries(class: &str, samples: &[Sample], vocab: &Vocab, rng: &mut SmallRng) -> Vec<Query> {
    samples
        .iter()
        .map(|sample| {
            let words: Vec<&str> = sample.statement.split_whitespace().collect();
            let text = match class {
                // The fact's rarest word, so the query has a hit and few others.
                "rare" => words
                    .iter()
                    .max_by_key(|word| vocab.rank(word))
                    .map(|word| (*word).to_owned())
                    .unwrap_or_default(),
                "common1" => vocab.word(rng.gen_range(0..5)).to_owned(),
                "common2" => format!(
                    "{} {}",
                    vocab.word(rng.gen_range(0..10)),
                    vocab.word(rng.gen_range(0..10))
                ),
                // Several of the fact's words in order, common and rare alike.
                "natural" => {
                    let take = words.len().min(5);
                    let from = rng.gen_range(0..=words.len() - take);
                    words[from..from + take].join(" ")
                }
                other => panic!("unknown class {other}"),
            };
            Query {
                text,
                vector: query_vector(sample, rng),
                root: None,
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn scenario(
    name: &str,
    op: Op,
    tenant: &str,
    scopes: &[&str],
    queries: Vec<Query>,
    as_of: Option<DateTime<Utc>>,
    known_at: Option<DateTime<Utc>>,
    full: bool,
) -> Res<Scenario> {
    Ok(Scenario {
        name: name.to_owned(),
        op,
        tenant: tenant.to_owned(),
        scopes: scopes.iter().map(|s| s.parse()).collect::<Result<_, _>>()?,
        queries,
        as_of,
        known_at,
        full,
    })
}

/// Adds recall at every concurrency level and text and vector alone at 1, for each class.
fn add_scope_scenarios(
    list: &mut Vec<Scenario>,
    label: &str,
    tenant: &str,
    scope: &str,
    samples: &[Sample],
    vocab: &Vocab,
    rng: &mut SmallRng,
) -> Res<()> {
    let classes = ["rare", "common1", "common2", "natural"];
    let year_ago = timeline_end() - Days::days(365);
    for class in classes {
        let set = queries(class, samples, vocab, rng);
        list.push(scenario(
            &format!("{label} recall {class}"),
            Op::Recall,
            tenant,
            &[scope],
            set.clone(),
            None,
            None,
            true,
        )?);
        list.push(scenario(
            &format!("{label} text-only {class}"),
            Op::Text,
            tenant,
            &[scope],
            set,
            None,
            None,
            false,
        )?);
    }
    let set = queries("rare", samples, vocab, rng);
    list.push(scenario(
        &format!("{label} vector-only"),
        Op::Vector,
        tenant,
        &[scope],
        set.clone(),
        None,
        None,
        false,
    )?);
    list.push(scenario(
        &format!("{label} vector-only as_of"),
        Op::Vector,
        tenant,
        &[scope],
        set.clone(),
        Some(year_ago),
        None,
        false,
    )?);
    list.push(scenario(
        &format!("{label} recall as_of rare"),
        Op::Recall,
        tenant,
        &[scope],
        set,
        Some(year_ago),
        None,
        true,
    )?);
    let common = queries("common1", samples, vocab, rng);
    list.push(scenario(
        &format!("{label} recall as_of common1"),
        Op::Recall,
        tenant,
        &[scope],
        common.clone(),
        Some(year_ago),
        None,
        true,
    )?);
    list.push(scenario(
        &format!("{label} text-only as_of common1"),
        Op::Text,
        tenant,
        &[scope],
        common.clone(),
        Some(year_ago),
        None,
        false,
    )?);
    list.push(scenario(
        &format!("{label} recall known_at common1"),
        Op::Recall,
        tenant,
        &[scope],
        common,
        None,
        Some(year_ago),
        false,
    )?);
    Ok(())
}

/// The largest, a middle and a small scope of shape `a`, with their tenant.
async fn scope_at_rank(pool: &PgPool, rank: i64) -> Res<(String, String, i64)> {
    let row = sqlx::query(
        "SELECT tenant, scope, count(*) AS n FROM facts GROUP BY tenant, scope \
         ORDER BY n DESC, scope OFFSET $1 LIMIT 1",
    )
    .bind(rank)
    .fetch_one(pool)
    .await?;
    Ok((row.try_get("tenant")?, row.try_get("scope")?, row.try_get("n")?))
}

async fn scenarios(pool: &PgPool, shape: &str) -> Res<Vec<Scenario>> {
    let vocab = Vocab::new();
    let mut rng = SmallRng::seed_from_u64(7);
    let mut list = Vec::new();
    match shape {
        "a" => {
            let mut three = Vec::new();
            for (label, rank) in [("a-large", 0), ("a-mid", 100), ("a-small", 800)] {
                let (tenant, scope, count) = scope_at_rank(pool, rank).await?;
                println!("{label}: {scope} in {tenant}, {count} facts");
                let samples = sample_scope(pool, &scope, SAMPLED_FACTS).await?;
                add_scope_scenarios(&mut list, label, &tenant, &scope, &samples, &vocab, &mut rng)?;
                three.push(tenant);
            }
            // A key spanning several scopes, within one tenant.
            let scopes: Vec<(String, i64)> = sqlx::query_as(
                "SELECT scope, count(*) FROM facts WHERE tenant = $1 GROUP BY scope ORDER BY 2 DESC LIMIT 5",
            )
            .bind(&three[0])
            .fetch_all(pool)
            .await?;
            let names: Vec<&str> = scopes.iter().map(|(name, _)| name.as_str()).collect();
            let samples = sample_scope(pool, names[0], SAMPLED_FACTS).await?;
            for class in ["rare", "common1"] {
                list.push(scenario(
                    &format!("a-5-scopes recall {class}"),
                    Op::Recall,
                    &three[0],
                    &names,
                    queries(class, &samples, &vocab, &mut rng),
                    None,
                    None,
                    true,
                )?);
            }
        }
        "b" => {
            let samples = sample_scope(pool, "user:big", SAMPLED_FACTS).await?;
            add_scope_scenarios(&mut list, "b-big", "t-big", "user:big", &samples, &vocab, &mut rng)?;
        }
        "c" => {
            // The hub's own facts, so the hub is among the subjects recall expands.
            let hub: Uuid = sqlx::query_scalar("SELECT id FROM entities WHERE scope = $1 AND name = 'hub'")
                .bind(HUB_SCOPE)
                .fetch_one(pool)
                .await?;
            let samples = sample_hub(pool, hub).await?;
            let year_ago = timeline_end() - Days::days(365);
            for class in ["rare", "common1", "natural"] {
                let set = queries(class, &samples, &vocab, &mut rng);
                list.push(scenario(
                    &format!("c-hub recall {class}"),
                    Op::Recall,
                    HUB_TENANT,
                    &[HUB_SCOPE],
                    set.clone(),
                    None,
                    None,
                    true,
                )?);
                list.push(scenario(
                    &format!("c-hub recall as_of {class}"),
                    Op::Recall,
                    HUB_TENANT,
                    &[HUB_SCOPE],
                    set,
                    Some(year_ago),
                    None,
                    true,
                )?);
            }
            let walk = vec![Query {
                text: String::new(),
                vector: Vec::new(),
                root: Some(hub),
            }];
            list.push(scenario(
                "c-hub neighbors depth 1",
                Op::Neighbors,
                HUB_TENANT,
                &[HUB_SCOPE],
                walk.clone(),
                None,
                None,
                false,
            )?);
            list.push(scenario(
                "c-hub neighbors depth 1 as_of",
                Op::Neighbors,
                HUB_TENANT,
                &[HUB_SCOPE],
                walk,
                Some(year_ago),
                None,
                false,
            )?);
            let set = queries("rare", &samples, &vocab, &mut rng);
            list.push(scenario(
                "c-hub vector-only",
                Op::Vector,
                HUB_TENANT,
                &[HUB_SCOPE],
                set.clone(),
                None,
                None,
                false,
            )?);
            list.push(scenario(
                "c-hub text-only rare",
                Op::Text,
                HUB_TENANT,
                &[HUB_SCOPE],
                set,
                None,
                None,
                false,
            )?);
        }
        other => return Err(format!("unknown shape {other}").into()),
    }
    Ok(list)
}

async fn sample_hub(pool: &PgPool, hub: Uuid) -> Res<Vec<Sample>> {
    let rows = sqlx::query(
        "SELECT statement, embedding::text AS vector FROM facts \
         WHERE subject_id = $1 AND status = 'active' ORDER BY md5(id::text) LIMIT $2",
    )
    .bind(hub)
    .bind(SAMPLED_FACTS)
    .fetch_all(pool)
    .await?;
    let mut samples = Vec::new();
    for row in rows {
        let text: String = row.try_get("vector")?;
        let vector = text
            .trim_matches(['[', ']'])
            .split(',')
            .map(str::parse::<f32>)
            .collect::<Result<Vec<_>, _>>()?;
        samples.push(Sample {
            statement: row.try_get("statement")?,
            vector,
        });
    }
    Ok(samples)
}

struct Outcome {
    latencies: Vec<Duration>,
    errors: usize,
    /// Calls that returned no fact at all.
    empty: usize,
}

async fn run_once(
    retriever: &Retriever,
    store: &PgStore,
    scenario: &Scenario,
    index: usize,
) -> CoreResult<usize> {
    let ctx = RequestContext {
        tenant: scenario.tenant.clone(),
        actor: "loadtest".into(),
    };
    let query = &scenario.queries[index % scenario.queries.len()];
    let filter = FactFilter {
        as_of: scenario.as_of,
        known_at: scenario.known_at,
    };
    match scenario.op {
        Op::Recall => {
            let request = RecallRequest {
                scopes: scenario.scopes.clone(),
                query: query.text.clone(),
                limit: 10,
                as_of: scenario.as_of,
                known_at: scenario.known_at,
            };
            Ok(retriever.recall(&ctx, request).await?.facts.len())
        }
        Op::Vector => Ok(store
            .search_facts_vector(&ctx, &scenario.scopes, &query.vector, &filter, 50)
            .await?
            .len()),
        Op::Text => Ok(store
            .search_facts_text(&ctx, &scenario.scopes, &query.text, &filter, 50)
            .await?
            .len()),
        Op::Neighbors => {
            let root = query.root.expect("neighbors scenario has a root");
            Ok(store
                .neighbors(&ctx, &scenario.scopes, root, 1, &filter, 50)
                .await?
                .edges
                .len())
        }
    }
}

async fn run_level(
    retriever: Arc<Retriever>,
    store: Arc<PgStore>,
    scenario: Arc<Scenario>,
    concurrency: usize,
    ops: usize,
    budget: Duration,
) -> Outcome {
    let next = Arc::new(AtomicUsize::new(0));
    let deadline = Instant::now() + budget;
    let mut workers = Vec::new();
    for _ in 0..concurrency {
        let (retriever, store, scenario, next) =
            (retriever.clone(), store.clone(), scenario.clone(), next.clone());
        workers.push(tokio::spawn(async move {
            let mut latencies = Vec::new();
            let (mut errors, mut empty) = (0, 0);
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= ops || Instant::now() >= deadline {
                    break;
                }
                let started = Instant::now();
                let result = run_once(&retriever, &store, &scenario, index).await;
                latencies.push(started.elapsed());
                match result {
                    Ok(0) => empty += 1,
                    Ok(_) => {}
                    Err(error) => {
                        errors += 1;
                        eprintln!("{}: {error}", scenario.name);
                    }
                }
            }
            (latencies, errors, empty)
        }));
    }
    let mut outcome = Outcome {
        latencies: Vec::new(),
        errors: 0,
        empty: 0,
    };
    for worker in workers {
        if let Ok((latencies, errors, empty)) = worker.await {
            outcome.latencies.extend(latencies);
            outcome.errors += errors;
            outcome.empty += empty;
        }
    }
    outcome.latencies.sort();
    outcome
}

fn percentile(sorted: &[Duration], percent: usize) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (sorted.len() * percent).div_ceil(100).max(1) - 1;
    sorted[rank].as_secs_f64() * 1000.0
}

pub async fn run(options: &Options) -> Res<()> {
    let url = database_url(&options.admin_url, &options.database);
    let sampler = PgPoolOptions::new().max_connections(1).connect(&url).await?;
    let list = scenarios(&sampler, &options.shape).await?;

    let store = Arc::new(PgStore::connect(&url, options.pool).await?);
    let mut rng = SmallRng::seed_from_u64(11);
    let mut lines = Vec::new();
    println!("| label | scenario | concurrency | n | p50 ms | p95 ms | p99 ms | max ms | errors | empty |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut gate: BTreeMap<String, f64> = BTreeMap::new();
    for item in list {
        if options.only.as_ref().is_some_and(|only| !item.name.contains(only.as_str())) {
            continue;
        }
        let mut vectors: Vec<Vec<f32>> = item.queries.iter().map(|query| query.vector.clone()).collect();
        if vectors.iter().all(Vec::is_empty) {
            vectors = vec![data::random_unit(&mut rng)];
        }
        // Each level of a scenario sees the same pairs of text and vector, in the same order, at
        // concurrency 1. Above that the pairing shifts, which does not change their cost.
        let retriever = Arc::new(Retriever::new(
            store.clone() as Arc<dyn MemoryStore>,
            Arc::new(StoredVectors {
                vectors,
                next: AtomicUsize::new(0),
            }),
        ));
        let item = Arc::new(item);
        let warm = run_level(
            retriever.clone(),
            store.clone(),
            item.clone(),
            1,
            WARM_OPS,
            Duration::from_secs(options.seconds),
        )
        .await;
        drop(warm);
        let levels: Vec<usize> = if item.full {
            options.concurrency.clone()
        } else {
            vec![1]
        };
        for concurrency in levels {
            let outcome = run_level(
                retriever.clone(),
                store.clone(),
                item.clone(),
                concurrency,
                options.ops,
                Duration::from_secs(options.seconds),
            )
            .await;
            let (p50, p95, p99) = (
                percentile(&outcome.latencies, 50),
                percentile(&outcome.latencies, 95),
                percentile(&outcome.latencies, 99),
            );
            let max = outcome.latencies.last().map_or(f64::NAN, |d| d.as_secs_f64() * 1000.0);
            println!(
                "| {} | {} | {concurrency} | {} | {p50:.1} | {p95:.1} | {p99:.1} | {max:.1} | {} | {} |",
                options.label,
                item.name,
                outcome.latencies.len(),
                outcome.errors,
                outcome.empty,
            );
            lines.push(
                json!({
                    "label": options.label,
                    "scenario": item.name,
                    "concurrency": concurrency,
                    "n": outcome.latencies.len(),
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "max_ms": max,
                    "errors": outcome.errors,
                    "empty": outcome.empty,
                })
                .to_string(),
            );
            if item.op == Op::Recall {
                gate.insert(format!("{} at {concurrency}", item.name), p95);
            }
        }
    }
    let over: Vec<_> = gate.iter().filter(|(_, p95)| **p95 >= 200.0).collect();
    println!("\nrecall scenarios: {}, p95 at or over 200 ms: {}", gate.len(), over.len());
    for (name, p95) in over {
        println!("  {name}: {p95:.1} ms");
    }
    if let Some(path) = &options.out {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        for line in lines {
            writeln!(file, "{line}")?;
        }
    }
    Ok(())
}
