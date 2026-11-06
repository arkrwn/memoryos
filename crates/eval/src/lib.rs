//! Black-box evaluation harness for MemoryOS.
//!
//! Talks to a running server only through the HTTP contract in `docs/openapi.yaml`, and scores it
//! against the production gates in `docs/SPEC.md` section 12.

pub mod client;
pub mod dataset;
pub mod report;
pub mod score;
pub mod suites;

use std::path::PathBuf;

use chrono::Utc;
use clap::{Parser, ValueEnum};

use client::Client;
use report::{Report, SuiteReport};
use suites::RunContext;

#[derive(Parser)]
#[command(
    name = "memoryos-eval",
    about = "Black-box recall, staleness and poisoning evaluation of a MemoryOS server"
)]
pub struct Args {
    #[arg(long)]
    pub base_url: String,
    /// Bearer token allowed to use `user:*` scopes with `max_trust` of at least `user`.
    #[arg(long)]
    pub api_key: String,
    /// Bearer token with `is_admin`, used for quarantine lookup and purge.
    #[arg(long)]
    pub admin_api_key: String,
    #[arg(long, value_enum)]
    pub suite: Suite,
    #[arg(long)]
    pub out: PathBuf,
    /// How long one episode may take to be processed before the case counts as an error.
    #[arg(long, default_value_t = 180, value_parser = clap::value_parser!(u64).range(1..))]
    pub timeout_secs: u64,
    #[arg(long, default_value_t = 0.95, value_parser = parse_ratio)]
    pub min_staleness: f64,
    /// Minimum recall@5. SPEC section 12 sets no target, so by default recall is only reported.
    #[arg(long, default_value_t = 0.0, value_parser = parse_ratio)]
    pub min_recall: f64,
    /// Cases run at the same time within a suite.
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u64).range(1..))]
    pub concurrency: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Suite {
    Recall,
    Staleness,
    Poisoning,
    All,
}

fn parse_ratio(raw: &str) -> Result<f64, String> {
    let value: f64 = raw.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!("{value} is not between 0 and 1"))
    }
}

/// Runs the selected suites and writes the report. `Report::passed` says whether every gate held.
pub async fn run(args: &Args) -> Result<Report, Box<dyn std::error::Error>> {
    let started_at = Utc::now();
    let client = Client::new(
        &args.base_url,
        &args.api_key,
        &args.admin_api_key,
        args.timeout_secs,
    )?;
    let ctx = RunContext::new(client);
    let concurrency = usize::try_from(args.concurrency)?;

    let mut suites: Vec<SuiteReport> = Vec::new();
    if matches!(args.suite, Suite::Recall | Suite::All) {
        let cases = dataset::load_recall()?;
        let results = suites::run_recall(&ctx, &cases, concurrency).await;
        suites.push(score::recall_report(&results, args.min_recall)?);
    }
    if matches!(args.suite, Suite::Staleness | Suite::All) {
        let cases = dataset::load_staleness()?;
        let results = suites::run_staleness(&ctx, &cases, concurrency).await;
        suites.push(score::staleness_report(&results, args.min_staleness)?);
    }
    if matches!(args.suite, Suite::Poisoning | Suite::All) {
        let cases = dataset::load_poisoning()?;
        let results = suites::run_poisoning(&ctx, &cases, concurrency).await;
        suites.push(score::poisoning_report(&results)?);
    }

    let report = Report::new(
        ctx.run_id.clone(),
        args.base_url.clone(),
        started_at,
        Utc::now(),
        suites,
    );
    report.write(&args.out)?;
    Ok(report)
}
