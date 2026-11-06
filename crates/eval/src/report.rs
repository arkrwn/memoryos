//! The JSON report and the printed summary table. API keys never reach either.

use std::fmt::Write as _;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;

pub const REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct Gate {
    pub name: String,
    pub actual: f64,
    /// `at_least` or `at_most`.
    pub comparison: &'static str,
    pub threshold: f64,
    pub passed: bool,
}

impl Gate {
    pub fn at_least(name: impl Into<String>, actual: f64, threshold: f64) -> Self {
        Self {
            name: name.into(),
            actual,
            comparison: "at_least",
            threshold,
            passed: actual >= threshold,
        }
    }

    pub fn at_most(name: impl Into<String>, actual: f64, threshold: f64) -> Self {
        Self {
            name: name.into(),
            actual,
            comparison: "at_most",
            threshold,
            passed: actual <= threshold,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SuiteReport {
    pub suite: String,
    pub dataset: String,
    pub passed: bool,
    pub total: usize,
    pub cases_passed: usize,
    pub cases_failed: usize,
    pub cases_errored: usize,
    pub metrics: Value,
    pub gates: Vec<Gate>,
    pub cases: Vec<Value>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: usize,
}

impl SuiteReport {
    pub fn new(
        suite: &str,
        dataset: &str,
        counts: Counts,
        metrics: Value,
        gates: Vec<Gate>,
        cases: Vec<Value>,
    ) -> Self {
        // A suite that ran nothing proved nothing.
        let passed = counts.total > 0 && gates.iter().all(|gate| gate.passed);
        Self {
            suite: suite.to_owned(),
            dataset: dataset.to_owned(),
            passed,
            total: counts.total,
            cases_passed: counts.passed,
            cases_failed: counts.failed,
            cases_errored: counts.errors,
            metrics,
            gates,
            cases,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub run_id: String,
    pub base_url: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub passed: bool,
    pub suites: Vec<SuiteReport>,
}

impl Report {
    pub fn new(
        run_id: String,
        base_url: String,
        started_at: DateTime<Utc>,
        finished_at: DateTime<Utc>,
        suites: Vec<SuiteReport>,
    ) -> Self {
        let passed = !suites.is_empty() && suites.iter().all(|suite| suite.passed);
        Self {
            schema_version: REPORT_SCHEMA_VERSION,
            run_id,
            base_url,
            started_at,
            finished_at,
            passed,
            suites,
        }
    }

    pub fn write(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn summary_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<10} {:>5} {:>5} {:>5} {:>5}  {:<5}  metrics",
            "suite", "cases", "pass", "fail", "error", "gate"
        );
        for suite in &self.suites {
            let _ = writeln!(
                out,
                "{:<10} {:>5} {:>5} {:>5} {:>5}  {:<5}  {}",
                suite.suite,
                suite.total,
                suite.cases_passed,
                suite.cases_failed,
                suite.cases_errored,
                if suite.passed { "PASS" } else { "FAIL" },
                metrics_line(&suite.metrics),
            );
        }
        for gate in self
            .suites
            .iter()
            .flat_map(|s| &s.gates)
            .filter(|g| !g.passed)
        {
            let _ = writeln!(
                out,
                "gate failed: {} = {} ({} {})",
                gate.name,
                format_number(gate.actual),
                gate.comparison,
                format_number(gate.threshold),
            );
        }
        let _ = writeln!(out, "result: {}", if self.passed { "PASS" } else { "FAIL" });
        out
    }
}

fn metrics_line(metrics: &Value) -> String {
    let Some(map) = metrics.as_object() else {
        return String::new();
    };
    map.iter()
        .map(|(key, value)| match value.as_f64() {
            Some(number) => format!("{key}={}", format_number(number)),
            None => format!("{key}={value}"),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn suite(gates: Vec<Gate>, total: usize) -> SuiteReport {
        SuiteReport::new(
            "recall",
            "recall.v1",
            Counts {
                total,
                passed: total,
                ..Counts::default()
            },
            json!({ "recall_at_5": 0.9 }),
            gates,
            Vec::new(),
        )
    }

    fn report(suites: Vec<SuiteReport>) -> Report {
        Report::new(
            "run".to_owned(),
            "http://x".to_owned(),
            Utc::now(),
            Utc::now(),
            suites,
        )
    }

    #[test]
    fn gate_comparisons_include_the_threshold() {
        assert!(Gate::at_least("a", 0.95, 0.95).passed);
        assert!(!Gate::at_least("a", 0.94, 0.95).passed);
        assert!(Gate::at_most("a", 0.0, 0.0).passed);
        assert!(!Gate::at_most("a", 1.0, 0.0).passed);
    }

    #[test]
    fn one_failed_gate_fails_the_report() {
        let bad = suite(vec![Gate::at_most("x", 1.0, 0.0)], 1);
        let good = suite(vec![Gate::at_most("x", 0.0, 0.0)], 1);
        assert!(!report(vec![good, bad]).passed);
    }

    #[test]
    fn an_empty_report_or_suite_does_not_pass() {
        assert!(!report(Vec::new()).passed);
        assert!(!suite(Vec::new(), 0).passed);
    }

    #[test]
    fn summary_names_failed_gates() {
        let table = report(vec![suite(
            vec![Gate::at_most("recall.errors", 2.0, 0.0)],
            1,
        )])
        .summary_table();
        assert!(table.contains("gate failed: recall.errors = 2"), "{table}");
        assert!(table.contains("result: FAIL"), "{table}");
        assert!(table.contains("recall_at_5=0.900"), "{table}");
    }
}
