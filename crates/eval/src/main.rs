use std::process::ExitCode;

use clap::Parser;
use memoryos_eval::{run, Args};

const EXIT_GATE_FAILED: u8 = 1;
const EXIT_HARNESS_ERROR: u8 = 2;

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args).await {
        Ok(report) => {
            print!("{}", report.summary_table());
            println!("report: {}", args.out.display());
            if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(EXIT_GATE_FAILED)
            }
        }
        Err(e) => {
            eprintln!("memoryos-eval: {e}");
            ExitCode::from(EXIT_HARNESS_ERROR)
        }
    }
}
