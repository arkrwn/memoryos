use std::process::ExitCode;

use clap::Parser;
use memoryos_server::cli::{self, Cli};

fn main() -> ExitCode {
    match cli::dispatch(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("memoryos: {message}");
            ExitCode::FAILURE
        }
    }
}
