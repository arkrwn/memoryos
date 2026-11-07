//! `memoryos setup`, the agent hooks and their installer, the stdio MCP proxy, and the commands that
//! talk to a running server. They read `credentials.json`, with `MEMORYOS_URL` and
//! `MEMORYOS_API_KEY` as overrides. See `docs/SPEC.md` section 3.
//!
//! The server binary flattens [`Command`] into its own command line and calls [`run`].

mod client;
mod config;
mod credentials;
mod doc;
mod files;
mod hook;
mod install;
mod mcp;
mod setup;
mod turns;

use std::path::PathBuf;

use clap::{Args, Subcommand};
use memoryos_sdk::Scope;
use uuid::Uuid;

pub use install::Agent;

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Search memories.
    Recall {
        query: String,
        #[command(flatten)]
        scopes: Scopes,
        #[arg(long)]
        limit: Option<i64>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Show current facts related to a statement you may commit.
    Candidates {
        statement: String,
        #[command(flatten)]
        scopes: Scopes,
        #[arg(long)]
        limit: Option<i64>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Commit facts from a CommitRequest JSON document.
    Commit {
        /// File holding the JSON. Reads stdin when absent or `-`.
        file: Option<PathBuf>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Review quarantined facts. Needs an admin key.
    #[command(subcommand)]
    Quarantine(QuarantineCommand),
    /// Agent hooks and their installer.
    #[command(subcommand)]
    Hook(HookCommand),
    /// Start MemoryOS in Docker and connect an agent to it. Safe to run again.
    Setup(SetupArgs),
    /// Stdio MCP server for an agent, forwarding to the running server with the agent key.
    Mcp,
}

#[derive(Debug, Args)]
pub struct SetupArgs {
    /// The agent to connect. Without it, `--uninstall` covers every agent.
    #[arg(long, value_enum, required_unless_present = "uninstall")]
    pub agent: Option<Agent>,
    /// Use this project's configuration, in the current directory. This is the default.
    #[arg(long, conflicts_with = "global")]
    pub project: bool,
    /// Use the user's configuration under the home directory.
    #[arg(long)]
    pub global: bool,
    /// Remove the hooks and MCP configuration. Data stays unless `--delete-data` is given.
    #[arg(long)]
    pub uninstall: bool,
    /// With `--uninstall`: also delete the database, the model cache and the credentials.
    #[arg(long, requires = "uninstall")]
    pub delete_data: bool,
}

#[derive(Debug, Args)]
pub struct Scopes {
    /// Scope to search, like `user:u1`. Repeat it or separate with commas.
    #[arg(long = "scope", value_delimiter = ',', required = true)]
    pub scopes: Vec<Scope>,
}

#[derive(Debug, Subcommand)]
pub enum QuarantineCommand {
    /// List quarantined facts.
    List {
        #[command(flatten)]
        scopes: Scopes,
        #[arg(long)]
        limit: Option<i64>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Approve a quarantined fact.
    Approve {
        id: Uuid,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Reject a quarantined fact.
    Reject {
        id: Uuid,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum HookCommand {
    /// Print recalled memories for the host's context. Never fails the host.
    SessionStart,
    /// Print memories related to the user's prompt that this session has not shown yet, and a
    /// reminder to save when the prompt seems to state a fact. Never fails the host.
    Prompt,
    /// Periodically ask the host to save durable facts. Never fails the host.
    Stop,
    /// Add the hooks and the MCP server to an agent's configuration.
    Install(InstallArgs),
    /// Remove what `hook install` added.
    Uninstall(InstallArgs),
}

#[derive(Debug, Args)]
pub struct InstallArgs {
    #[arg(long, value_enum)]
    pub agent: Agent,
    /// Use this project's configuration, in the current directory. This is the default.
    #[arg(long, conflicts_with = "global")]
    pub project: bool,
    /// Use the user's configuration under the home directory.
    #[arg(long)]
    pub global: bool,
    /// Print what would change and write nothing.
    #[arg(long)]
    pub dry_run: bool,
}

/// Returns a message for the caller to print when a command fails. Hook commands always return
/// `Ok`, having already logged any failure.
/// The server URL and key `memoryos ui` forwards with.
pub fn ui_connection() -> Result<(String, String), String> {
    let env: config::Env<'_> = &config::process_env;
    Ok((config::base_url(env)?, config::ui_key(env)?))
}

pub fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Hook(HookCommand::SessionStart) => {
            hook::run_session_start();
            Ok(())
        }
        Command::Hook(HookCommand::Prompt) => {
            hook::run_prompt();
            Ok(())
        }
        Command::Hook(HookCommand::Stop) => {
            hook::run_stop();
            Ok(())
        }
        Command::Setup(args) => setup::run(&args),
        Command::Mcp => mcp::run(),
        Command::Hook(HookCommand::Install(args)) => install::run_install(&args),
        Command::Hook(HookCommand::Uninstall(args)) => install::run_uninstall(&args),
        Command::Recall {
            query,
            scopes,
            limit,
            json,
        } => client::recall(&query, scopes.scopes, limit, json),
        Command::Candidates {
            statement,
            scopes,
            limit,
            json,
        } => client::candidates(&statement, &scopes.scopes, limit, json),
        Command::Commit { file, json } => client::commit(file.as_deref(), json),
        Command::Quarantine(QuarantineCommand::List {
            scopes,
            limit,
            json,
        }) => client::quarantine_list(&scopes.scopes, limit, json),
        Command::Quarantine(QuarantineCommand::Approve { id, json }) => client::approve(id, json),
        Command::Quarantine(QuarantineCommand::Reject { id, json }) => client::reject(id, json),
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    #[derive(Debug, Parser)]
    struct Cli {
        #[command(subcommand)]
        command: Command,
    }

    fn parse(args: &[&str]) -> Result<Command, clap::Error> {
        let all = std::iter::once("memoryos").chain(args.iter().copied());
        Cli::try_parse_from(all).map(|cli| cli.command)
    }

    #[test]
    fn the_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn scopes_repeat_or_split_on_commas() {
        let command = parse(&[
            "recall",
            "q",
            "--scope",
            "user:a,agent:b",
            "--scope",
            "shared:c",
        ])
        .expect("ok");
        let Command::Recall { scopes, .. } = command else {
            panic!("wrong command");
        };
        assert_eq!(
            scopes.scopes,
            [Scope::user("a"), Scope::agent("b"), Scope::shared("c")]
        );
    }

    #[test]
    fn a_scope_is_required_and_validated() {
        assert!(parse(&["recall", "q"]).is_err());
        assert!(parse(&["recall", "q", "--scope", "nope"]).is_err());
    }

    #[test]
    fn project_and_global_conflict() {
        let args = [
            "hook",
            "install",
            "--agent",
            "opencode",
            "--project",
            "--global",
        ];
        assert!(parse(&args).is_err());
    }

    #[test]
    fn agents_use_their_published_names() {
        assert!(parse(&["hook", "install", "--agent", "claude-code"]).is_ok());
        assert!(parse(&["hook", "uninstall", "--agent", "opencode", "--dry-run"]).is_ok());
        assert!(parse(&["hook", "install", "--agent", "pi"]).is_ok());
        assert!(parse(&["hook", "install", "--agent", "other"]).is_err());
    }

    #[test]
    fn setup_needs_an_agent_unless_uninstalling() {
        assert!(parse(&["setup"]).is_err());
        assert!(parse(&["setup", "--agent", "opencode", "--project"]).is_ok());
        assert!(parse(&["setup", "--agent", "pi"]).is_ok());
        assert!(parse(&["setup", "--agent", "opencode", "--project", "--global"]).is_err());
        assert!(parse(&["setup", "--uninstall"]).is_ok());
        assert!(parse(&["setup", "--uninstall", "--delete-data"]).is_ok());
    }

    #[test]
    fn delete_data_is_only_valid_with_uninstall() {
        assert!(parse(&["setup", "--agent", "opencode", "--delete-data"]).is_err());
    }
}
