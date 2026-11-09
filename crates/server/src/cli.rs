//! The `memoryos` command line.
//!
//! Commands that talk to a running server through `MEMORYOS_URL` plug in by adding one variant to
//! `Command`, `#[command(flatten)] Client(memoryos_hooks::Command)`, and one arm to `dispatch`.
//! Nothing here reads the environment or starts logging before `dispatch` picks a command, so such
//! commands need no database and keep stdout clean.

use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use memoryos_core::{MemoryStore, Trust};
use memoryos_store::Store;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::auth::{generate_token, hash_token, validate_pattern};
use crate::config::{database_from_env, Config};
use crate::{app, telemetry, tui, ui};

const ADMIN_DB_CONNECTIONS: u32 = 2;
/// Blocking tasks cannot be cancelled, and a stalled model download is one. Past this the process
/// exits without it.
const RUNTIME_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Parser)]
#[command(name = "memoryos", version, about = "Long-term memory for AI agents")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the API server.
    Serve,
    /// Apply database migrations.
    Migrate,
    /// Manage API keys.
    #[command(subcommand)]
    Key(KeyCommand),
    /// Open a local page to see and review what is remembered. Keeps running until Ctrl+C.
    Ui {
        /// Local port. Falls back to a free one when the default is taken.
        #[arg(long)]
        port: Option<u16>,
        /// Print the address without opening a browser.
        #[arg(long)]
        no_open: bool,
    },
    /// The same views as `ui` in this terminal: facts, review, ask and timeline. `q` quits.
    Tui,
    #[command(flatten)]
    Client(memoryos_hooks::Command),
}

#[derive(Debug, Subcommand)]
pub enum KeyCommand {
    /// Create a key. Prints the token once, on stdout. Only its hash is stored.
    Create {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        name: String,
        /// Comma-separated patterns: `*`, `user:*`, or an exact scope like `agent:a1`.
        #[arg(long, value_delimiter = ',', required = true)]
        scopes: Vec<String>,
        /// Highest trust this key may write: external, agent, user or system.
        #[arg(long)]
        max_trust: String,
        /// Allow quarantine review, purge and user deletion, within the key's scopes.
        #[arg(long)]
        admin: bool,
    },
    /// Revoke a key by id. Takes effect on the next request.
    Revoke { id: Uuid },
}

pub fn dispatch(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Serve => serve(),
        Command::Migrate => block_on(migrate()),
        Command::Key(KeyCommand::Create {
            tenant,
            name,
            scopes,
            max_trust,
            admin,
        }) => block_on(create_key(tenant, name, scopes, max_trust, admin)),
        Command::Key(KeyCommand::Revoke { id }) => block_on(revoke_key(id)),
        Command::Ui { port, no_open } => block_on(ui::run(port, !no_open)),
        Command::Tui => tui::run(),
        Command::Client(command) => memoryos_hooks::run(command),
    }
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the runtime: {e}"))
}

fn block_on<F: std::future::Future<Output = Result<(), String>>>(work: F) -> Result<(), String> {
    runtime()?.block_on(work)
}

/// Loads configuration first, so a missing requirement fails before anything starts.
fn serve() -> Result<(), String> {
    let config = Config::from_env()?;
    let telemetry = telemetry::init()?;
    let runtime = runtime()?;
    let result = runtime.block_on(async {
        let shutdown = CancellationToken::new();
        tokio::spawn(cancel_on_signal(shutdown.clone()));
        let listener = TcpListener::bind(&config.bind)
            .await
            .map_err(|e| format!("cannot bind {}: {e}", config.bind))?;
        app::serve(config, listener, shutdown).await
    });
    runtime.shutdown_timeout(RUNTIME_SHUTDOWN_TIMEOUT);
    telemetry.shutdown();
    result
}

pub(crate) async fn cancel_on_signal(shutdown: CancellationToken) {
    wait_for_signal().await;
    tracing::info!("shutdown signal received");
    shutdown.cancel();
}

#[cfg(unix)]
async fn wait_for_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let (Ok(mut term), Ok(mut int)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    ) else {
        tracing::error!("cannot install signal handlers");
        return std::future::pending().await;
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}

#[cfg(not(unix))]
async fn wait_for_signal() {
    if tokio::signal::ctrl_c().await.is_err() {
        tracing::error!("cannot install signal handlers");
        std::future::pending::<()>().await;
    }
}

async fn admin_store() -> Result<Arc<Store>, String> {
    let (backend, url) = database_from_env()?;
    Store::connect(backend, &url, ADMIN_DB_CONNECTIONS)
        .await
        .map(Arc::new)
        .map_err(|e| format!("cannot connect to the database: {e}"))
}

async fn migrate() -> Result<(), String> {
    admin_store()
        .await?
        .migrate()
        .await
        .map_err(|e| format!("migration failed: {e}"))?;
    println!("migrations applied");
    Ok(())
}

async fn create_key(
    tenant: String,
    name: String,
    scopes: Vec<String>,
    max_trust: String,
    admin: bool,
) -> Result<(), String> {
    scopes
        .iter()
        .try_for_each(|scope| validate_pattern(scope))?;
    let max_trust: Trust = max_trust.parse().map_err(|e| format!("{e}"))?;
    if tenant.trim().is_empty() || name.trim().is_empty() {
        return Err("--tenant and --name must not be empty".into());
    }
    let token = generate_token()?;
    let store = admin_store().await?;
    let key = store
        .create_api_key(
            &tenant,
            &name,
            &hash_token(&token),
            &scopes,
            max_trust,
            admin,
        )
        .await
        .map_err(|e| format!("cannot create the key: {e}"))?;
    // Stdout holds the token alone, so `TOKEN=$(memoryos key create ...)` works.
    eprintln!(
        "key {} created; store this token now, it is not shown again",
        key.id
    );
    println!("{token}");
    Ok(())
}

async fn revoke_key(id: Uuid) -> Result<(), String> {
    admin_store()
        .await?
        .revoke_api_key(id)
        .await
        .map_err(|e| format!("cannot revoke the key: {e}"))?;
    println!("key {id} revoked");
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn the_command_line_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn key_create_splits_scopes_on_commas() {
        let cli = Cli::parse_from([
            "memoryos",
            "key",
            "create",
            "--tenant",
            "t",
            "--name",
            "n",
            "--scopes",
            "user:*,agent:a",
            "--max-trust",
            "agent",
            "--admin",
        ]);
        let Command::Key(KeyCommand::Create { scopes, admin, .. }) = cli.command else {
            panic!("wrong command");
        };
        assert_eq!(scopes, ["user:*", "agent:a"]);
        assert!(admin);
    }
}
