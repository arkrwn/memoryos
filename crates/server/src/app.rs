//! Builds the service from configuration and runs it until shutdown.

use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::http::Response;
use axum::Router;
use memoryos_agent::{AgentConfig, AgentRuntime};
use memoryos_core::{Embedder, Llm, MemoryStore};
use memoryos_ingest::{CancellationToken, HostCommitter, Ingestor, Worker, WorkerConfig};
use memoryos_llm::{AnthropicClient, LocalEmbedder, OpenAiClient, OpenAiEmbedder};
use memoryos_retrieve::Retriever;
use memoryos_store::PgStore;
use tokio::net::TcpListener;
use tower_http::catch_panic::CatchPanicLayer;

use crate::config::{Config, EmbedderKind, LlmConfig, LlmProvider, Mode};
use crate::error::ApiError;
use crate::metered::MeteredStore;
use crate::rate_limit::RateLimiter;
use crate::readiness::{warm_up, EmbedderReadiness, WarmUpTiming};
use crate::service::{RetrieverRecall, Services};
use crate::telemetry::Metrics;

const MAX_BODY_BYTES: usize = 1024 * 1024;
/// How long in-flight requests may finish after a shutdown signal. Under the 30 seconds an
/// orchestrator usually allows before it kills the process.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(25);
const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Clone)]
pub struct AppState {
    pub(crate) services: Arc<Services>,
    pub(crate) metrics: Metrics,
    pub(crate) limiter: Arc<RateLimiter>,
    pub(crate) readiness: Arc<EmbedderReadiness>,
    pub(crate) request_timeout: Duration,
    pub(crate) turn_timeout: Duration,
    pub(crate) mcp_allowed_origins: Arc<Vec<String>>,
}

/// The parts `serve` and `worker` share.
struct Providers {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn Embedder>,
    llm: Option<Arc<dyn Llm>>,
}

async fn providers(config: &Config) -> Result<Providers, String> {
    let store = PgStore::connect(&config.database_url, config.db_max_connections)
        .await
        .map_err(|e| format!("cannot connect to the database: {e}"))?;
    Ok(Providers {
        store: Arc::new(store),
        embedder: build_embedder(config)?,
        llm: config.llm.as_ref().map(build_llm),
    })
}

fn build_embedder(config: &Config) -> Result<Arc<dyn Embedder>, String> {
    match config.embedder {
        EmbedderKind::Local => LocalEmbedder::new(None)
            .map(|embedder| Arc::new(embedder) as Arc<dyn Embedder>)
            .map_err(|e| format!("cannot set up the local embedder: {e}")),
        EmbedderKind::OpenAi => {
            let api_key = config
                .openai_api_key
                .clone()
                .ok_or("the openai embedder requires OPENAI_API_KEY")?;
            Ok(Arc::new(OpenAiEmbedder::new(
                api_key,
                config.embedding_model.clone(),
                config.openai_base_url.clone(),
            )))
        }
    }
}

fn build_llm(config: &LlmConfig) -> Arc<dyn Llm> {
    let (key, model, url) = (
        config.api_key.clone(),
        config.model.clone(),
        config.base_url.clone(),
    );
    match config.provider {
        LlmProvider::Anthropic => Arc::new(AnthropicClient::new(key, model, url)),
        LlmProvider::OpenAi => Arc::new(OpenAiClient::new(key, model, url)),
    }
}

fn build_worker(
    config: &Config,
    providers: &Providers,
    llm: Arc<dyn Llm>,
    metrics: Metrics,
) -> Worker {
    let ingestor = Arc::new(Ingestor::new(
        providers.store.clone(),
        llm,
        providers.embedder.clone(),
    ));
    let metered: Arc<dyn MemoryStore> =
        Arc::new(MeteredStore::new(providers.store.clone(), metrics));
    Worker::new(
        metered,
        ingestor,
        WorkerConfig {
            concurrency: config.worker_concurrency,
            shutdown_timeout: WORKER_SHUTDOWN_TIMEOUT,
            ..WorkerConfig::default()
        },
    )
}

/// Serves on `listener` until `shutdown` fires, then finishes in-flight requests and worker jobs.
/// `run_worker` applies only in worker mode, because host mode has no worker at all.
pub async fn serve(
    config: Config,
    listener: TcpListener,
    run_worker: bool,
    shutdown: CancellationToken,
) -> Result<(), String> {
    let providers = providers(&config).await?;
    let metrics = Metrics::new().map_err(|e| format!("cannot create metrics: {e}"))?;
    let retriever = Arc::new(Retriever::new(
        providers.store.clone(),
        providers.embedder.clone(),
    ));
    let runtime = providers.llm.as_ref().map(|llm| {
        AgentRuntime::new(
            providers.store.clone(),
            Arc::new(RetrieverRecall(retriever.clone())),
            llm.clone(),
            AgentConfig::default(),
        )
    });
    let services = Services {
        store: providers.store.clone(),
        retriever,
        committer: HostCommitter::new(providers.store.clone(), providers.embedder.clone()),
        runtime,
        mode: config.mode,
        metrics: metrics.clone(),
    };

    let readiness = Arc::new(EmbedderReadiness::default());
    if config.embedder == EmbedderKind::Local {
        readiness.start_loading();
        tokio::spawn(warm_up(
            providers.embedder.clone(),
            readiness.clone(),
            WarmUpTiming::default(),
            shutdown.clone(),
        ));
    }

    let state = AppState {
        services: Arc::new(services),
        metrics: metrics.clone(),
        limiter: Arc::new(RateLimiter::new(config.rate_limit_rps)),
        readiness,
        request_timeout: config.request_timeout,
        turn_timeout: config.turn_timeout,
        mcp_allowed_origins: Arc::new(config.mcp_allowed_origins.clone()),
    };

    let worker = match (&providers.llm, run_worker, config.mode) {
        (Some(llm), true, Mode::Worker) => {
            let worker = build_worker(&config, &providers, llm.clone(), metrics);
            Some(tokio::spawn(worker.run(shutdown.clone())))
        }
        _ => None,
    };

    tracing::info!(mode = ?config.mode, embedder = ?config.embedder, worker = worker.is_some(), "server started");
    let drained = serve_http(build_router(state), listener, shutdown).await;
    if let Some(worker) = worker {
        if worker.await.is_err() {
            tracing::error!("the worker task panicked");
        }
    }
    tracing::info!("server stopped");
    drained
}

/// Runs only the worker loop: `memoryos worker`.
pub async fn run_worker(config: Config, shutdown: CancellationToken) -> Result<(), String> {
    if config.mode != Mode::Worker {
        return Err("memoryos worker requires MEMORYOS_MODE=worker".into());
    }
    let providers = providers(&config).await?;
    let llm = providers
        .llm
        .clone()
        .ok_or("worker mode requires an LLM configuration")?;
    let metrics = Metrics::new().map_err(|e| format!("cannot create metrics: {e}"))?;
    build_worker(&config, &providers, llm, metrics)
        .run(shutdown)
        .await;
    Ok(())
}

pub fn build_router(state: AppState) -> Router {
    crate::routes::router(&state)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(CatchPanicLayer::custom(panic_response))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::middleware::observe,
        ))
        .with_state(state)
}

/// The panic payload can quote request data, so it is not logged.
fn panic_response(_payload: Box<dyn std::any::Any + Send>) -> Response<axum::body::Body> {
    use axum::response::IntoResponse;
    tracing::error!("a request handler panicked");
    ApiError::internal().into_response()
}

async fn serve_http(
    router: Router,
    listener: TcpListener,
    shutdown: CancellationToken,
) -> Result<(), String> {
    let stopping = shutdown.clone();
    let server = axum::serve(listener, router)
        .with_graceful_shutdown(async move { stopping.cancelled().await });
    let drain_expired = async {
        shutdown.cancelled().await;
        tokio::time::sleep(DRAIN_TIMEOUT).await;
    };
    tokio::select! {
        result = server.into_future() => result.map_err(|e| format!("server error: {e}")),
        () = drain_expired => {
            tracing::warn!("shutdown drain timed out, dropping remaining connections");
            Ok(())
        }
    }
}
