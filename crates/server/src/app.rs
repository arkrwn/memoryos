//! Builds the service from configuration and runs it until shutdown.

use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::http::Response;
use axum::Router;
use memoryos_core::{Embedder, MemoryStore, Reranker};
use memoryos_embed::{LocalEmbedder, LocalReranker};
use memoryos_ingest::HostCommitter;
use memoryos_retrieve::Retriever;
use memoryos_store::Store;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower_http::catch_panic::CatchPanicLayer;

use crate::config::{Config, RerankerMode};
use crate::error::ApiError;
use crate::rate_limit::{FailedAuthLimiter, RateLimiter};
use crate::readiness::{warm_up, warm_up_embedder, ModelReadiness, WarmUpTiming};
use crate::service::Services;
use crate::telemetry::Metrics;

const MAX_BODY_BYTES: usize = 1024 * 1024;
/// How long in-flight requests may finish after a shutdown signal. Under the 30 seconds an
/// orchestrator usually allows before it kills the process.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(25);

#[derive(Clone)]
pub struct AppState {
    pub(crate) services: Arc<Services>,
    pub(crate) metrics: Metrics,
    pub(crate) limiter: Arc<RateLimiter>,
    pub(crate) failed_auth: Arc<FailedAuthLimiter>,
    pub(crate) readiness: Arc<ModelReadiness>,
    pub(crate) reranker_readiness: Arc<ModelReadiness>,
    pub(crate) request_timeout: Duration,
}

/// The local reranker, unless the configuration turns it off. A test embedder build that uses the
/// test embedder runs without one, so no test downloads the reranking model. A reranker that cannot
/// be set up leaves recall on the fused order instead of stopping the server.
fn build_reranker(config: &Config) -> (Option<Arc<dyn Reranker>>, ModelReadiness) {
    #[cfg(feature = "test-embedder")]
    if config.test_embedder_delay.is_some() {
        return (None, ModelReadiness::off());
    }
    if config.reranker == RerankerMode::Off {
        return (None, ModelReadiness::off());
    }
    match LocalReranker::new(None) {
        Ok(reranker) => (Some(Arc::new(reranker)), ModelReadiness::default()),
        Err(error) => {
            tracing::warn!(reason = %error, "cannot set up the local reranker, recall keeps the fused order");
            let readiness = ModelReadiness::default();
            readiness.set_failed();
            (None, readiness)
        }
    }
}

fn build_embedder(config: &Config) -> Result<Arc<dyn Embedder>, String> {
    #[cfg(feature = "test-embedder")]
    if let Some(delay) = config.test_embedder_delay {
        return Ok(Arc::new(
            memoryos_embed::TestEmbedder::new().with_delay(delay),
        ));
    }
    #[cfg(not(feature = "test-embedder"))]
    let _ = config;
    LocalEmbedder::new(None)
        .map(|embedder| Arc::new(embedder) as Arc<dyn Embedder>)
        .map_err(|e| format!("cannot set up the local embedder: {e}"))
}

/// Serves on `listener` until `shutdown` fires, then finishes in-flight requests.
pub async fn serve(
    config: Config,
    listener: TcpListener,
    shutdown: CancellationToken,
) -> Result<(), String> {
    let store = Arc::new(
        Store::connect(
            config.store_backend,
            &config.database_url,
            config.db_max_connections,
        )
        .await
        .map_err(|e| format!("cannot connect to the database: {e}"))?,
    );
    let memory_store: Arc<dyn MemoryStore> = store.clone();
    let embedder = build_embedder(&config)?;
    let (reranker, reranker_readiness) = build_reranker(&config);
    let mut retriever = Retriever::new(memory_store.clone(), embedder.clone());
    if let Some(reranker) = &reranker {
        retriever = retriever.with_reranker(reranker.clone(), config.rerank_top);
    }
    let metrics = Metrics::new().map_err(|e| format!("cannot create metrics: {e}"))?;
    let services = Services {
        store,
        retriever: Arc::new(retriever),
        committer: HostCommitter::new(memory_store, embedder.clone())
            .with_single_valued(config.single_valued.clone()),
        metrics: metrics.clone(),
    };

    let readiness = Arc::new(ModelReadiness::default());
    let reranker_readiness = Arc::new(reranker_readiness);
    tokio::spawn(warm_up_models(
        embedder,
        readiness.clone(),
        reranker,
        reranker_readiness.clone(),
        shutdown.clone(),
    ));

    let state = AppState {
        services: Arc::new(services),
        metrics,
        limiter: Arc::new(RateLimiter::new(config.rate_limit_rps)),
        failed_auth: Arc::new(FailedAuthLimiter::new()),
        readiness,
        reranker_readiness,
        request_timeout: config.request_timeout,
    };

    tracing::info!("server started");
    let drained = serve_http(build_router(state), listener, shutdown).await;
    tracing::info!("server stopped");
    drained
}

/// The embedder first, because `/readyz` waits for it, then the reranker. One at a time also keeps
/// the first start from holding two downloads and two model loads in memory together.
async fn warm_up_models(
    embedder: Arc<dyn Embedder>,
    embedder_readiness: Arc<ModelReadiness>,
    reranker: Option<Arc<dyn Reranker>>,
    reranker_readiness: Arc<ModelReadiness>,
    shutdown: CancellationToken,
) {
    let timing = WarmUpTiming::default();
    let embed = move || warm_up_embedder(embedder.clone());
    warm_up(
        "local embedding model",
        embed,
        embedder_readiness,
        timing,
        shutdown.clone(),
    )
    .await;
    let Some(reranker) = reranker else {
        return;
    };
    let rerank = move || {
        let reranker = reranker.clone();
        async move { reranker.warm_up().await }
    };
    warm_up(
        "local reranking model",
        rerank,
        reranker_readiness,
        timing,
        shutdown,
    )
    .await;
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
