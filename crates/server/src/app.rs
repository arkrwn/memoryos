//! Builds the service from configuration and runs it until shutdown.

use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::http::Response;
use axum::Router;
use memoryos_core::{Embedder, MemoryStore};
use memoryos_embed::LocalEmbedder;
use memoryos_ingest::HostCommitter;
use memoryos_retrieve::Retriever;
use memoryos_store::PgStore;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower_http::catch_panic::CatchPanicLayer;

use crate::config::Config;
use crate::error::ApiError;
use crate::rate_limit::RateLimiter;
use crate::readiness::{warm_up, EmbedderReadiness, WarmUpTiming};
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
    pub(crate) readiness: Arc<EmbedderReadiness>,
    pub(crate) request_timeout: Duration,
    pub(crate) mcp_allowed_origins: Arc<Vec<String>>,
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
    let pg = PgStore::connect(&config.database_url, config.db_max_connections)
        .await
        .map_err(|e| format!("cannot connect to the database: {e}"))?;
    let store: Arc<dyn MemoryStore> = Arc::new(pg.clone());
    let embedder = build_embedder(&config)?;
    let metrics = Metrics::new().map_err(|e| format!("cannot create metrics: {e}"))?;
    let services = Services {
        store: store.clone(),
        pg,
        retriever: Arc::new(Retriever::new(store.clone(), embedder.clone())),
        committer: HostCommitter::new(store, embedder.clone()),
        metrics: metrics.clone(),
    };

    let readiness = Arc::new(EmbedderReadiness::default());
    tokio::spawn(warm_up(
        embedder,
        readiness.clone(),
        WarmUpTiming::default(),
        shutdown.clone(),
    ));

    let state = AppState {
        services: Arc::new(services),
        metrics,
        limiter: Arc::new(RateLimiter::new(config.rate_limit_rps)),
        readiness,
        request_timeout: config.request_timeout,
        mcp_allowed_origins: Arc::new(config.mcp_allowed_origins.clone()),
    };

    tracing::info!("server started");
    let drained = serve_http(build_router(state), listener, shutdown).await;
    tracing::info!("server stopped");
    drained
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
