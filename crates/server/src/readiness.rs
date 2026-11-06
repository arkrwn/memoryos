//! Local model warm-up. The server starts first and the model loads behind it, so a stalled
//! download costs readiness and vector search, never `/healthz` or the API itself.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use memoryos_core::Embedder;
use tokio_util::sync::CancellationToken;

const LOADING: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;

const WARM_UP_INPUT: &str = "warm up";

/// The embedder half of `/readyz`.
#[derive(Default)]
pub struct EmbedderReadiness(AtomicU8);

impl EmbedderReadiness {
    pub fn is_ready(&self) -> bool {
        self.0.load(Ordering::Acquire) == READY
    }

    pub fn label(&self) -> &'static str {
        match self.0.load(Ordering::Acquire) {
            LOADING => "loading",
            READY => "ready",
            _ => "failed",
        }
    }

    fn set(&self, state: u8) {
        self.0.store(state, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WarmUpTiming {
    /// How often to log that the first call has not returned.
    pub still_loading_every: Duration,
    pub retry_base: Duration,
    pub retry_max: Duration,
}

impl Default for WarmUpTiming {
    fn default() -> Self {
        Self {
            still_loading_every: Duration::from_secs(60),
            retry_base: Duration::from_secs(30),
            retry_max: Duration::from_secs(300),
        }
    }
}

/// Makes one embed call, retrying a failure with backoff until it succeeds or `shutdown` fires. A
/// failed download can recover, for example when the network returns, so it is not fatal.
pub async fn warm_up(
    embedder: Arc<dyn Embedder>,
    readiness: Arc<EmbedderReadiness>,
    timing: WarmUpTiming,
    shutdown: CancellationToken,
) {
    let input = [WARM_UP_INPUT.to_owned()];
    let mut retry = timing.retry_base;
    let mut ticker = tokio::time::interval(timing.still_loading_every);
    ticker.tick().await;
    tracing::info!("loading the local embedding model");
    loop {
        let attempt = embedder.embed(&input);
        tokio::pin!(attempt);
        let outcome = loop {
            tokio::select! {
                () = shutdown.cancelled() => return,
                outcome = &mut attempt => break outcome,
                _ = ticker.tick() => {
                    tracing::warn!(
                        reason = "the first embed call has not returned, the model may still be downloading",
                        "not ready"
                    );
                }
            }
        };
        match outcome {
            Ok(_) => {
                readiness.set(READY);
                tracing::info!("local embedding model ready");
                return;
            }
            Err(error) => {
                readiness.set(FAILED);
                tracing::warn!(
                    reason = %error,
                    retry_secs = retry.as_secs(),
                    "not ready: the local embedding model failed to load"
                );
            }
        }
        tokio::select! {
            () = shutdown.cancelled() => return,
            () = tokio::time::sleep(retry) => {}
        }
        retry = (retry * 2).min(timing.retry_max);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use async_trait::async_trait;
    use memoryos_core::{Error, Result};

    use super::*;

    /// Fails `failures` times, then answers. `hang` never answers.
    struct FakeEmbedder {
        failures: usize,
        calls: AtomicUsize,
        hang: bool,
    }

    #[async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, _inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            if self.hang {
                std::future::pending::<()>().await;
            }
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call < self.failures {
                return Err(Error::Provider("download failed".into()));
            }
            Ok(vec![vec![1.0]])
        }
    }

    fn timing() -> WarmUpTiming {
        WarmUpTiming {
            still_loading_every: Duration::from_millis(5),
            retry_base: Duration::from_millis(1),
            retry_max: Duration::from_millis(2),
        }
    }

    fn fake(failures: usize, hang: bool) -> Arc<FakeEmbedder> {
        Arc::new(FakeEmbedder {
            failures,
            calls: AtomicUsize::new(0),
            hang,
        })
    }

    #[test]
    fn a_new_server_is_loading_and_not_ready() {
        let readiness = EmbedderReadiness::default();
        assert!(!readiness.is_ready());
        assert_eq!(readiness.label(), "loading");
    }

    #[tokio::test]
    async fn ready_after_one_successful_call() {
        let readiness = Arc::new(EmbedderReadiness::default());
        let embedder = fake(0, false);
        warm_up(
            embedder.clone(),
            readiness.clone(),
            timing(),
            CancellationToken::new(),
        )
        .await;
        assert!(readiness.is_ready());
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failure_is_retried_until_it_succeeds() {
        let readiness = Arc::new(EmbedderReadiness::default());
        let embedder = fake(3, false);
        warm_up(
            embedder.clone(),
            readiness.clone(),
            timing(),
            CancellationToken::new(),
        )
        .await;
        assert!(readiness.is_ready());
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn a_stalled_load_stays_not_ready_and_stops_on_shutdown() {
        let readiness = Arc::new(EmbedderReadiness::default());
        let shutdown = CancellationToken::new();
        let task = tokio::spawn(warm_up(
            fake(0, true),
            readiness.clone(),
            timing(),
            shutdown.clone(),
        ));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!readiness.is_ready());
        assert_eq!(readiness.label(), "loading");
        shutdown.cancel();
        task.await.expect("warm-up task");
        assert!(!readiness.is_ready());
    }
}
