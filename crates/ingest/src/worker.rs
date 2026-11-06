//! Job worker (SPEC section 3, steps 3 to 5). A worker can die at any point: leases recover its
//! jobs, and the store fences every job update, so a worker that lost its claim cannot overwrite
//! the worker that took over.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use memoryos_core::{Error, Job, MemoryStore, RequestContext, Result};
use tokio::sync::Semaphore;
use tokio::task::{JoinError, JoinSet};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::ingestor::Ingestor;

const EXTRACT_JOB: &str = "extract";
/// `RequestContext::actor` for store calls made on behalf of a job.
const WORKER_ACTOR: &str = "worker";
/// Time for in-flight jobs to record their release once told to stop.
const RELEASE_GRACE: Duration = Duration::from_secs(10);

/// `max_attempts`, `backoff_base` and `backoff_cap` default to SPEC section 3.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Must be unique per process: it is part of the fence on every job update.
    pub worker_id: String,
    /// Jobs run at the same time.
    pub concurrency: usize,
    /// Job lease. Renewed three times per lease while the job runs.
    pub lease_secs: i64,
    /// Mean sleep between polls while idle, jittered by up to half either way.
    pub poll_interval: Duration,
    pub max_attempts: i32,
    pub backoff_base: Duration,
    pub backoff_cap: Duration,
    /// How long in-flight jobs may finish after shutdown before they are released.
    pub shutdown_timeout: Duration,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            worker_id: format!("worker-{}", Uuid::now_v7().simple()),
            concurrency: 4,
            lease_secs: 60,
            poll_interval: Duration::from_secs(1),
            max_attempts: 5,
            backoff_base: Duration::from_secs(30),
            backoff_cap: Duration::from_secs(3600),
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

pub struct Worker {
    runner: Runner,
}

impl Worker {
    pub fn new(store: Arc<dyn MemoryStore>, ingestor: Arc<Ingestor>, config: WorkerConfig) -> Self {
        let config = WorkerConfig {
            concurrency: config.concurrency.max(1),
            lease_secs: config.lease_secs.max(1),
            max_attempts: config.max_attempts.max(1),
            ..config
        };
        Self {
            runner: Runner {
                store,
                ingestor,
                config: Arc::new(config),
            },
        }
    }

    /// Runs until `shutdown` is cancelled, then gives in-flight jobs `shutdown_timeout` to finish
    /// and releases the rest. Dropping this future instead aborts in-flight jobs, whose leases
    /// then expire.
    pub async fn run(self, shutdown: CancellationToken) {
        let Self { runner } = self;
        let config = Arc::clone(&runner.config);
        let slots = Arc::new(Semaphore::new(config.concurrency));
        let release = CancellationToken::new();
        let mut tasks = JoinSet::new();
        tracing::info!(
            worker_id = %config.worker_id,
            concurrency = config.concurrency,
            "worker started"
        );

        loop {
            while let Some(result) = tasks.try_join_next() {
                log_task_result(result);
            }
            let permit = tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                permit = Arc::clone(&slots).acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(_) => break,
                },
            };
            // Not raced against shutdown: abandoning a claim in flight could burn an attempt on a
            // job no one is running.
            match runner
                .store
                .claim_job(&config.worker_id, config.lease_secs)
                .await
            {
                Ok(Some(job)) => {
                    let runner = runner.clone();
                    let release = release.clone();
                    tasks.spawn(async move {
                        let _permit = permit;
                        runner.run_job(job, release).await;
                    });
                    continue;
                }
                Ok(None) => drop(permit),
                Err(err) => {
                    drop(permit);
                    tracing::warn!(error = %err, "could not claim a job");
                }
            }
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                () = tokio::time::sleep(jittered(config.poll_interval, fastrand::f64())) => {}
            }
        }

        tracing::info!(in_flight = tasks.len(), "worker stopping");
        if tokio::time::timeout(config.shutdown_timeout, drain(&mut tasks))
            .await
            .is_err()
        {
            tracing::warn!(in_flight = tasks.len(), "shutdown timeout, releasing jobs");
            release.cancel();
            if tokio::time::timeout(RELEASE_GRACE, drain(&mut tasks))
                .await
                .is_err()
            {
                // Their leases expire, and the jobs are claimable again after that.
                tasks.abort_all();
                drain(&mut tasks).await;
            }
        }
        tracing::info!("worker stopped");
    }
}

async fn drain(tasks: &mut JoinSet<()>) {
    while let Some(result) = tasks.join_next().await {
        log_task_result(result);
    }
}

/// A panicking job loses its lease to expiry, which counts as a crash against `max_attempts`.
fn log_task_result(result: std::result::Result<(), JoinError>) {
    if let Err(err) = result {
        if err.is_panic() {
            tracing::error!("job task panicked");
        }
    }
}

/// `unit` is a random value in `[0, 1)`.
fn jittered(interval: Duration, unit: f64) -> Duration {
    interval.mul_f64(0.5 + unit)
}

/// `attempts` counts this attempt, so the first failure waits `base`.
fn backoff(attempts: i32, base: Duration, cap: Duration) -> Duration {
    let doublings = u32::try_from(attempts.saturating_sub(1))
        .unwrap_or(0)
        .min(31);
    base.saturating_mul(2u32.saturating_pow(doublings)).min(cap)
}

enum Outcome {
    Finished(Result<()>),
    LeaseLost,
    Released,
}

#[derive(Clone)]
struct Runner {
    store: Arc<dyn MemoryStore>,
    ingestor: Arc<Ingestor>,
    config: Arc<WorkerConfig>,
}

impl Runner {
    async fn run_job(&self, job: Job, release: CancellationToken) {
        // Claiming counted this attempt. Past the limit means earlier attempts crashed before they
        // could report, so running it again could take down another worker.
        if job.attempts > self.config.max_attempts {
            tracing::error!(
                job_id = %job.id,
                tenant = %job.tenant,
                kind = %job.kind,
                attempts = job.attempts,
                "job exceeded max attempts, marking dead"
            );
            let reason = "exceeded max attempts: earlier attempts did not report a result";
            self.fail(&job, reason, Utc::now(), self.config.max_attempts)
                .await;
            return;
        }

        let started = Instant::now();
        tracing::info!(
            job_id = %job.id,
            tenant = %job.tenant,
            kind = %job.kind,
            attempts = job.attempts,
            "job started"
        );
        // Dropping the losing branches cancels them. An open store transaction rolls back.
        let outcome = tokio::select! {
            biased;
            () = release.cancelled() => Outcome::Released,
            () = self.keep_lease(&job) => Outcome::LeaseLost,
            result = self.execute(&job) => Outcome::Finished(result),
        };
        let elapsed_ms = started.elapsed().as_millis() as u64;

        match outcome {
            Outcome::LeaseLost => {
                tracing::warn!(job_id = %job.id, elapsed_ms, "job lease lost, abandoning");
            }
            // Must not go dead here: shutting down says nothing about the job. The attempt stays
            // counted, because the store has no way to give it back.
            Outcome::Released => {
                tracing::info!(job_id = %job.id, elapsed_ms, "job released at shutdown");
                self.fail(&job, "released at worker shutdown", Utc::now(), i32::MAX)
                    .await;
            }
            Outcome::Finished(Ok(())) => {
                match self.store.complete_job(&job, &self.config.worker_id).await {
                    Ok(true) => tracing::info!(job_id = %job.id, elapsed_ms, "job completed"),
                    Ok(false) => {
                        tracing::warn!(job_id = %job.id, "job finished after its claim was lost");
                    }
                    // The lease expires and a retry finds the episode already processed.
                    Err(err) => {
                        tracing::warn!(job_id = %job.id, error = %err, "could not complete job");
                    }
                }
            }
            Outcome::Finished(Err(err)) => {
                let delay = backoff(
                    job.attempts,
                    self.config.backoff_base,
                    self.config.backoff_cap,
                );
                tracing::warn!(
                    job_id = %job.id,
                    tenant = %job.tenant,
                    attempts = job.attempts,
                    delay_ms = delay.as_millis() as u64,
                    error = %err,
                    "job failed"
                );
                let run_after = Utc::now()
                    + chrono::Duration::from_std(delay).unwrap_or(chrono::Duration::hours(1));
                self.fail(&job, &err.to_string(), run_after, self.config.max_attempts)
                    .await;
            }
        }
    }

    async fn fail(&self, job: &Job, reason: &str, run_after: DateTime<Utc>, max_attempts: i32) {
        match self
            .store
            .fail_job(job, &self.config.worker_id, reason, run_after, max_attempts)
            .await
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(job_id = %job.id, "job failure not recorded, claim lost"),
            // The lease expires and the job runs again, or goes dead at the check after claiming.
            Err(err) => tracing::warn!(job_id = %job.id, error = %err, "could not record failure"),
        }
    }

    /// Returns only when the claim is lost. A store error may be temporary, so it keeps trying: if
    /// the lease does expire meanwhile, fencing still stops this worker from writing job state.
    async fn keep_lease(&self, job: &Job) {
        let interval = Duration::from_secs(self.config.lease_secs.unsigned_abs()) / 3;
        loop {
            tokio::time::sleep(interval).await;
            match self
                .store
                .renew_job(job, &self.config.worker_id, self.config.lease_secs)
                .await
            {
                Ok(true) => {}
                Ok(false) => return,
                Err(err) => {
                    tracing::warn!(job_id = %job.id, error = %err, "could not renew job lease");
                }
            }
        }
    }

    async fn execute(&self, job: &Job) -> Result<()> {
        if job.kind != EXTRACT_JOB {
            return Err(Error::InvalidInput(format!(
                "unknown job kind: {}",
                job.kind
            )));
        }
        let episode_id = job
            .payload
            .get("episode_id")
            .and_then(|value| value.as_str())
            .and_then(|value| value.parse::<Uuid>().ok())
            .ok_or_else(|| Error::InvalidInput("job payload has no valid episode_id".into()))?;
        let ctx = RequestContext {
            tenant: job.tenant.clone(),
            actor: WORKER_ACTOR.to_owned(),
        };
        match self.ingestor.process_episode(&ctx, episode_id).await {
            // Only a purged episode ends the job. A candidate fact or an entity purged mid-run is
            // also NotFound, and a retry with fresh candidates handles that.
            Err(Error::NotFound(what)) => match self.store.get_episode(&ctx, episode_id).await {
                Err(Error::NotFound(_)) => {
                    tracing::info!(job_id = %job.id, %episode_id, "episode was purged");
                    Ok(())
                }
                _ => Err(Error::NotFound(what)),
            },
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_from_30s_and_caps_at_1h() {
        let base = Duration::from_secs(30);
        let cap = Duration::from_secs(3600);
        let secs = |attempts| backoff(attempts, base, cap).as_secs();
        assert_eq!(secs(1), 30);
        assert_eq!(secs(2), 60);
        assert_eq!(secs(3), 120);
        assert_eq!(secs(4), 240);
        assert_eq!(secs(5), 480);
        assert_eq!(secs(8), 3600);
        assert_eq!(secs(40), 3600);
        assert_eq!(secs(i32::MAX), 3600);
        assert_eq!(secs(0), 30);
    }

    #[test]
    fn jitter_stays_within_half_of_the_interval() {
        let interval = Duration::from_secs(2);
        assert_eq!(jittered(interval, 0.0), Duration::from_secs(1));
        assert!(jittered(interval, 0.999) < Duration::from_secs(3));
    }

    #[test]
    fn defaults_match_the_spec() {
        let config = WorkerConfig::default();
        assert_eq!(config.max_attempts, 5);
        assert_eq!(config.backoff_base, Duration::from_secs(30));
        assert_eq!(config.backoff_cap, Duration::from_secs(3600));
        assert_ne!(config.worker_id, WorkerConfig::default().worker_id);
    }
}
