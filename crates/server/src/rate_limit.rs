//! Per-key token bucket (SPEC section 9). In-process, so the limit applies per server instance.
//! Holds one entry per key that has authenticated, so bad tokens cannot grow it.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use uuid::Uuid;

struct Bucket {
    tokens: f64,
    updated: Instant,
}

pub struct RateLimiter {
    per_second: f64,
    buckets: Mutex<HashMap<Uuid, Bucket>>,
}

impl RateLimiter {
    /// The bucket holds one second of traffic.
    pub fn new(per_second: u32) -> Self {
        Self {
            per_second: f64::from(per_second),
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// `Err` holds how long until one request would be allowed.
    pub fn check(&self, key: Uuid) -> Result<(), Duration> {
        self.check_at(key, Instant::now())
    }

    fn check_at(&self, key: Uuid, now: Instant) -> Result<(), Duration> {
        let mut buckets = self
            .buckets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let bucket = buckets.entry(key).or_insert(Bucket {
            tokens: self.per_second,
            updated: now,
        });
        let elapsed = now.saturating_duration_since(bucket.updated).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.per_second).min(self.per_second);
        bucket.updated = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(())
        } else {
            Err(Duration::from_secs_f64(
                (1.0 - bucket.tokens) / self.per_second,
            ))
        }
    }
}

/// Tokens matching no key that may reach the database at once, and how fast that budget returns.
/// Enough for a client with a stale token, far less than a guessing loop wants.
const FAILED_AUTH_BURST: f64 = 100.0;
const FAILED_AUTH_PER_SECOND: f64 = 20.0;

struct FailedAuth {
    bucket: Bucket,
    /// Hashes of tokens that have worked since start, at most one per key that exists. With the
    /// budget spent these still reach the database, so bad tokens cannot lock out working clients.
    known: HashSet<String>,
}

/// One budget for the whole process, spent only by tokens the database did not recognize.
pub struct FailedAuthLimiter {
    state: Mutex<FailedAuth>,
}

impl FailedAuthLimiter {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(FailedAuth {
                bucket: Bucket {
                    tokens: FAILED_AUTH_BURST,
                    updated: Instant::now(),
                },
                known: HashSet::new(),
            }),
        }
    }

    /// Before the lookup. `Err` holds how long until the budget allows another unknown token.
    pub fn check(&self, token_hash: &str) -> Result<(), Duration> {
        self.check_at(token_hash, Instant::now())
    }

    pub fn failed(&self) {
        self.failed_at(Instant::now());
    }

    pub fn succeeded(&self, token_hash: &str) {
        let mut state = self.lock();
        if !state.known.contains(token_hash) {
            state.known.insert(token_hash.to_owned());
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FailedAuth> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn check_at(&self, token_hash: &str, now: Instant) -> Result<(), Duration> {
        let mut state = self.lock();
        refill(&mut state.bucket, now);
        if state.bucket.tokens >= 1.0 || state.known.contains(token_hash) {
            Ok(())
        } else {
            Err(Duration::from_secs_f64(
                (1.0 - state.bucket.tokens) / FAILED_AUTH_PER_SECOND,
            ))
        }
    }

    fn failed_at(&self, now: Instant) {
        let mut state = self.lock();
        refill(&mut state.bucket, now);
        // Requests already past the check can overdraw it, which only delays the refill.
        state.bucket.tokens = (state.bucket.tokens - 1.0).max(0.0);
    }
}

impl Default for FailedAuthLimiter {
    fn default() -> Self {
        Self::new()
    }
}

fn refill(bucket: &mut Bucket, now: Instant) {
    let elapsed = now.saturating_duration_since(bucket.updated).as_secs_f64();
    bucket.tokens = (bucket.tokens + elapsed * FAILED_AUTH_PER_SECOND).min(FAILED_AUTH_BURST);
    bucket.updated = now;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_a_burst_of_one_second_then_refuses() {
        let limiter = RateLimiter::new(3);
        let key = Uuid::now_v7();
        let now = Instant::now();
        for _ in 0..3 {
            assert!(limiter.check_at(key, now).is_ok());
        }
        let wait = limiter.check_at(key, now).expect_err("limited");
        assert!(wait > Duration::ZERO && wait <= Duration::from_secs(1));
    }

    #[test]
    fn refills_over_time_up_to_capacity() {
        let limiter = RateLimiter::new(2);
        let key = Uuid::now_v7();
        let now = Instant::now();
        limiter.check_at(key, now).expect("first");
        limiter.check_at(key, now).expect("second");
        assert!(limiter.check_at(key, now).is_err());
        let later = now + Duration::from_millis(500);
        assert!(limiter.check_at(key, later).is_ok());
        assert!(limiter.check_at(key, later).is_err());
        let much_later = later + Duration::from_secs(60);
        assert!(limiter.check_at(key, much_later).is_ok());
        assert!(limiter.check_at(key, much_later).is_ok());
        assert!(limiter.check_at(key, much_later).is_err());
    }

    #[test]
    fn failed_tokens_spend_one_budget_that_refills() {
        let limiter = FailedAuthLimiter::new();
        let now = Instant::now();
        for _ in 0..100 {
            assert!(limiter.check_at("bad", now).is_ok());
            limiter.failed_at(now);
        }
        assert!(limiter.check_at("bad", now).is_err());
        // Any other unknown token shares it.
        assert!(limiter.check_at("worse", now).is_err());
        let later = now + Duration::from_millis(100);
        assert!(limiter.check_at("bad", later).is_ok());
    }

    #[test]
    fn a_spent_budget_does_not_lock_out_a_token_that_has_worked() {
        let limiter = FailedAuthLimiter::new();
        limiter.succeeded("good");
        let now = Instant::now();
        for _ in 0..100 {
            limiter.failed_at(now);
        }
        assert!(limiter.check_at("bad", now).is_err());
        assert!(limiter.check_at("good", now).is_ok());
    }

    #[test]
    fn keys_do_not_share_a_bucket() {
        let limiter = RateLimiter::new(1);
        let now = Instant::now();
        let first = Uuid::now_v7();
        limiter.check_at(first, now).expect("first key");
        assert!(limiter.check_at(first, now).is_err());
        assert!(limiter.check_at(Uuid::now_v7(), now).is_ok());
    }
}
