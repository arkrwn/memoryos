//! Per-key token bucket (SPEC section 9). In-process, so the limit applies per server instance.
//! Holds one entry per key that has authenticated, so bad tokens cannot grow it.

use std::collections::HashMap;
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
    fn keys_do_not_share_a_bucket() {
        let limiter = RateLimiter::new(1);
        let now = Instant::now();
        let first = Uuid::now_v7();
        limiter.check_at(first, now).expect("first key");
        assert!(limiter.check_at(first, now).is_err());
        assert!(limiter.check_at(Uuid::now_v7(), now).is_ok());
    }
}
