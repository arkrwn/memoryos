//! A deterministic embedder, so tests need neither the 481 MB model nor a network. Compiled only with
//! the `test-embedder` feature, which the release image does not enable.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use memoryos_core::{Embedder, Error, Result, EMBEDDING_DIMS};

/// What one scripted input answers.
#[derive(Clone)]
enum Reply {
    Vector(Vec<f32>),
    Slow(Duration, Vec<f32>),
    Failure,
}

/// Answers each input with `vector_for` unless the test scripted that input, and records every batch.
#[derive(Default)]
pub struct TestEmbedder {
    delay: Duration,
    script: Mutex<HashMap<String, Reply>>,
    batches: Mutex<Vec<Vec<String>>>,
}

impl TestEmbedder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every call waits this long before it answers.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    pub fn set_vector(&self, input: &str, vector: Vec<f32>) {
        self.set(input, Reply::Vector(vector));
    }

    pub fn set_slow(&self, input: &str, delay: Duration, vector: Vec<f32>) {
        self.set(input, Reply::Slow(delay, vector));
    }

    /// A batch holding this input fails as a whole.
    pub fn set_failure(&self, input: &str) {
        self.set(input, Reply::Failure);
    }

    /// Back to `vector_for`.
    pub fn clear(&self, input: &str) {
        self.script.lock().expect("script").remove(input);
    }

    /// The inputs of every call so far, in order.
    pub fn batches(&self) -> Vec<Vec<String>> {
        self.batches.lock().expect("batches").clone()
    }

    fn set(&self, input: &str, reply: Reply) {
        self.script
            .lock()
            .expect("script")
            .insert(input.to_owned(), reply);
    }

    fn reply(&self, input: &str) -> Option<Reply> {
        self.script.lock().expect("script").get(input).cloned()
    }
}

#[async_trait]
impl Embedder for TestEmbedder {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.batches.lock().expect("batches").push(inputs.to_vec());
        tokio::time::sleep(self.delay).await;
        let mut vectors = Vec::with_capacity(inputs.len());
        for input in inputs {
            vectors.push(match self.reply(input) {
                None => vector_for(input),
                Some(Reply::Vector(vector)) => vector,
                Some(Reply::Slow(delay, vector)) => {
                    tokio::time::sleep(delay).await;
                    vector
                }
                Some(Reply::Failure) => {
                    return Err(Error::Provider("test embedder: scripted failure".into()))
                }
            });
        }
        Ok(vectors)
    }
}

/// Deterministic per text, and unrelated between texts, so two names never merge by similarity by
/// accident. Never all zero, which the store refuses.
pub fn vector_for(text: &str) -> Vec<f32> {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.bytes() {
        state ^= u64::from(byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (0..EMBEDDING_DIMS)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|text| (*text).to_owned()).collect()
    }

    #[tokio::test]
    async fn the_same_text_gets_the_same_vector_of_the_fixed_width() {
        let embedder = TestEmbedder::new();
        let vectors = embedder.embed(&inputs(&["a", "b", "a"])).await.expect("embed");
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0].len(), EMBEDDING_DIMS);
        assert_eq!(vectors[0], vectors[2]);
        assert_ne!(vectors[0], vectors[1]);
        assert_eq!(embedder.batches(), [inputs(&["a", "b", "a"])]);
    }

    #[test]
    fn unrelated_texts_are_far_apart() {
        let (a, b) = (vector_for("Robert"), vector_for("Bob"));
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((dot / (norm(&a) * norm(&b))).abs() < 0.5);
    }

    #[tokio::test]
    async fn scripted_inputs_answer_as_scripted() {
        let embedder = TestEmbedder::new();
        embedder.set_vector("a", vec![1.0]);
        embedder.set_failure("b");
        let vectors = embedder.embed(&inputs(&["a"])).await.expect("embed");
        assert_eq!(vectors, [vec![1.0]]);
        let failed = embedder.embed(&inputs(&["a", "b"])).await;
        assert!(matches!(failed, Err(Error::Provider(_))));
        embedder.clear("b");
        assert!(embedder.embed(&inputs(&["b"])).await.is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn delays_wait_before_answering() {
        let embedder = TestEmbedder::new().with_delay(Duration::from_secs(5));
        embedder.set_slow("a", Duration::from_secs(5), vec![1.0]);
        let started = tokio::time::Instant::now();
        embedder.embed(&inputs(&["a"])).await.expect("embed");
        assert_eq!(started.elapsed().as_secs(), 10);
    }
}
