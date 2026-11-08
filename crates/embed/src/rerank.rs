//! On-device reranking with the int8 ONNX export of `BAAI/bge-reranker-v2-m3`, through
//! `fastembed` and ONNX Runtime.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use fastembed::{
    AutoDevicePolicy, RerankInitOptionsUserDefined, TextRerank, TokenizerFiles,
    UserDefinedRerankingModel,
};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use memoryos_core::{Error, Reranker, Result};
use tokio::sync::Mutex;

use crate::local::{model_cache_dir, provider_error_for};

const PROVIDER: &str = "local reranker";
const REPO: &str = "onnx-community/bge-reranker-v2-m3-ONNX";
/// Pinned, so every machine downloads the same files.
const REVISION: &str = "6f5ff65298512715a1e669753bc754d2bc8f367b";
/// 571 MB.
const MODEL_FILE: &str = "onnx/model_int8.onnx";
/// Each (query, statement) pair is cut to this many tokens, the longer side first. Facts are short
/// statements, and 512 made a recall of 20 long statements take 8.7 s, past the timeout.
const MAX_TOKENS: usize = 256;
/// ONNX Runtime threads for one reranking call, see [`intra_threads`].
const MAX_THREADS: usize = 8;

/// Reranks on this machine, with no API key. Nothing loads until `warm_up`, which downloads the
/// model into the embedder's cache directory on the first start. Until it has loaded, `score`
/// fails at once instead of waiting for the download. Calls run one at a time on the blocking
/// pool, all pairs in one batch, on the CPU.
///
/// Like the embedder, it gives `HF_HOME` priority over the cache directory and reads
/// `HF_ENDPOINT` and `HF_TOKEN` for the download.
pub struct LocalReranker {
    engine: Engine<TextRerank>,
}

impl LocalReranker {
    /// Uses `cache_dir`, else `MEMORYOS_MODEL_CACHE`, else `memoryos/models` in the user cache
    /// directory. Creates the directory but loads nothing.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache_dir = model_cache_dir(cache_dir)?;
        Ok(Self {
            engine: Engine::new(move || load_model(cache_dir.clone())),
        })
    }
}

#[async_trait]
impl Reranker for LocalReranker {
    async fn score(&self, query: &str, statements: &[String]) -> Result<Vec<f32>> {
        self.engine.score(query, statements).await
    }

    async fn warm_up(&self) -> Result<()> {
        self.engine.warm_up().await
    }
}

fn load_model(cache_dir: PathBuf) -> Result<TextRerank> {
    let repo = repo(cache_dir)?;
    let fetch = |file: &str| {
        repo.get(file)
            .map_err(|e| provider_error_for(PROVIDER, &format!("download of {file} failed"), &e))
    };
    let read = |file: &str| {
        let path = fetch(file)?;
        std::fs::read(&path)
            .map_err(|e| Error::Provider(format!("{PROVIDER} cannot read {}: {e}", path.display())))
    };
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };
    let model = UserDefinedRerankingModel::new(fetch(MODEL_FILE)?, tokenizer_files);
    // Without PreferCPU, ort registers CoreML on macOS.
    let options = RerankInitOptionsUserDefined::new()
        .with_max_length(MAX_TOKENS)
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .with_intra_threads(intra_threads());
    TextRerank::try_new_from_user_defined(model, options)
        .map_err(|e| provider_error_for(PROVIDER, "model load failed", &e))
}

/// Every core, as fastembed would use, is slower on a machine with efficiency cores: on a 10 core
/// Apple M-series CPU, 10 pairs took a median 549 ms with 10 threads, 204 ms with 4 and 179 ms with
/// 8. In the Linux container on that machine, 10 pairs took a median 254 to 279 ms with 8 and 293
/// to 315 ms with 4, so 8 it is. Recalls rerank one at a time, so this is the most one recall takes.
fn intra_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |cores| cores.get())
        .min(MAX_THREADS)
}

/// The Hugging Face settings fastembed uses for the embedder, so both models share one cache and
/// one set of overrides.
fn repo(cache_dir: PathBuf) -> Result<ApiRepo> {
    let cache_dir = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or(cache_dir);
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_owned());
    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
        .or_else(default_token);
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .with_endpoint(endpoint)
        .with_token(token)
        .with_progress(false)
        .build()
        .map_err(|e| provider_error_for(PROVIDER, "cannot set up the download", &e))?;
    Ok(api.repo(Repo::with_revision(
        REPO.to_owned(),
        RepoType::Model,
        REVISION.to_owned(),
    )))
}

/// The token `huggingface-cli login` writes. `Cache::default` panics without a home directory.
fn default_token() -> Option<String> {
    let has_home = std::env::var_os("HOME").is_some() || std::env::var_os("USERPROFILE").is_some();
    has_home.then(|| hf_hub::Cache::default().token()).flatten()
}

/// A score the caller cannot compare would reorder recall at random, so nothing is taken on trust.
fn validate(expected: usize, scores: Vec<f32>) -> Result<Vec<f32>> {
    if scores.len() != expected {
        return Err(Error::Provider(format!(
            "{PROVIDER} returned {} scores for {expected} statements",
            scores.len()
        )));
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(Error::Provider(format!(
            "{PROVIDER} returned a score that is not a finite number"
        )));
    }
    Ok(scores)
}

trait Model: Send + 'static {
    fn score(&mut self, query: &str, statements: &[String]) -> Result<Vec<f32>>;
}

impl Model for TextRerank {
    fn score(&mut self, query: &str, statements: &[String]) -> Result<Vec<f32>> {
        let documents: Vec<&str> = statements.iter().map(String::as_str).collect();
        // One batch for every pair. fastembed returns them best first, so put them back in order.
        let ranked = self
            .rerank(query, documents, false, Some(statements.len()))
            .map_err(|e| provider_error_for(PROVIDER, "inference failed", &e))?;
        let mut scores = vec![f32::NAN; statements.len()];
        for result in ranked {
            if let Some(slot) = scores.get_mut(result.index) {
                *slot = result.score;
            }
        }
        Ok(scores)
    }
}

type Loader<M> = Arc<dyn Fn() -> Result<M> + Send + Sync>;

/// Loads the model when asked and runs one call at a time. Callers queue on the async lock, so
/// waiting holds neither a runtime thread nor a blocking pool thread.
struct Engine<M> {
    slot: Arc<Mutex<Option<M>>>,
    /// Set by the first successful `warm_up`. Before that, `score` fails without waiting.
    loaded: AtomicBool,
    load: Loader<M>,
}

impl<M: Model> Engine<M> {
    fn new(load: impl Fn() -> Result<M> + Send + Sync + 'static) -> Self {
        Self {
            slot: Arc::new(Mutex::new(None)),
            loaded: AtomicBool::new(false),
            load: Arc::new(load),
        }
    }

    async fn warm_up(&self) -> Result<()> {
        let mut slot = self.slot.clone().lock_owned().await;
        if slot.is_none() {
            let load = self.load.clone();
            tokio::task::spawn_blocking(move || {
                *slot = Some(load()?);
                Ok::<_, Error>(())
            })
            .await
            .map_err(|e| Error::Provider(format!("{PROVIDER} task failed: {e}")))??;
        }
        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    async fn score(&self, query: &str, statements: &[String]) -> Result<Vec<f32>> {
        if statements.is_empty() {
            return Ok(Vec::new());
        }
        if !self.loaded.load(Ordering::Acquire) {
            return Err(Error::Provider(format!(
                "{PROVIDER} model is not loaded yet"
            )));
        }
        let query = query.to_owned();
        let texts = statements.to_vec();
        let mut slot = self.slot.clone().lock_owned().await;
        let load = self.load.clone();
        let scores = tokio::task::spawn_blocking(move || {
            // Out of the slot while running: a panic drops the model, and the next call reloads
            // it from the cache instead of reusing one left in an unknown state.
            let mut model = match slot.take() {
                Some(model) => model,
                None => load()?,
            };
            let scores = model.score(&query, &texts);
            *slot = Some(model);
            scores
        })
        .await
        .map_err(|e| Error::Provider(format!("{PROVIDER} task failed: {e}")))??;
        validate(statements.len(), scores)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    /// Scores a statement by its length, and panics on "boom".
    struct FakeModel;

    impl Model for FakeModel {
        fn score(&mut self, _query: &str, statements: &[String]) -> Result<Vec<f32>> {
            assert!(statements.iter().all(|s| s != "boom"), "model blew up");
            Ok(statements.iter().map(|s| s.len() as f32).collect())
        }
    }

    fn engine() -> (Engine<FakeModel>, Arc<AtomicUsize>) {
        let loads = Arc::new(AtomicUsize::new(0));
        let counter = loads.clone();
        let engine = Engine::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(FakeModel)
        });
        (engine, loads)
    }

    fn strings(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|text| text.to_string()).collect()
    }

    #[tokio::test]
    async fn score_fails_at_once_before_warm_up_and_loads_nothing() {
        let (engine, loads) = engine();
        let error = engine.score("q", &strings(&["a"])).await.unwrap_err();
        assert!(matches!(error, Error::Provider(m) if m.contains("not loaded")));
        assert_eq!(loads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn scores_come_back_in_input_order_after_warm_up() {
        let (engine, loads) = engine();
        engine.warm_up().await.unwrap();
        engine.warm_up().await.unwrap();
        let scores = engine.score("q", &strings(&["abc", "a", "ab"])).await;
        assert_eq!(scores.unwrap(), [3.0, 1.0, 2.0]);
        assert!(engine.score("q", &[]).await.unwrap().is_empty());
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failed_warm_up_is_returned_and_retried() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let engine = Engine::<FakeModel>::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Err(Error::Provider("download failed".into()))
        });
        for attempt in 1..=2 {
            let error = engine.warm_up().await.unwrap_err();
            assert!(matches!(error, Error::Provider(m) if m == "download failed"));
            assert_eq!(attempts.load(Ordering::SeqCst), attempt);
        }
        assert!(engine.score("q", &strings(&["a"])).await.is_err());
    }

    #[tokio::test]
    async fn a_panic_is_a_provider_error_and_the_next_call_reloads() {
        let (engine, loads) = engine();
        engine.warm_up().await.unwrap();
        let error = engine.score("q", &strings(&["boom"])).await.unwrap_err();
        assert!(matches!(error, Error::Provider(m) if m.contains("task failed")));
        assert_eq!(engine.score("q", &strings(&["a"])).await.unwrap(), [1.0]);
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn validate_rejects_a_wrong_count_and_non_finite_scores() {
        assert!(validate(2, vec![1.0, -3.5]).is_ok());
        let error = validate(2, vec![1.0]).unwrap_err();
        assert!(matches!(error, Error::Provider(m) if m.contains("1 scores for 2")));
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(validate(1, vec![bad]).is_err());
        }
    }

    #[test]
    fn new_creates_the_cache_directory_without_loading() {
        let dir = std::env::temp_dir().join(format!("memoryos-rerank-{}", std::process::id()));
        let cache = dir.join("nested");
        LocalReranker::new(Some(cache.clone())).unwrap();
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
