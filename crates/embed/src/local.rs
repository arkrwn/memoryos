//! On-device embeddings with `multilingual-e5-small` through `fastembed` and ONNX Runtime.

use std::error::Error as StdError;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use fastembed::{AutoDevicePolicy, EmbeddingModel, TextEmbedding, TextInitOptions};
use memoryos_core::{Embedder, Error, Result, EMBEDDING_DIMS};
use tokio::sync::Mutex;

const PROVIDER: &str = "local embedder";
const MODEL: EmbeddingModel = EmbeddingModel::MultilingualE5Small;
const CACHE_DIR_ENV: &str = "MEMORYOS_MODEL_CACHE";
const QUERY_PREFIX: &str = "query: ";
/// Inputs are padded to the longest in a batch and truncated at 512 tokens, so the output tensor
/// of one batch can reach 16 * 512 * 384 floats, about 12.6 MB.
const BATCH_SIZE: usize = 16;
/// A unit vector summed in `f32` lands well inside this. Zero and NaN norms land outside.
const UNIT_NORM_TOLERANCE: f32 = 1e-3;

/// Embeds on this machine, with no API key. The model downloads on the first non-empty `embed`
/// call, and that call waits for it. Calls run one at a time on the blocking pool.
///
/// fastembed gives `HF_HOME` priority over the cache directory, and reads `HF_ENDPOINT` and
/// `HF_TOKEN` for the download.
pub struct LocalEmbedder {
    engine: Engine<TextEmbedding>,
}

impl LocalEmbedder {
    /// Uses `cache_dir`, else `MEMORYOS_MODEL_CACHE`, else `memoryos/models` in the user cache
    /// directory. Creates the directory but loads nothing.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache_dir = model_cache_dir(cache_dir)?;
        Ok(Self {
            engine: Engine::new(move || load_model(cache_dir.clone())),
        })
    }
}

/// `explicit`, else `MEMORYOS_MODEL_CACHE`, else `memoryos/models` in the user cache directory,
/// created if missing. The embedder and the reranker share it.
pub(crate) fn model_cache_dir(explicit: Option<PathBuf>) -> Result<PathBuf> {
    let cache_dir = resolve_cache_dir(explicit, std::env::var_os(CACHE_DIR_ENV), dirs::cache_dir())
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "no model cache directory: set {CACHE_DIR_ENV} or pass one"
            ))
        })?;
    std::fs::create_dir_all(&cache_dir).map_err(|e| {
        Error::Internal(format!(
            "cannot create model cache directory {}: {e}",
            cache_dir.display()
        ))
    })?;
    Ok(cache_dir)
}

#[async_trait]
impl Embedder for LocalEmbedder {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.engine.embed(inputs).await
    }
}

fn load_model(cache_dir: PathBuf) -> Result<TextEmbedding> {
    // Without PreferCPU, ort registers CoreML on macOS, so vectors would depend on the machine.
    let options = TextInitOptions::new(MODEL)
        .with_cache_dir(cache_dir)
        .with_show_download_progress(false)
        .with_auto_device(AutoDevicePolicy::PreferCPU);
    TextEmbedding::try_new(options).map_err(|e| provider_error("model load failed", &e))
}

fn provider_error(context: &str, error: &(dyn StdError + 'static)) -> Error {
    provider_error_for(PROVIDER, context, error)
}

/// fastembed's error text leaves out the cause, such as the network error behind a failed download.
pub(crate) fn provider_error_for(
    provider: &str,
    context: &str,
    error: &(dyn StdError + 'static),
) -> Error {
    let mut message = format!("{provider} {context}: {error}");
    let mut source = error.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        if !message.ends_with(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        source = cause.source();
    }
    Error::Provider(message)
}

fn resolve_cache_dir(
    explicit: Option<PathBuf>,
    from_env: Option<OsString>,
    user_cache: Option<PathBuf>,
) -> Option<PathBuf> {
    explicit
        .or_else(|| {
            from_env
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| user_cache.map(|dir| dir.join("memoryos").join("models")))
}

/// The form the model documents for symmetric similarity. Text that already has the prefix keeps
/// one copy, because a doubled prefix changes the vector.
fn with_query_prefix(text: &str) -> String {
    let body = text.strip_prefix(QUERY_PREFIX).unwrap_or(text);
    format!("{QUERY_PREFIX}{body}")
}

/// A wrong vector would be stored and silently break search, so nothing is taken on trust.
fn validate(expected: usize, vectors: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
    if vectors.len() != expected {
        return Err(Error::Provider(format!(
            "{PROVIDER} returned {} vectors for {expected} inputs",
            vectors.len()
        )));
    }
    for vector in &vectors {
        if vector.len() != EMBEDDING_DIMS {
            return Err(Error::Provider(format!(
                "{PROVIDER} returned an embedding of width {}, expected {EMBEDDING_DIMS}",
                vector.len()
            )));
        }
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let is_unit = (norm - 1.0).abs() <= UNIT_NORM_TOLERANCE;
        if !is_unit {
            return Err(Error::Provider(format!(
                "{PROVIDER} returned an embedding with norm {norm}, expected a unit vector"
            )));
        }
    }
    Ok(vectors)
}

trait Model: Send + 'static {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

impl Model for TextEmbedding {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        TextEmbedding::embed(self, texts, Some(BATCH_SIZE))
            .map_err(|e| provider_error("inference failed", &e))
    }
}

type Loader<M> = Arc<dyn Fn() -> Result<M> + Send + Sync>;

/// Loads the model lazily and runs one call at a time. Callers queue on the async lock, so waiting
/// holds neither a runtime thread nor a blocking pool thread.
struct Engine<M> {
    slot: Arc<Mutex<Option<M>>>,
    load: Loader<M>,
}

impl<M: Model> Engine<M> {
    fn new(load: impl Fn() -> Result<M> + Send + Sync + 'static) -> Self {
        Self {
            slot: Arc::new(Mutex::new(None)),
            load: Arc::new(load),
        }
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = inputs.iter().map(|t| with_query_prefix(t)).collect();
        let mut slot = self.slot.clone().lock_owned().await;
        let load = self.load.clone();
        let vectors = tokio::task::spawn_blocking(move || {
            // Out of the slot while running: a panic drops the model, and the next call reloads
            // instead of reusing one left in an unknown state. A failed load stays retryable.
            let mut model = match slot.take() {
                Some(model) => model,
                None => load()?,
            };
            let vectors = model.embed(&texts);
            *slot = Some(model);
            vectors
        })
        .await
        .map_err(|e| Error::Provider(format!("{PROVIDER} task failed: {e}")))??;
        validate(inputs.len(), vectors)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    fn unit_vector(position: usize) -> Vec<f32> {
        let mut vector = vec![0.0; EMBEDDING_DIMS];
        vector[position] = 1.0;
        vector
    }

    /// Answers with a function of the text, so tests can see what reached the model and in what
    /// order.
    struct FakeModel {
        answer: fn(&[String]) -> Vec<Vec<f32>>,
        running: Arc<AtomicBool>,
        overlapped: Arc<AtomicBool>,
    }

    impl Model for FakeModel {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if self.running.swap(true, Ordering::SeqCst) {
                self.overlapped.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(5));
            let vectors = (self.answer)(texts);
            self.running.store(false, Ordering::SeqCst);
            Ok(vectors)
        }
    }

    struct Fixture {
        engine: Engine<FakeModel>,
        loads: Arc<AtomicUsize>,
        overlapped: Arc<AtomicBool>,
    }

    fn fixture(answer: fn(&[String]) -> Vec<Vec<f32>>) -> Fixture {
        let loads = Arc::new(AtomicUsize::new(0));
        let overlapped = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(false));
        let engine = {
            let loads = loads.clone();
            let overlapped = overlapped.clone();
            Engine::new(move || {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(FakeModel {
                    answer,
                    running: running.clone(),
                    overlapped: overlapped.clone(),
                })
            })
        };
        Fixture {
            engine,
            loads,
            overlapped,
        }
    }

    fn unit_vectors(texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|_| unit_vector(0)).collect()
    }

    fn strings(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|text| text.to_string()).collect()
    }

    #[test]
    fn prefix_is_added_to_plain_text() {
        assert_eq!(with_query_prefix("I moved"), "query: I moved");
    }

    #[test]
    fn prefix_is_added_to_empty_text() {
        assert_eq!(with_query_prefix(""), "query: ");
    }

    #[test]
    fn prefix_is_not_doubled() {
        assert_eq!(with_query_prefix("query: I moved"), "query: I moved");
    }

    #[test]
    fn prefix_is_only_recognized_at_the_start() {
        assert_eq!(with_query_prefix("my query: why"), "query: my query: why");
        assert_eq!(with_query_prefix("query:no space"), "query: query:no space");
    }

    #[test]
    fn cache_dir_prefers_explicit_then_env_then_user_cache() {
        let explicit = Some(PathBuf::from("/explicit"));
        let env = Some(OsString::from("/env"));
        let user = Some(PathBuf::from("/home/cache"));
        assert_eq!(
            resolve_cache_dir(explicit, env.clone(), user.clone()),
            Some(PathBuf::from("/explicit"))
        );
        assert_eq!(
            resolve_cache_dir(None, env, user.clone()),
            Some(PathBuf::from("/env"))
        );
        assert_eq!(
            resolve_cache_dir(None, Some(OsString::new()), user),
            Some(PathBuf::from("/home/cache/memoryos/models"))
        );
        assert_eq!(resolve_cache_dir(None, None, None), None);
    }

    #[test]
    fn model_width_matches_the_schema() {
        let info = TextEmbedding::get_model_info(&MODEL).unwrap();
        assert_eq!(info.dim, EMBEDDING_DIMS);
    }

    #[test]
    fn validate_accepts_unit_vectors() {
        assert!(validate(1, vec![unit_vector(3)]).is_ok());
    }

    #[test]
    fn validate_rejects_wrong_count() {
        let error = validate(2, vec![unit_vector(0)]).unwrap_err();
        assert!(matches!(error, Error::Provider(_)), "{error}");
    }

    #[test]
    fn validate_rejects_wrong_width() {
        let error = validate(1, vec![vec![1.0; 1536]]).unwrap_err();
        assert!(matches!(error, Error::Provider(m) if m.contains("width 1536")));
    }

    #[test]
    fn validate_rejects_zero_nan_and_unnormalized_vectors() {
        let mut not_a_number = unit_vector(0);
        not_a_number[1] = f32::NAN;
        let mut too_long = unit_vector(0);
        too_long[0] = 2.0;
        for vector in [vec![0.0; EMBEDDING_DIMS], not_a_number, too_long] {
            let error = validate(1, vec![vector]).unwrap_err();
            assert!(matches!(error, Error::Provider(m) if m.contains("norm")));
        }
    }

    #[tokio::test]
    async fn empty_input_loads_nothing() {
        let fixture = fixture(unit_vectors);
        assert!(fixture.engine.embed(&[]).await.unwrap().is_empty());
        assert_eq!(fixture.loads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn every_input_reaches_the_model_prefixed_once_and_in_order() {
        fn by_prefixed_text(texts: &[String]) -> Vec<Vec<f32>> {
            texts
                .iter()
                .map(|text| match text.as_str() {
                    "query: a" => unit_vector(0),
                    "query: b" => unit_vector(1),
                    other => panic!("unexpected text {other}"),
                })
                .collect()
        }
        let fixture = fixture(by_prefixed_text);
        let vectors = fixture
            .engine
            .embed(&strings(&["a", "query: b", "a"]))
            .await
            .unwrap();
        assert_eq!(vectors, [unit_vector(0), unit_vector(1), unit_vector(0)]);
    }

    #[tokio::test]
    async fn wrong_width_from_the_model_is_an_error() {
        let fixture = fixture(|texts| texts.iter().map(|_| vec![1.0; 10]).collect());
        let error = fixture.engine.embed(&strings(&["a"])).await.unwrap_err();
        assert!(matches!(error, Error::Provider(m) if m.contains("width 10")));
    }

    #[tokio::test]
    async fn panic_is_a_provider_error_and_the_next_call_reloads() {
        fn panics_on_boom(texts: &[String]) -> Vec<Vec<f32>> {
            assert!(texts[0] != "query: boom", "model blew up");
            unit_vectors(texts)
        }
        let fixture = fixture(panics_on_boom);
        fixture.engine.embed(&strings(&["a"])).await.unwrap();
        let error = fixture.engine.embed(&strings(&["boom"])).await.unwrap_err();
        assert!(matches!(error, Error::Provider(m) if m.contains("task failed")));
        assert_eq!(fixture.loads.load(Ordering::SeqCst), 1);

        fixture.engine.embed(&strings(&["a"])).await.unwrap();
        assert_eq!(fixture.loads.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn failed_load_is_returned_and_retried() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let engine = {
            let attempts = attempts.clone();
            Engine::<FakeModel>::new(move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(Error::Provider("download failed".into()))
            })
        };
        for attempt in 1..=2 {
            let error = engine.embed(&strings(&["a"])).await.unwrap_err();
            assert!(matches!(error, Error::Provider(m) if m == "download failed"));
            assert_eq!(attempts.load(Ordering::SeqCst), attempt);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_calls_load_once_and_never_overlap() {
        let fixture = Arc::new(fixture(unit_vectors));
        let calls: Vec<_> = (0..16)
            .map(|_| {
                let fixture = fixture.clone();
                tokio::spawn(async move { fixture.engine.embed(&strings(&["a"])).await })
            })
            .collect();
        for call in calls {
            assert!(call.await.unwrap().is_ok());
        }
        assert_eq!(fixture.loads.load(Ordering::SeqCst), 1);
        assert!(!fixture.overlapped.load(Ordering::SeqCst));
    }

    /// Finishes only when the runtime thread is the one that releases it.
    struct WaitsForRuntime(mpsc::Receiver<()>);

    impl Model for WaitsForRuntime {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.0
                .recv_timeout(Duration::from_secs(5))
                .map_err(|e| Error::Provider(e.to_string()))?;
            Ok(unit_vectors(texts))
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inference_leaves_the_runtime_thread() {
        let (release, wait) = mpsc::channel();
        let wait = std::sync::Mutex::new(Some(wait));
        let engine = Arc::new(Engine::new(move || {
            let wait = wait.lock().unwrap().take().unwrap();
            Ok(WaitsForRuntime(wait))
        }));
        let call = {
            let engine = engine.clone();
            tokio::spawn(async move { engine.embed(&strings(&["a"])).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        release.send(()).unwrap();
        assert!(call.await.unwrap().is_ok());
    }

    #[test]
    fn new_creates_the_cache_directory_without_loading() {
        let dir = std::env::temp_dir().join(format!("memoryos-local-{}", std::process::id()));
        let cache = dir.join("nested");
        LocalEmbedder::new(Some(cache.clone())).unwrap();
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
