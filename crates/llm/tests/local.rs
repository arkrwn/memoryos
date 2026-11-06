#![cfg(feature = "integration-local")]
//! Runs the real model, downloading it on first use, so it needs `--features integration-local`.
//! Tests take turns on `SERIAL` so the download happens once.

use std::sync::Arc;
use std::time::Instant;

use memoryos_core::{Embedder, EMBEDDING_DIMS};
use memoryos_llm::LocalEmbedder;
use tokio::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::const_new(());

fn embedder() -> LocalEmbedder {
    LocalEmbedder::new(None).expect("cache directory is available")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

async fn embed_all(embedder: &LocalEmbedder, texts: &[&str]) -> Vec<Vec<f32>> {
    let inputs: Vec<String> = texts.iter().map(|text| text.to_string()).collect();
    embedder.embed(&inputs).await.expect("embedding succeeds")
}

async fn embed_one(embedder: &LocalEmbedder, text: &str) -> Vec<f32> {
    embed_all(embedder, &[text]).await.remove(0)
}

fn resident_mb() -> f64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps runs");
    let kb: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("ps prints a number");
    kb / 1024.0
}

#[tokio::test]
async fn vectors_are_384_wide_and_unit_length() {
    let _serial = SERIAL.lock().await;
    let vectors = embed_all(&embedder(), &["I moved to Bandung", "", "   "]).await;
    assert_eq!(vectors.len(), 3);
    for vector in vectors {
        assert_eq!(vector.len(), EMBEDDING_DIMS);
        let norm = cosine(&vector, &vector).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
    }
}

#[tokio::test]
async fn related_sentences_score_above_unrelated_across_languages() {
    let _serial = SERIAL.lock().await;
    let embedder = embedder();
    let english = embed_one(&embedder, "I moved to Bandung").await;
    let indonesian = embed_one(&embedder, "Saya pindah ke Bandung").await;
    let english_paraphrase = embed_one(&embedder, "I relocated to Bandung last year").await;
    let indonesian_paraphrase = embed_one(&embedder, "Saya sekarang tinggal di Bandung").await;
    let unrelated_en = embed_one(&embedder, "The recipe needs two cups of flour").await;
    let unrelated_id = embed_one(&embedder, "Resep ini membutuhkan dua cangkir tepung").await;

    let cross_language = cosine(&english, &indonesian);
    let same_english = cosine(&english, &english_paraphrase);
    let same_indonesian = cosine(&indonesian, &indonesian_paraphrase);
    let unrelated = [
        cosine(&english, &unrelated_en),
        cosine(&english, &unrelated_id),
        cosine(&indonesian, &unrelated_en),
        cosine(&indonesian, &unrelated_id),
    ];
    println!(
        "cosine: en-id {cross_language:.4}, en paraphrase {same_english:.4}, \
         id paraphrase {same_indonesian:.4}, unrelated {unrelated:.4?}"
    );

    let highest_unrelated = unrelated.into_iter().fold(f32::MIN, f32::max);
    assert!(cross_language > highest_unrelated);
    assert!(same_english > highest_unrelated);
    assert!(same_indonesian > highest_unrelated);
}

#[tokio::test]
async fn existing_prefix_does_not_change_the_vector() {
    let _serial = SERIAL.lock().await;
    let embedder = embedder();
    let plain = embed_one(&embedder, "I moved to Bandung").await;
    let prefixed = embed_one(&embedder, "query: I moved to Bandung").await;
    assert_eq!(plain, prefixed);
}

#[tokio::test]
async fn batch_order_is_preserved_across_internal_batches() {
    let _serial = SERIAL.lock().await;
    let embedder = embedder();
    let texts: Vec<String> = (0..40)
        .map(|i| format!("Customer number {i} moved to city number {i}"))
        .collect();
    let batch = embedder.embed(&texts).await.expect("embedding succeeds");
    assert_eq!(batch.len(), texts.len());
    for (position, (text, vector)) in texts.iter().zip(&batch).enumerate() {
        let alone = embed_one(&embedder, text).await;
        let nearest = batch
            .iter()
            .map(|other| cosine(&alone, other))
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index);
        assert_eq!(nearest, Some(position), "{text}");
        assert!(cosine(&alone, vector) > 0.999, "{text}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_calls_succeed_from_a_cold_start() {
    let _serial = SERIAL.lock().await;
    let embedder = Arc::new(embedder());
    let calls: Vec<_> = (0..8)
        .map(|i| {
            let embedder = embedder.clone();
            tokio::spawn(async move {
                let text = format!("Concurrent sentence number {i}");
                let vector = embed_one(&embedder, &text).await;
                (text, vector)
            })
        })
        .collect();
    let mut results = Vec::new();
    for call in calls {
        results.push(call.await.expect("task finishes"));
    }
    for (text, vector) in results {
        let alone = embed_one(&embedder, &text).await;
        assert!(cosine(&alone, &vector) > 0.999, "{text}");
    }
}

/// Prints the figures. Run it alone with `--nocapture`, because memory is per process.
#[tokio::test]
async fn measurements() {
    let _serial = SERIAL.lock().await;
    let before = resident_mb();
    let embedder = embedder();

    let start = Instant::now();
    embed_one(&embedder, "warm up").await;
    println!(
        "first call, load included: {} ms; rss {before:.0} MB -> {:.0} MB",
        start.elapsed().as_millis(),
        resident_mb()
    );

    const ROUNDS: u32 = 50;
    let start = Instant::now();
    for i in 0..ROUNDS {
        embed_one(&embedder, &format!("Saya pindah ke Bandung pada tahun {i}")).await;
    }
    println!(
        "single text: {:.1} ms per text over {ROUNDS} calls",
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(ROUNDS)
    );

    let texts: Vec<String> = (0..256)
        .map(|i| format!("User {i} said they moved to Bandung and started a new job there"))
        .collect();
    let start = Instant::now();
    embedder.embed(&texts).await.expect("embedding succeeds");
    println!(
        "batch of {}: {:.1} ms per text; rss after {:.0} MB",
        texts.len(),
        start.elapsed().as_secs_f64() * 1000.0 / texts.len() as f64,
        resident_mb()
    );

    let long = "Saya pindah ke Bandung pada tahun lalu. ".repeat(200);
    let start = Instant::now();
    embed_one(&embedder, &long).await;
    println!(
        "one text truncated at 512 tokens: {} ms; rss after {:.0} MB",
        start.elapsed().as_millis(),
        resident_mb()
    );
}
