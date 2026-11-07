#![cfg(feature = "integration-local")]
//! Runs the real reranking model, downloading 571 MB on first use, so it needs
//! `--features integration-local`. Run it with `--release` for meaningful timings. Tests take turns
//! on `SERIAL` so the download happens once.

use std::time::{Duration, Instant};

use memoryos_core::Reranker;
use memoryos_embed::LocalReranker;
use tokio::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::const_new(());

const FACTS: [&str; 30] = [
    "The project stores its data in PostgreSQL 17 with the pgvector extension.",
    "Redis was considered for caching but rejected because it adds another service to run.",
    "The team deploys the API with Docker Compose on a single VM.",
    "The CI pipeline runs on GitHub Actions.",
    "Ada prefers tabs over spaces in Go code.",
    "Ada's nickname is aloi.",
    "Ada lives in Bandung.",
    "Ada works at Praja as a backend engineer.",
    "Ada uses Zed as her main editor.",
    "Ada drinks oolong tea every morning.",
    "The production deploy happens every Thursday afternoon.",
    "Release notes are written in English and published on the wiki.",
    "The payment service retries failed webhooks three times with exponential backoff.",
    "The search API returns at most 100 results per page.",
    "Login uses one-time passwords sent by SMS.",
    "The mobile app is written in Flutter.",
    "Error logs are shipped to Datadog.",
    "The staging database is reset every Monday.",
    "Kafka topics use a 7 day retention period.",
    "The team decided to drop MongoDB because transactions across collections were unreliable.",
    "项目的数据库是 PostgreSQL。",
    "团队每周四下午发布生产版本。",
    "Ada 喜欢喝乌龙茶。",
    "支付服务的回调失败后会重试三次。",
    "移动端应用使用 Flutter 开发。",
    "日志统一发送到 Datadog。",
    "Ada 住在万隆。",
    "代码评审至少需要两位同事批准。",
    "Code review needs approval from at least two teammates.",
    "The office closes at 6 pm on Fridays.",
];

/// A question, the facts that answer it, and the ten facts fused recall returned for it, in that
/// order. Fused recall put a right fact first for 8 of the 15.
struct Case {
    query: &'static str,
    answers: &'static [usize],
    candidates: [usize; 10],
}

#[rustfmt::skip]
const CASES: [Case; 15] = [
    Case { query: "what database do we use?", answers: &[0, 20], candidates: [20, 17, 19, 16, 14, 4, 26, 18, 13, 21] },
    Case { query: "why didn't we use Redis?", answers: &[1], candidates: [20, 19, 25, 24, 12, 27, 26, 23, 1, 16] },
    Case { query: "where does Ada live?", answers: &[6, 26], candidates: [26, 20, 5, 9, 6, 16, 8, 4, 7, 13] },
    Case { query: "what is my nickname?", answers: &[5], candidates: [20, 26, 25, 14, 5, 18, 17, 16, 15, 27] },
    Case { query: "which editor does Ada use?", answers: &[8], candidates: [26, 8, 4, 20, 5, 16, 9, 7, 13, 21] },
    Case { query: "when do we release to production?", answers: &[10, 21], candidates: [21, 29, 10, 20, 18, 17, 26, 19, 11, 28] },
    Case { query: "how many times are webhooks retried?", answers: &[12, 23], candidates: [18, 17, 19, 12, 26, 23, 16, 27, 25, 20] },
    Case { query: "where do error logs go?", answers: &[16, 25], candidates: [16, 25, 20, 17, 26, 19, 14, 27, 13, 4] },
    Case { query: "why was MongoDB dropped?", answers: &[19], candidates: [19, 20, 17, 16, 18, 12, 26, 21, 15, 1] },
    Case { query: "我们用什么数据库？", answers: &[0, 20], candidates: [20, 25, 27, 21, 26, 24, 19, 23, 22, 16] },
    Case { query: "Ada 喝什么茶？", answers: &[9, 22], candidates: [26, 20, 21, 25, 22, 27, 9, 24, 5, 23] },
    Case { query: "代码评审需要几个人批准？", answers: &[27, 28], candidates: [27, 28, 20, 21, 26, 25, 24, 23, 19, 22] },
    Case { query: "移动应用用什么框架？", answers: &[15, 24], candidates: [20, 27, 24, 21, 25, 26, 23, 22, 19, 4] },
    Case { query: "how long do Kafka messages stay?", answers: &[18], candidates: [18, 17, 29, 14, 20, 19, 0, 4, 1, 28] },
    Case { query: "how does login work?", answers: &[14], candidates: [25, 26, 16, 14, 20, 15, 17, 24, 18, 19] },
];

fn reranker() -> LocalReranker {
    LocalReranker::new(None).expect("cache directory is available")
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

fn best(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .expect("scores")
}

#[tokio::test]
async fn score_fails_at_once_until_warm_up() {
    let _serial = SERIAL.lock().await;
    let reranker = reranker();
    let started = Instant::now();
    let result = reranker.score("q", &["a".to_owned()]).await;
    assert!(result.is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn the_right_fact_comes_first_in_english_and_chinese() {
    let _serial = SERIAL.lock().await;
    let before = resident_mb();
    let reranker = reranker();
    let started = Instant::now();
    reranker.warm_up().await.expect("model loads");
    let load = started.elapsed();
    let loaded = resident_mb();

    let mut first = 0;
    let mut millis = Vec::new();
    for case in &CASES {
        let statements: Vec<String> = case
            .candidates
            .iter()
            .map(|&index| FACTS[index].to_owned())
            .collect();
        let started = Instant::now();
        let scores = reranker
            .score(case.query, &statements)
            .await
            .expect("scores");
        millis.push(started.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(scores.len(), statements.len());
        let top = case.candidates[best(&scores)];
        let right = case.answers.contains(&top);
        first += usize::from(right);
        println!(
            "{} {} -> {}",
            if right { "OK  " } else { "MISS" },
            case.query,
            FACTS[top]
        );
    }
    millis.sort_by(f64::total_cmp);
    println!(
        "right fact first {first}/{}, 10 pairs per query: median {:.0} ms, max {:.0} ms; \
         load {:.1} s; resident {before:.0} MB before, {loaded:.0} MB loaded, {:.0} MB after",
        CASES.len(),
        millis[millis.len() / 2],
        millis[millis.len() - 1],
        load.as_secs_f64(),
        resident_mb()
    );
    assert!(first >= 14, "right fact first only {first}/15");
}

/// The worst case recall can send: 20 statements, each past the 512 token cut.
#[tokio::test]
async fn long_statements_are_truncated_not_refused() {
    let _serial = SERIAL.lock().await;
    let reranker = reranker();
    reranker.warm_up().await.expect("model loads");
    let mut statements = vec!["Ada lives in Bandung. ".repeat(400)];
    statements.extend((0..19).map(|_| "The office closes at 6 pm on Fridays. ".repeat(300)));
    let started = Instant::now();
    let scores = reranker
        .score("where does Ada live?", &statements)
        .await
        .expect("scores");
    println!(
        "20 pairs of 512 tokens: {:.0} ms, resident {:.0} MB",
        started.elapsed().as_secs_f64() * 1000.0,
        resident_mb()
    );
    assert_eq!(best(&scores), 0, "{scores:?}");
}
