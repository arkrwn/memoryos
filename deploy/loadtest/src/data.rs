//! Deterministic synthetic data: vocabulary, statements, vectors and dataset shapes. The same seed
//! gives the same rows, so a run can be reproduced.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub const DIMS: usize = 384;
const VOCABULARY: usize = 100_000;
const TOPICS: usize = 5_000;

/// Real English words first, so the head of the Zipf curve looks like real text. The `simple` text
/// search configuration keeps stop words, so these are the costly terms in production as well.
const COMMON: &str = "the of and to a in is that for it was on with as be at by this have from or \
    an but not are they which you one had were all we can her has there been if more when will \
    would who so no user prefers likes works lives uses wants said about into than them then some \
    its also what his only other new two may first any time very after most over such where much \
    before must through back years how many well because good each those people make way even \
    between could should still these think same another know while last might great old year off \
    come since against go came right used take three himself few house going say little world own \
    project team meeting coffee office home language city company email phone address birthday \
    name manager client server database report budget travel flight hotel music book movie food";

/// Words by frequency rank, sampled by a Zipf curve with exponent 1.
pub struct Vocab {
    words: Vec<String>,
    cdf: Vec<f64>,
    ranks: HashMap<String, usize>,
}

impl Vocab {
    pub fn new() -> Self {
        let mut words: Vec<String> = COMMON.split_whitespace().map(str::to_owned).collect();
        let mut index = 0;
        while words.len() < VOCABULARY {
            words.push(pseudo_word(index));
            index += 1;
        }
        let mut total = 0.0;
        let cdf = (0..words.len())
            .map(|rank| {
                total += 1.0 / (rank as f64 + 1.0);
                total
            })
            .collect();
        let ranks = words
            .iter()
            .enumerate()
            .map(|(rank, word)| (word.clone(), rank))
            .collect();
        Self { words, cdf, ranks }
    }

    pub fn sample(&self, rng: &mut SmallRng) -> &str {
        let total = self.cdf[self.cdf.len() - 1];
        let wanted = rng.gen::<f64>() * total;
        let rank = self.cdf.partition_point(|&value| value < wanted);
        &self.words[rank.min(self.words.len() - 1)]
    }

    pub fn word(&self, rank: usize) -> &str {
        &self.words[rank]
    }

    /// Unknown words rank last, which treats an entity name as rare.
    pub fn rank(&self, word: &str) -> usize {
        self.ranks.get(word).copied().unwrap_or(usize::MAX)
    }
}

/// A pronounceable, unique word for the long tail, such as `bakomi`.
fn pseudo_word(mut index: usize) -> String {
    const SYLLABLES: [&str; 32] = [
        "ba", "ko", "mi", "ta", "re", "lu", "sa", "ne", "do", "vi", "ga", "po", "zu", "fe", "hi",
        "jo", "ca", "de", "li", "mo", "nu", "pa", "ri", "so", "te", "wa", "ye", "bo", "ki", "la",
        "me", "ru",
    ];
    let mut word = String::from("x");
    loop {
        word.push_str(SYLLABLES[index % 32]);
        index /= 32;
        if index == 0 {
            return word;
        }
    }
}

/// Unit vectors grouped around topics, as real embeddings are. Purely random unit vectors would be
/// harder for HNSW than anything a real model produces, which would misstate the result.
pub struct Topics {
    centers: Vec<Vec<f32>>,
}

impl Topics {
    pub fn new(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let centers = (0..TOPICS).map(|_| random_unit(&mut rng)).collect();
        Self { centers }
    }

    pub fn count(&self) -> usize {
        TOPICS
    }

    pub fn near(&self, topic: usize, rng: &mut SmallRng) -> Vec<f32> {
        let center = &self.centers[topic % TOPICS];
        // Noise of about 0.6 of the center's length in total, spread over every dimension.
        let sigma = 0.6 / (DIMS as f32).sqrt();
        let mut vector: Vec<f32> = center
            .iter()
            .map(|value| value + sigma * gaussian(rng))
            .collect();
        normalize(&mut vector);
        vector
    }
}

pub fn random_unit(rng: &mut SmallRng) -> Vec<f32> {
    let mut vector: Vec<f32> = (0..DIMS).map(|_| gaussian(rng)).collect();
    normalize(&mut vector);
    vector
}

pub fn normalize(vector: &mut [f32]) {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if length > 0.0 {
        for value in vector {
            *value /= length;
        }
    }
}

pub fn gaussian(rng: &mut SmallRng) -> f32 {
    let (u, v): (f32, f32) = (rng.gen::<f32>().max(1e-9), rng.gen());
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}

/// One scope and how many facts it holds.
#[derive(Debug, Clone)]
pub struct ScopeSpec {
    pub tenant: String,
    pub scope: String,
    pub facts: usize,
    pub hub_edges: usize,
}

/// The tenant and scope holding the hub entity in shape `c`.
pub const HUB_TENANT: &str = "t-hub";
pub const HUB_SCOPE: &str = "shared:hub";
pub const HUB_EDGES: usize = 100_000;

/// Scopes of a shape, given as facts in total.
pub fn shape(name: &str, total: usize) -> Result<Vec<ScopeSpec>, String> {
    match name {
        // 1000 scopes in 20 tenants, sized by a power law, so a few are large and most are small.
        "a" => {
            let weights: Vec<f64> = (1..=1000).map(|i| 1.0 / f64::from(i).powf(0.9)).collect();
            let sum: f64 = weights.iter().sum();
            let mut specs: Vec<ScopeSpec> = weights
                .iter()
                .enumerate()
                .map(|(i, weight)| ScopeSpec {
                    tenant: format!("t-{:02}", i % 20),
                    scope: match i % 5 {
                        0..=3 => format!("user:u{i:04}"),
                        _ => format!("agent:a{i:04}"),
                    },
                    facts: ((weight / sum * total as f64) as usize).max(1),
                    hub_edges: 0,
                })
                .collect();
            let assigned: usize = specs.iter().map(|spec| spec.facts).sum();
            specs[0].facts += total - assigned;
            Ok(specs)
        }
        "b" => Ok(vec![ScopeSpec {
            tenant: "t-big".into(),
            scope: "user:big".into(),
            facts: total,
            hub_edges: 0,
        }]),
        // A hub with 10% of all facts as edges, inside a scope 1.5 times that, plus other scopes.
        "c" => {
            let hub_edges = total / 10;
            let hub_scope = hub_edges + hub_edges / 2;
            let others = 200;
            let each = (total - hub_scope) / others;
            let mut specs = vec![ScopeSpec {
                tenant: HUB_TENANT.into(),
                scope: HUB_SCOPE.into(),
                facts: total - each * others,
                hub_edges,
            }];
            specs.extend((0..others).map(|i| ScopeSpec {
                tenant: format!("t-{:02}", i % 10),
                scope: format!("user:o{i:03}"),
                facts: each,
                hub_edges: 0,
            }));
            Ok(specs)
        }
        other => Err(format!("unknown shape {other}, expected a, b or c")),
    }
}
