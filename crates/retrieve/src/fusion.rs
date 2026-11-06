//! Pure ranking logic: reciprocal rank fusion and the graph candidate list.

use std::collections::{HashMap, HashSet};

use memoryos_core::{Error, Fact, GraphNeighborhood, Result};
use uuid::Uuid;

const RRF_K: f64 = 60.0;

/// Reciprocal rank fusion, `score = sum(1 / (60 + rank))` with rank starting at 1, highest first.
///
/// A fact counts once per list, at its best rank. Equal scores order by fact id, so the same
/// inputs always give the same order.
pub(crate) fn fuse<'a>(lists: &[&'a [Fact]]) -> Vec<(&'a Fact, f64)> {
    let mut scored: HashMap<Uuid, (&'a Fact, f64)> = HashMap::new();
    for list in lists {
        let mut seen = HashSet::new();
        let unique = list.iter().filter(|fact| seen.insert(fact.id));
        for (index, fact) in unique.enumerate() {
            let contribution = 1.0 / (RRF_K + (index + 1) as f64);
            scored.entry(fact.id).or_insert((fact, 0.0)).1 += contribution;
        }
    }
    let mut fused: Vec<_> = scored.into_values().collect();
    fused.sort_unstable_by(|(a, a_score), (b, b_score)| {
        b_score.total_cmp(a_score).then_with(|| a.id.cmp(&b.id))
    });
    fused
}

/// Distinct subjects of the first `count` fused facts, best first.
pub(crate) fn top_subjects(fused: &[(&Fact, f64)], count: usize) -> Vec<Uuid> {
    let mut seen = HashSet::new();
    fused
        .iter()
        .take(count)
        .map(|(fact, _)| fact.subject_id)
        .filter(|subject| seen.insert(*subject))
        .collect()
}

/// Edges per subject, newest first, dropping subjects whose expansion failed. Edges carry no
/// relevance score and recency is the only signal left. The warning names the subject and the
/// error kind only, because error text can quote content.
pub(crate) fn usable_edges(results: Vec<(Uuid, Result<GraphNeighborhood>)>) -> Vec<Vec<Fact>> {
    results
        .into_iter()
        .filter_map(|(subject, result)| match result {
            Ok(neighborhood) => {
                let mut edges = neighborhood.edges;
                edges.sort_unstable_by_key(|edge| std::cmp::Reverse(edge.id));
                Some(edges)
            }
            Err(error) => {
                let kind = error_kind(&error);
                tracing::warn!(%subject, kind, "graph expansion failed for one subject");
                None
            }
        })
        .collect()
}

/// Takes one fact from each list in turn, up to `cap`, so a subject with many edges cannot crowd
/// out the others. A fact reached through two subjects is kept once.
pub(crate) fn interleave(lists: Vec<Vec<Fact>>, cap: usize) -> Vec<Fact> {
    let mut iters: Vec<_> = lists.into_iter().map(Vec::into_iter).collect();
    let mut seen = HashSet::new();
    let mut merged = Vec::new();
    while merged.len() < cap {
        let before = merged.len();
        for iter in &mut iters {
            if merged.len() == cap {
                break;
            }
            if let Some(fact) = iter.find(|fact| !seen.contains(&fact.id)) {
                seen.insert(fact.id);
                merged.push(fact);
            }
        }
        if merged.len() == before {
            break;
        }
    }
    merged
}

pub(crate) fn error_kind(error: &Error) -> &'static str {
    match error {
        Error::NotFound(_) => "not_found",
        Error::InvalidInput(_) => "invalid_input",
        Error::Forbidden(_) => "forbidden",
        Error::VersionConflict { .. } => "version_conflict",
        Error::BlockTooLarge { .. } => "block_too_large",
        Error::Provider(_) => "provider",
        Error::Storage(_) => "storage",
        Error::Internal(_) => "internal",
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use memoryos_core::{FactStatus, Scope, Trust};

    use super::*;

    fn id(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn fact(n: u128, subject: u128) -> Fact {
        Fact {
            id: id(n),
            scope: Scope::user("u"),
            subject_id: id(subject),
            predicate: "p".into(),
            object_id: None,
            statement: "s".into(),
            trust: Trust::User,
            status: FactStatus::Active,
            valid_from: Utc::now(),
            valid_to: None,
            recorded_at: Utc::now(),
            invalidated_at: None,
            superseded_by: None,
            pending_supersede_of: None,
            source_episode_ids: vec![],
        }
    }

    fn facts(ns: &[u128]) -> Vec<Fact> {
        ns.iter().map(|n| fact(*n, 1000 + n)).collect()
    }

    fn fused_ids(fused: &[(&Fact, f64)]) -> Vec<u128> {
        fused.iter().map(|(fact, _)| fact.id.as_u128()).collect()
    }

    fn list_ids(list: &[Fact]) -> Vec<u128> {
        list.iter().map(|fact| fact.id.as_u128()).collect()
    }

    fn neighborhood(root: Uuid, edges: Vec<Fact>) -> GraphNeighborhood {
        GraphNeighborhood {
            root,
            entities: vec![],
            edges,
        }
    }

    #[test]
    fn score_is_the_sum_of_reciprocal_ranks_from_rank_one() {
        let (first, second) = (facts(&[1, 2]), facts(&[2]));
        let fused = fuse(&[&first, &second]);
        assert_eq!(fused_ids(&fused), [2, 1]);
        assert_eq!(fused[0].1, 1.0 / 62.0 + 1.0 / 61.0);
        assert_eq!(fused[1].1, 1.0 / 61.0);
    }

    #[test]
    fn a_fact_in_two_lists_beats_a_better_rank_in_one() {
        let (first, second) = (facts(&[1, 2, 3]), facts(&[3]));
        assert_eq!(fused_ids(&fuse(&[&first, &second]))[0], 3);
    }

    #[test]
    fn dedupes_across_lists() {
        let (first, second, third) = (facts(&[1, 2]), facts(&[2, 1]), facts(&[1]));
        assert_eq!(fused_ids(&fuse(&[&first, &second, &third])), [1, 2]);
    }

    #[test]
    fn a_repeat_within_a_list_counts_once_and_takes_no_rank() {
        let list = facts(&[1, 1, 2]);
        let fused = fuse(&[&list]);
        assert_eq!(fused_ids(&fused), [1, 2]);
        assert_eq!(fused[0].1, 1.0 / 61.0);
        assert_eq!(fused[1].1, 1.0 / 62.0);
    }

    #[test]
    fn ties_order_by_id_whatever_the_list_order() {
        let (a, b) = (facts(&[9, 4]), facts(&[4, 9]));
        assert_eq!(fused_ids(&fuse(&[&a, &b])), [4, 9]);
        assert_eq!(fused_ids(&fuse(&[&b, &a])), [4, 9]);

        let (only_a, only_b, only_c) = (facts(&[7]), facts(&[3]), facts(&[5]));
        assert_eq!(fused_ids(&fuse(&[&only_a, &only_b, &only_c])), [3, 5, 7]);
    }

    #[test]
    fn empty_lists_fuse_to_nothing() {
        let empty: Vec<Fact> = vec![];
        assert!(fuse(&[&empty, &empty]).is_empty());
        assert!(fuse(&[]).is_empty());
    }

    #[test]
    fn top_subjects_looks_only_at_the_first_facts_and_dedupes() {
        let list = vec![fact(1, 100), fact(2, 100), fact(3, 200), fact(4, 300)];
        let fused = fuse(&[&list]);
        assert_eq!(top_subjects(&fused, 3), [id(100), id(200)]);
    }

    #[test]
    fn a_failed_subject_is_dropped_and_the_others_survive() {
        let results = vec![
            (id(100), Err(Error::NotFound("entity".into()))),
            (id(200), Ok(neighborhood(id(200), facts(&[1])))),
            (id(300), Err(Error::Storage("pool timed out".into()))),
        ];
        let lists = usable_edges(results);
        assert_eq!(lists.len(), 1);
        assert_eq!(list_ids(&lists[0]), [1]);
    }

    #[test]
    fn edges_come_newest_first() {
        let results = vec![(id(100), Ok(neighborhood(id(100), facts(&[1, 3, 2]))))];
        assert_eq!(list_ids(&usable_edges(results)[0]), [3, 2, 1]);
    }

    #[test]
    fn interleave_alternates_and_stops_at_the_cap() {
        let merged = interleave(vec![facts(&[1, 2, 3, 4]), facts(&[5, 6])], 5);
        assert_eq!(list_ids(&merged), [1, 5, 2, 6, 3]);
    }

    #[test]
    fn interleave_keeps_a_shared_edge_once_without_using_up_the_cap() {
        let merged = interleave(vec![facts(&[1, 2]), facts(&[1, 3])], 3);
        assert_eq!(list_ids(&merged), [1, 3, 2]);
    }

    #[test]
    fn interleave_ends_when_every_list_is_used_up() {
        assert_eq!(interleave(vec![facts(&[1]), vec![]], 50).len(), 1);
        assert!(interleave(vec![], 50).is_empty());
    }
}
