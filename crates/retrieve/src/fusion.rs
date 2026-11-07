//! Pure ranking logic: reciprocal rank fusion, the source count nudge, near copy removal and the
//! graph candidate list.

use std::collections::{HashMap, HashSet};

use memoryos_core::{Error, Fact, GraphNeighborhood, Result};
use uuid::Uuid;

const RRF_K: f64 = 60.0;

/// Score boost for each source episode past the first, see [`boost_by_sources`].
const SOURCE_BOOST_PER_EXTRA: f64 = 0.02;
/// Sources past the first that count toward the boost, so the boost tops out at 6%.
const SOURCE_BOOST_MAX_EXTRA: usize = 3;

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
    sort_by_score(&mut fused);
    fused
}

/// Highest score first, equal scores by fact id.
fn sort_by_score(ranked: &mut [(&Fact, f64)]) {
    ranked.sort_unstable_by(|(a, a_score), (b, b_score)| {
        b_score.total_cmp(a_score).then_with(|| a.id.cmp(&b.id))
    });
}

/// Multiplies each fused score by `1 + 0.02 * extra`, where `extra` is the number of source
/// episodes past the first, capped at 3, then sorts again. A fact with one source keeps its score.
///
/// Why 6% at most: near the top of a list it is worth about four ranks (`65 / 61 = 1.066`), so it
/// breaks ties and nudges. It stays below the 10.9% gap between the best fact found by one list
/// (`1 / 61`) and the worst found by two (`2 / 110`), so sources never lift a fact found by one
/// list over a fact found by two.
pub(crate) fn boost_by_sources(fused: Vec<(&Fact, f64)>) -> Vec<(&Fact, f64)> {
    let mut boosted: Vec<_> = fused
        .into_iter()
        .map(|(fact, score)| (fact, score * source_boost(fact)))
        .collect();
    sort_by_score(&mut boosted);
    boosted
}

fn source_boost(fact: &Fact) -> f64 {
    let extra = fact
        .source_episode_ids
        .len()
        .saturating_sub(1)
        .min(SOURCE_BOOST_MAX_EXTRA);
    1.0 + SOURCE_BOOST_PER_EXTRA * extra as f64
}

/// Reorders the first `scores.len()` facts by `scores`, highest first, and leaves the rest as they
/// were. The sort is stable, so equal scores keep the order they came in, which is the boosted
/// fused order. Each moved fact takes the fused score of the position it lands on, so scores stay
/// descending and on one scale for the whole list.
pub(crate) fn reordered_by<'a>(
    ranked: Vec<(&'a Fact, f64)>,
    scores: &[f32],
) -> Vec<(&'a Fact, f64)> {
    let window = scores.len().min(ranked.len());
    let slot_scores: Vec<f64> = ranked[..window].iter().map(|(_, score)| *score).collect();
    let mut order: Vec<usize> = (0..window).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let reordered = order
        .iter()
        .zip(slot_scores)
        .map(|(&index, slot_score)| (ranked[index].0, slot_score));
    reordered.chain(ranked[window..].iter().copied()).collect()
}

/// Drops each fact that is a near copy of a higher ranked one, keeping the order, so the next
/// distinct fact takes its slot. Near copy means the same subject, predicate and object and the
/// same statement after [`normalize`]. Candidate lists carry no embeddings, so this is the test
/// that needs no extra read.
pub(crate) fn without_near_copies(ranked: Vec<(&Fact, f64)>) -> Vec<(&Fact, f64)> {
    let mut seen = HashSet::new();
    ranked
        .into_iter()
        .filter(|(fact, _)| {
            let key = (
                fact.subject_id,
                fact.predicate.as_str(),
                fact.object_id,
                normalize(&fact.statement),
            );
            seen.insert(key)
        })
        .collect()
}

/// Lowercase words, split on anything that is not a letter or digit, joined by one space.
fn normalize(statement: &str) -> String {
    statement
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
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

    fn with_sources(mut fact: Fact, count: u128) -> Fact {
        fact.source_episode_ids = (0..count).map(|n| id(9000 + n)).collect();
        fact
    }

    fn stated(n: u128, subject: u128, statement: &str) -> Fact {
        Fact {
            statement: statement.into(),
            ..fact(n, subject)
        }
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
    fn one_source_keeps_the_fused_score() {
        let list = vec![with_sources(fact(1, 100), 1), fact(2, 200)];
        let boosted = boost_by_sources(fuse(&[&list]));
        assert_eq!(fused_ids(&boosted), [1, 2]);
        assert_eq!(boosted[0].1, 1.0 / 61.0);
        assert_eq!(boosted[1].1, 1.0 / 62.0);
    }

    #[test]
    fn sources_break_a_tie() {
        // Same rank in two lists, so the fused scores are equal and id order would put 1 first.
        let (a, b) = (vec![fact(1, 100)], vec![with_sources(fact(2, 200), 3)]);
        let fused = fuse(&[&a, &b]);
        assert_eq!(fused[0].1, fused[1].1);
        assert_eq!(fused_ids(&boost_by_sources(fused)), [2, 1]);
    }

    #[test]
    fn sources_nudge_a_few_ranks_and_stop_at_the_cap() {
        let list = vec![
            fact(1, 100),
            with_sources(fact(2, 200), 2),
            with_sources(fact(3, 300), 50),
            fact(4, 400),
            with_sources(fact(5, 500), 50),
        ];
        let boosted = boost_by_sources(fuse(&[&list]));
        // 1/61, 1.02/62, 1.06/63, 1/64 and 1.06/65: rank 5 stays behind rank 1.
        assert_eq!(fused_ids(&boosted), [3, 2, 1, 5, 4]);
        assert!((boosted[0].1 - 1.06 / 63.0).abs() < 1e-12);
    }

    #[test]
    fn many_sources_do_not_lift_a_fact_over_a_clearly_better_match() {
        // Found by one list only, at its best rank, with as many sources as anything can have.
        let lone = with_sources(fact(1, 100), 100);
        // Found by two lists, each at its worst rank, with one source.
        let filler = |offset: u128| (0..49).map(move |n| fact(offset + n, offset + n));
        let first: Vec<Fact> = std::iter::once(lone).chain(filler(10)).collect();
        let second: Vec<Fact> = filler(100).chain([fact(2, 200)]).collect();
        let third: Vec<Fact> = filler(200).chain([fact(2, 200)]).collect();

        let boosted = boost_by_sources(fuse(&[&first, &second, &third]));
        let order = fused_ids(&boosted);
        let rank = |n: u128| order.iter().position(|&id| id == n).expect("ranked");
        assert!(rank(2) < rank(1));
    }

    #[test]
    fn reorder_sorts_the_window_and_keeps_the_slot_scores() {
        let list = facts(&[1, 2, 3, 4]);
        let fused = fuse(&[&list]);
        let reordered = reordered_by(fused, &[0.1, 0.9, 0.5]);
        assert_eq!(fused_ids(&reordered), [2, 3, 1, 4]);
        let scores: Vec<f64> = reordered.iter().map(|(_, score)| *score).collect();
        assert_eq!(scores, [1.0 / 61.0, 1.0 / 62.0, 1.0 / 63.0, 1.0 / 64.0]);
    }

    #[test]
    fn reorder_keeps_the_incoming_order_on_equal_scores() {
        let list = facts(&[1, 2, 3]);
        let reordered = reordered_by(fuse(&[&list]), &[0.5, 0.7, 0.5]);
        assert_eq!(fused_ids(&reordered), [2, 1, 3]);
    }

    #[test]
    fn reorder_with_more_scores_than_facts_or_none_is_safe() {
        let list = facts(&[1, 2]);
        assert_eq!(
            fused_ids(&reordered_by(fuse(&[&list]), &[0.0, 1.0, 2.0])),
            [2, 1]
        );
        assert_eq!(fused_ids(&reordered_by(fuse(&[&list]), &[])), [1, 2]);
    }

    #[test]
    fn near_copies_give_way_to_the_next_distinct_fact() {
        let list = vec![
            stated(1, 100, "The zebra was seen."),
            stated(2, 100, "the  zebra was SEEN"),
            stated(3, 100, "the zebra ran"),
            stated(4, 100, "THE ZEBRA RAN!"),
        ];
        let kept = without_near_copies(fuse(&[&list]));
        assert_eq!(fused_ids(&kept), [1, 3]);
        assert_eq!(kept[1].1, 1.0 / 63.0);
    }

    #[test]
    fn same_words_about_other_entities_are_not_copies() {
        let other_subject = stated(2, 200, "the zebra was seen");
        let other_predicate = Fact {
            predicate: "q".into(),
            ..stated(3, 100, "the zebra was seen")
        };
        let other_object = Fact {
            object_id: Some(id(500)),
            ..stated(4, 100, "the zebra was seen")
        };
        let list = vec![
            stated(1, 100, "the zebra was seen"),
            other_subject,
            other_predicate,
            other_object,
            stated(5, 100, "a zebra was seen"),
        ];
        assert_eq!(
            fused_ids(&without_near_copies(fuse(&[&list]))),
            [1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn normalize_keeps_words_and_drops_case_and_punctuation() {
        assert_eq!(normalize("  Uses  Postgres, 17!\n"), "uses postgres 17");
        assert_eq!(normalize("Ünïcode ÉTÉ"), "ünïcode été");
        assert_eq!(normalize("--"), "");
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
