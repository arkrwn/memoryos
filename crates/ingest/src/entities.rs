//! Entity resolution (SPEC section 3b), shared by both write modes.

use std::collections::HashMap;

use memoryos_core::{
    normalize_name, source_requires_quarantine, Embedder, Entity, Episode, Error, FactStatus,
    MemoryStore, NewEntity, RequestContext, Result, Scope,
};
use uuid::Uuid;

use crate::host::EntityRef;
use crate::validate::normalize_kind;

/// Minimum cosine similarity at which a new name merges into an entity of the same kind.
const MERGE_SIMILARITY: f64 = 0.90;
/// `similar_entities` cannot filter by kind, so this leaves room for other kinds in the results.
const SIMILAR_CANDIDATES: i64 = 20;

/// `(kind, normalized name)`: an entity's identity within a scope.
type EntityKey = (String, String);

fn key(entity: &EntityRef) -> EntityKey {
    (normalize_kind(&entity.kind), normalize_name(&entity.name))
}

/// One batch's embeddings and resolved entities.
pub(crate) struct ResolvedBatch {
    /// One vector per input statement, in order.
    pub(crate) statement_embeddings: Vec<Vec<f32>>,
    entity_ids: HashMap<EntityKey, Uuid>,
    /// Names merged into an entity. Written by `add_earned_aliases`, once statuses are known.
    pub(crate) merges: Vec<Merge>,
}

/// A name that resolved to an existing entity by similarity.
pub(crate) struct Merge {
    entity_id: Uuid,
    alias: String,
}

impl ResolvedBatch {
    pub(crate) fn entity_id(&self, entity: &EntityRef) -> Result<Uuid> {
        self.entity_ids
            .get(&key(entity))
            .copied()
            .ok_or_else(|| Error::Internal("entity was not resolved in this batch".into()))
    }
}

/// Embeds `statements` and the distinct entities in one provider call, statements first, then
/// resolves the entities one at a time, so a later name in the batch can merge into an earlier one.
///
/// Names are embedded in normalized form, so one normalized name always has one vector.
pub(crate) async fn embed_and_resolve(
    store: &dyn MemoryStore,
    embedder: &dyn Embedder,
    ctx: &RequestContext,
    scope: &Scope,
    statements: Vec<String>,
    entities: &[&EntityRef],
) -> Result<ResolvedBatch> {
    let mut distinct: Vec<(EntityKey, &str)> = Vec::new();
    for entity in entities.iter().copied() {
        let entity_key = key(entity);
        if !distinct.iter().any(|(seen, _)| *seen == entity_key) {
            distinct.push((entity_key, entity.name.trim()));
        }
    }

    let statement_count = statements.len();
    let mut inputs = statements;
    inputs.extend(
        distinct
            .iter()
            .map(|((_, normalized), _)| normalized.clone()),
    );
    let mut vectors = if inputs.is_empty() {
        Vec::new()
    } else {
        embedder.embed(&inputs).await?
    };
    if vectors.len() != inputs.len() {
        return Err(Error::Provider(format!(
            "embedder returned {} vectors for {} inputs",
            vectors.len(),
            inputs.len()
        )));
    }

    let entity_vectors = vectors.split_off(statement_count);
    let mut entity_ids = HashMap::with_capacity(distinct.len());
    let mut merges = Vec::new();
    for (((kind, normalized), name), embedding) in distinct.into_iter().zip(entity_vectors) {
        let id = resolve(store, ctx, scope, &kind, name, embedding, &mut merges).await?;
        entity_ids.insert((kind, normalized), id);
    }
    Ok(ResolvedBatch {
        statement_embeddings: vectors,
        entity_ids,
        merges,
    })
}

/// Runs after `apply_extraction`, because only then is each fact's status final. An alias changes
/// how a shared entity resolves for everyone, so a name is kept only when this episode created an
/// active fact on that entity: `created` is what `apply_extraction` returned, which leaves out
/// facts the episode only corroborated. A source whose facts are all quarantined earns none. A
/// failure here costs an alias and not the commit, which has already happened, so it is logged.
pub(crate) async fn add_earned_aliases(
    store: &dyn MemoryStore,
    ctx: &RequestContext,
    episode: &Episode,
    merges: Vec<Merge>,
    created: &[Uuid],
) {
    if merges.is_empty()
        || created.is_empty()
        || source_requires_quarantine(&episode.scope, episode.trust)
    {
        return;
    }
    let facts = match store.facts_for_episode(ctx, episode.id).await {
        Ok(facts) => facts,
        Err(err) => {
            tracing::warn!(episode_id = %episode.id, error = %err, "could not check aliases");
            return;
        }
    };
    for merge in merges {
        let earned = facts.iter().any(|fact| {
            created.contains(&fact.id)
                && fact.status == FactStatus::Active
                && (fact.subject_id == merge.entity_id || fact.object_id == Some(merge.entity_id))
        });
        if earned {
            if let Err(err) = store
                .add_entity_alias(ctx, merge.entity_id, &merge.alias, Some(episode.id))
                .await
            {
                tracing::warn!(entity_id = %merge.entity_id, error = %err, "could not add alias");
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum Choice {
    /// A different name similar enough to be the same entity.
    Merge(Uuid),
    Create,
}

/// `similar` is ordered by similarity, highest first, as `similar_entities` promises. Exact names
/// and known aliases never get here: `find_entity` has already answered those.
fn choose(similar: &[(Entity, f64)], kind: &str) -> Choice {
    let found = similar
        .iter()
        .find(|(entity, similarity)| entity.kind == kind && *similarity >= MERGE_SIMILARITY);
    match found {
        Some((entity, _)) => Choice::Merge(entity.id),
        None => Choice::Create,
    }
}

async fn resolve(
    store: &dyn MemoryStore,
    ctx: &RequestContext,
    scope: &Scope,
    kind: &str,
    name: &str,
    embedding: Vec<f32>,
    merges: &mut Vec<Merge>,
) -> Result<Uuid> {
    if let Some(entity) = store.find_entity(ctx, scope, kind, name).await? {
        return Ok(entity.id);
    }
    let similar = store
        .similar_entities(ctx, scope, &embedding, SIMILAR_CANDIDATES)
        .await?;
    match choose(&similar, kind) {
        Choice::Merge(id) => {
            merges.push(Merge {
                entity_id: id,
                alias: name.to_owned(),
            });
            Ok(id)
        }
        // Also covers two writers creating the same name at once: the store keeps one row.
        Choice::Create => {
            let entity = store
                .upsert_entity(
                    ctx,
                    NewEntity {
                        scope: scope.clone(),
                        kind: kind.to_owned(),
                        name: name.to_owned(),
                        aliases: Vec::new(),
                        embedding,
                    },
                )
                .await?;
            Ok(entity.id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(kind: &str, name: &str, aliases: &[&str]) -> Entity {
        Entity {
            id: Uuid::now_v7(),
            scope: Scope::user("u"),
            kind: kind.into(),
            name: name.into(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
        }
    }

    #[test]
    fn merges_at_the_threshold_and_not_below() {
        let found = entity("person", "Robert", &[]);
        let at = [(found.clone(), 0.90)];
        assert_eq!(choose(&at, "person"), Choice::Merge(found.id));
        let below = [(found, 0.8999)];
        assert_eq!(choose(&below, "person"), Choice::Create);
    }

    #[test]
    fn other_kinds_never_match() {
        let same_name = entity("company", "Apple", &[]);
        let similar = [(same_name, 1.0)];
        assert_eq!(choose(&similar, "fruit"), Choice::Create);
    }
}
