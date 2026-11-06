//! Entity resolution (SPEC section 3b), shared by both write modes.

use std::collections::HashMap;

use memoryos_core::{
    Embedder, Entity, Error, MemoryStore, NewEntity, RequestContext, Result, Scope,
};
use uuid::Uuid;

use crate::host::EntityRef;
use crate::validate::{normalize_kind, normalize_name};

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
/// Names are embedded in normalized form, so one normalized name always has one vector and an
/// existing entity with that name ranks first in `similar_entities`. The store trait has no lookup
/// by name, and this is what lets `choose` see the exact match.
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
    for (((kind, normalized), name), embedding) in distinct.into_iter().zip(entity_vectors) {
        let id = resolve(store, ctx, scope, &kind, &normalized, name, embedding).await?;
        entity_ids.insert((kind, normalized), id);
    }
    Ok(ResolvedBatch {
        statement_embeddings: vectors,
        entity_ids,
    })
}

#[derive(Debug, PartialEq)]
enum Choice {
    /// The name or a known alias already belongs to this entity.
    Existing(Uuid),
    /// A different name similar enough to be the same entity.
    Merge(Uuid),
    Create,
}

/// `similar` is ordered by similarity, highest first, as `similar_entities` promises.
fn choose(similar: &[(Entity, f64)], kind: &str, normalized: &str) -> Choice {
    let same_kind = || similar.iter().filter(|(entity, _)| entity.kind == kind);
    let known = same_kind().find(|(entity, _)| {
        normalize_name(&entity.name) == normalized
            || entity
                .aliases
                .iter()
                .any(|alias| normalize_name(alias) == normalized)
    });
    if let Some((entity, _)) = known {
        return Choice::Existing(entity.id);
    }
    match same_kind().find(|(_, similarity)| *similarity >= MERGE_SIMILARITY) {
        Some((entity, _)) => Choice::Merge(entity.id),
        None => Choice::Create,
    }
}

async fn resolve(
    store: &dyn MemoryStore,
    ctx: &RequestContext,
    scope: &Scope,
    kind: &str,
    normalized: &str,
    name: &str,
    embedding: Vec<f32>,
) -> Result<Uuid> {
    let similar = store
        .similar_entities(ctx, scope, &embedding, SIMILAR_CANDIDATES)
        .await?;
    match choose(&similar, kind, normalized) {
        Choice::Existing(id) => Ok(id),
        Choice::Merge(id) => {
            store.add_entity_alias(ctx, id, name).await?;
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
    fn exact_name_wins_over_a_closer_similar_entity() {
        let close = entity("person", "Other", &[]);
        let exact = entity("person", "Ada  Lovelace", &[]);
        let similar = [(close, 0.99), (exact.clone(), 0.95)];
        assert_eq!(
            choose(&similar, "person", "ada lovelace"),
            Choice::Existing(exact.id)
        );
    }

    #[test]
    fn known_alias_needs_no_new_alias() {
        let found = entity("person", "Robert", &["Bob"]);
        let similar = [(found.clone(), 0.5)];
        assert_eq!(
            choose(&similar, "person", "bob"),
            Choice::Existing(found.id)
        );
    }

    #[test]
    fn merges_at_the_threshold_and_not_below() {
        let found = entity("person", "Robert", &[]);
        let at = [(found.clone(), 0.90)];
        assert_eq!(choose(&at, "person", "bob"), Choice::Merge(found.id));
        let below = [(found, 0.8999)];
        assert_eq!(choose(&below, "person", "bob"), Choice::Create);
    }

    #[test]
    fn other_kinds_never_match() {
        let same_name = entity("company", "Apple", &[]);
        let similar = [(same_name, 1.0)];
        assert_eq!(choose(&similar, "fruit", "apple"), Choice::Create);
    }
}
