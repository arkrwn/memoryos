//! The API end to end, through the SDK where the SDK has the call.

use chrono::{Duration as ChronoDuration, Utc};
use memoryos_core::{Scope, Trust};
use memoryos_sdk::{Commit, CommitFact, EntityRef, Error, FactStatus, MemoryOs, PutBlock};
use reqwest::StatusCode;
use serde_json::json;

use crate::common::{assert_error, random_uuid, statement, uuid_at, Server};

fn fact(text: &str) -> CommitFact {
    CommitFact::new(text, EntityRef::new("Ada", "person"), "lives_in")
}

fn statements(response: &memoryos_sdk::RecallResponse) -> Vec<&str> {
    response
        .facts
        .iter()
        .map(|scored| scored.fact.statement.as_str())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn candidates_commit_and_supersede_with_as_of() {
    let server = Server::start().await;
    let root = server.root("host").await;
    let mem: MemoryOs = server.sdk(&root);
    let scope = Scope::agent("a1");

    let start = Utc::now() - ChronoDuration::days(30);
    let first = mem
        .commit(&Commit::new(
            scope.clone(),
            "chat",
            "Ada said she lives in Jakarta",
            vec![fact("Ada lives in Jakarta").valid_from(start)],
        ))
        .await
        .expect("first commit");
    assert_eq!(first.facts.len(), 1);
    assert_eq!(first.facts[0].status, FactStatus::Active);

    let candidates = mem
        .candidates(std::slice::from_ref(&scope), "Ada lives in Bandung", None)
        .await
        .expect("candidates");
    assert_eq!(candidates[0].fact.id, first.facts[0].id);

    let moved = Utc::now() - ChronoDuration::days(10);
    let second = mem
        .commit(&Commit::new(
            scope.clone(),
            "chat",
            "Ada said she moved to Bandung",
            vec![fact("Ada lives in Bandung")
                .valid_from(moved)
                .supersedes(candidates[0].fact.id)],
        ))
        .await
        .expect("second commit");
    assert_eq!(second.facts[0].status, FactStatus::Active);

    let now = mem
        .recall(scope.clone(), "Ada lives")
        .await
        .expect("recall");
    assert_eq!(statements(&now), ["Ada lives in Bandung"]);

    let then = mem
        .recall(scope.clone(), "Ada lives")
        .as_of(Utc::now() - ChronoDuration::days(20))
        .await
        .expect("recall as of");
    assert_eq!(statements(&then), ["Ada lives in Jakarta"]);

    let old = mem.get_fact(first.facts[0].id).await.expect("old fact");
    assert_eq!(old.superseded_by, Some(second.facts[0].id));
    assert_eq!(old.valid_to, Some(moved));
    let history = mem.fact_history(old.id).await.expect("history");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].id, old.id);

    // Superseding what is no longer current is refused, and nothing is left behind.
    let refused = mem
        .commit(&Commit::new(
            scope.clone(),
            "chat",
            "stale",
            vec![fact("Ada lives in Medan").supersedes(old.id)],
        ))
        .await;
    assert!(
        matches!(refused, Err(Error::InvalidInput { .. })),
        "{refused:?}"
    );

    // A corroboration links a source and creates nothing.
    let repeat = mem
        .commit(&Commit::new(
            scope.clone(),
            "chat",
            "Ada said it again",
            vec![fact("Ada lives in Bandung").corroborates(second.facts[0].id)],
        ))
        .await
        .expect("corroborate");
    assert_eq!(repeat.facts[0].id, second.facts[0].id);
    let source_count = mem
        .get_fact(second.facts[0].id)
        .await
        .expect("fact")
        .source_episode_ids
        .len();
    assert_eq!(source_count, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_trust_in_user_scope_is_quarantined_until_approved() {
    let server = Server::start().await;
    let root = server.root("quarantine").await;
    let mem = server.sdk(&root);
    let user = Scope::user("u1");

    let committed = mem
        .commit(&Commit::new(
            user.clone(),
            "web",
            "a page claimed this",
            vec![fact("Ada owes a large debt")],
        ))
        .await
        .expect("commit");
    let id = committed.facts[0].id;
    assert_eq!(committed.facts[0].status, FactStatus::Quarantined);

    let recalled = mem.recall(user.clone(), "Ada debt").await.expect("recall");
    assert!(recalled.facts.is_empty(), "poison reached recall");
    let candidates = mem
        .candidates(std::slice::from_ref(&user), "Ada debt", None)
        .await
        .expect("candidates");
    assert!(candidates.is_empty());
    let subject = mem.get_fact(id).await.expect("fact").subject_id;
    let graph = mem
        .neighbors(subject, std::slice::from_ref(&user), None, None)
        .await
        .expect("graph");
    assert!(graph.edges.is_empty(), "poison reached the graph");

    let listed = mem
        .quarantine(std::slice::from_ref(&user), None)
        .await
        .expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, id);

    let approved = mem.approve(id).await.expect("approve");
    assert_eq!(approved.status, FactStatus::Active);
    let recalled = mem.recall(user.clone(), "Ada debt").await.expect("recall");
    assert_eq!(statements(&recalled), ["Ada owes a large debt"]);
    assert!(mem
        .quarantine(std::slice::from_ref(&user), None)
        .await
        .expect("list")
        .is_empty());

    // A second one is rejected, and stays out.
    let other = mem
        .commit(&Commit::new(
            user.clone(),
            "web",
            "another claim",
            vec![fact("Ada robbed a bank")],
        ))
        .await
        .expect("commit");
    let rejected = mem.reject(other.facts[0].id).await.expect("reject");
    assert_eq!(rejected.status, FactStatus::Rejected);
    let recalled = mem.recall(user, "Ada bank").await.expect("recall");
    assert!(!statements(&recalled).contains(&"Ada robbed a bank"));
}

#[tokio::test(flavor = "multi_thread")]
async fn approval_held_on_a_moved_chain_asks_for_another_review() {
    let server = Server::start().await;
    let root = server.root("held").await;
    let mem = server.sdk(&root);
    let user = Scope::user("u1");

    let original = server
        .seed_fact("held", &user, Trust::User, "Ada lives in Jakarta", None)
        .await;
    let pending = mem
        .commit(&Commit::new(
            user.clone(),
            "chat",
            "a claim",
            // Later than the head below, so approving it really closes that head.
            vec![fact("Ada lives in Bandung")
                .valid_from(Utc::now() + ChronoDuration::hours(1))
                .supersedes(original)],
        ))
        .await
        .expect("commit");
    let id = pending.facts[0].id;
    assert_eq!(pending.facts[0].status, FactStatus::Quarantined);
    assert_eq!(
        mem.get_fact(id).await.expect("fact").pending_supersede_of,
        Some(original)
    );

    // While it waits, the user says something else, which outranks it.
    let head = server
        .seed_fact(
            "held",
            &user,
            Trust::User,
            "Ada lives in Medan",
            Some(original),
        )
        .await;

    let (status, reply) = server
        .call(
            "POST",
            &format!("/v1/quarantine/{id}/approve"),
            Some(&root),
            None,
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    let message = reply["error"]["message"].as_str().expect("message");
    assert!(message.contains("review it again"), "{message}");

    let held = mem.get_fact(id).await.expect("fact");
    assert_eq!(held.status, FactStatus::Quarantined);
    assert_eq!(held.pending_supersede_of, Some(head));
    let current = mem.get_fact(head).await.expect("head");
    assert!(
        current.valid_to.is_none(),
        "the higher-trust fact stays open"
    );

    // A second approval means the admin has now seen that fact.
    let approved = mem.approve(id).await.expect("second approval");
    assert_eq!(approved.status, FactStatus::Active);
    assert_eq!(
        mem.get_fact(head).await.expect("head").superseded_by,
        Some(id)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn purge_removes_a_source_and_restores_what_it_replaced() {
    let server = Server::start().await;
    let root = server.root("purge").await;
    let mem = server.sdk(&root);
    let scope = Scope::agent("a1");

    let good = mem
        .commit(&Commit::new(
            scope.clone(),
            "chat",
            "true",
            vec![fact("Ada lives in Jakarta")],
        ))
        .await
        .expect("commit");
    let poison = mem
        .commit(&Commit::new(
            scope.clone(),
            "web",
            "poison",
            vec![fact("Ada lives in Atlantis").supersedes(good.facts[0].id)],
        ))
        .await
        .expect("commit");
    let recalled = mem
        .recall(scope.clone(), "Ada lives")
        .await
        .expect("recall");
    assert_eq!(statements(&recalled), ["Ada lives in Atlantis"]);

    let report = mem.purge_episode(poison.episode_id).await.expect("purge");
    assert_eq!(report.facts_deleted, 1);
    assert_eq!(report.episodes_deleted, 1);

    let recalled = mem.recall(scope, "Ada lives").await.expect("recall");
    assert_eq!(statements(&recalled), ["Ada lives in Jakarta"]);
    assert!(matches!(
        mem.get_episode(poison.episode_id).await,
        Err(Error::NotFound { .. })
    ));
    assert!(matches!(
        mem.get_fact(poison.facts[0].id).await,
        Err(Error::NotFound { .. })
    ));

    // A refused purge of a missing episode is a plain 404.
    assert!(matches!(
        mem.purge_episode(random_uuid()).await,
        Err(Error::NotFound { .. })
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn deleting_a_user_removes_everything_under_their_scope() {
    let server = Server::start().await;
    let root = server.root("delete").await;
    let mem = server.sdk(&root);
    let (fact_id, _, episode) = server.commit_one(&root, "user:gone", "Ada sails").await;
    let (kept, _, _) = server.commit_one(&root, "user:stays", "Ada sails").await;

    let report = mem.delete_user_memory("gone").await.expect("delete");
    assert_eq!(report.facts_deleted, 1);
    assert!(matches!(
        mem.get_fact(fact_id).await,
        Err(Error::NotFound { .. })
    ));
    assert!(matches!(
        mem.get_episode(episode).await,
        Err(Error::NotFound { .. })
    ));
    assert!(mem.get_fact(kept).await.is_ok());

    let invalid = mem.delete_user_memory("bad id").await;
    assert!(
        matches!(invalid, Err(Error::InvalidInput { .. })),
        "{invalid:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn invalidated_facts_leave_recall() {
    let server = Server::start().await;
    let root = server.root("invalidate").await;
    let mem = server.sdk(&root);
    let (id, _, _) = server
        .commit_one(&root, "agent:a1", "Ada sails boats")
        .await;
    let invalidated = mem.invalidate_fact(id, "wrong").await.expect("invalidate");
    assert!(invalidated.invalidated_at.is_some());
    assert!(server
        .recall_statements(&root, "agent:a1", "sails")
        .await
        .is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_key_cannot_invalidate_a_fact_above_its_max_trust() {
    let server = Server::start().await;
    let tenant = crate::common::unique("invalidate-trust");
    let root = server.root(&tenant).await;
    let (id, _, _) = server
        .commit_one(&root, "agent:a1", "Ada sails boats")
        .await;

    let low = server
        .key(&tenant, &["agent:*"], memoryos_core::Trust::External, false)
        .await;
    assert!(server.sdk(&low).invalidate_fact(id, "bad").await.is_err());
    let fact = server.fact_json(&root, id).await;
    assert!(fact["invalidated_at"].is_null());

    let invalidated = server.sdk(&root).invalidate_fact(id, "wrong").await;
    assert!(invalidated.expect("root may").invalidated_at.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_source_of_a_commit_can_be_read() {
    let server = Server::start().await;
    let root = server.root("source").await;
    let (_, _, episode) = server.commit_one(&root, "agent:a1", "Ada sails").await;
    let (status, read) = server
        .call("GET", &format!("/v1/episodes/{episode}"), Some(&root), None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(read["processed_at"].is_string(), "{read}");
}

#[tokio::test(flavor = "multi_thread")]
async fn blocks_are_versioned_and_keep_their_limit() {
    let server = Server::start().await;
    let root = server.root("blocks").await;
    let mem = server.sdk(&root);

    let created = mem
        .put_block("a1", "notes", &PutBlock::create("one").max_chars(10))
        .await
        .expect("create");
    assert_eq!(created.version, 1);
    assert_eq!(created.max_chars, 10);

    let conflict = mem
        .put_block("a1", "notes", &PutBlock::create("again"))
        .await;
    assert!(
        matches!(conflict, Err(Error::VersionConflict { .. })),
        "{conflict:?}"
    );

    let updated = mem
        .put_block("a1", "notes", &PutBlock::update("two", 1))
        .await
        .expect("update");
    assert_eq!(updated.version, 2);
    assert_eq!(updated.max_chars, 10, "an edit must not reset the limit");

    let stale = mem
        .put_block("a1", "notes", &PutBlock::update("three", 1))
        .await;
    assert!(
        matches!(stale, Err(Error::VersionConflict { .. })),
        "{stale:?}"
    );

    let too_large = mem
        .put_block("a1", "notes", &PutBlock::update("x".repeat(11), 2))
        .await;
    assert!(
        matches!(too_large, Err(Error::BlockTooLarge { .. })),
        "{too_large:?}"
    );

    let read = mem.get_block("a1", "notes").await.expect("read");
    assert_eq!(read.content, "two");
    assert_eq!(mem.list_blocks("a1").await.expect("list").len(), 1);
    let versions = mem.block_versions("a1", "notes").await.expect("versions");
    assert_eq!(versions.len(), 2);

    let missing = mem.get_block("a1", "absent").await;
    assert!(
        matches!(missing, Err(Error::NotFound { .. })),
        "{missing:?}"
    );
    let default = mem
        .put_block("a1", "fresh", &PutBlock::create("x"))
        .await
        .expect("default limit");
    assert_eq!(default.max_chars, 2000);
}

#[tokio::test(flavor = "multi_thread")]
async fn neighbors_returns_the_graph_and_caps_edges_per_request() {
    let server = Server::start().await;
    let root = server.root("graph").await;
    let scope = Scope::agent("hub");
    let mem = server.sdk(&root);

    // A hub with more edges than one request may return.
    for batch in 0..11 {
        let facts: Vec<CommitFact> = (0..50)
            .map(|index| {
                CommitFact::new(
                    format!("Ada knows person {batch}-{index}"),
                    EntityRef::new("Ada", "person"),
                    "knows",
                )
                .object(EntityRef::new(format!("Friend {batch}-{index}"), "person"))
            })
            .collect();
        mem.commit(&Commit::new(scope.clone(), "seed", "seed", facts))
            .await
            .expect("commit");
    }
    let hit = mem
        .recall(scope.clone(), "Ada knows person")
        .limit(1)
        .await
        .expect("recall");
    let hub = hit.facts[0].fact.subject_id;

    let graph = mem
        .neighbors(hub, std::slice::from_ref(&scope), Some(1), None)
        .await
        .expect("neighbors");
    assert_eq!(graph.root, hub);
    assert_eq!(graph.edges.len(), 500);
    assert!(graph.entities.len() > 1);

    for bad in [0, 6] {
        let invalid = mem
            .neighbors(hub, std::slice::from_ref(&scope), Some(bad), None)
            .await;
        assert!(
            matches!(invalid, Err(Error::InvalidInput { .. })),
            "{invalid:?}"
        );
    }
    let (status, reply) = server
        .call(
            "GET",
            &format!("/v1/entities/{hub}/neighbors?scopes=agent:hub&as_of=yesterday"),
            Some(&root),
            None,
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    let (status, reply) = server
        .call(
            "GET",
            &format!("/v1/entities/{hub}/neighbors"),
            Some(&root),
            None,
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
}

#[tokio::test(flavor = "multi_thread")]
async fn commit_validates_its_limits_and_shape() {
    let server = Server::start().await;
    let root = server.root("commits").await;
    let many: Vec<_> = (0..51)
        .map(|i| statement(&format!("s{i}"), "Ada"))
        .collect();
    let bad_bodies = [
        json!({"scope": "agent:a", "source": "s", "content": "c", "facts": many}),
        json!({"scope": "agent:a", "source": "s", "content": "x".repeat(100_001), "facts": []}),
        json!({"scope": "team:a", "source": "s", "content": "c", "facts": []}),
        json!({"scope": "agent:a", "source": "s", "content": "c", "facts": [{
            "statement": "s", "subject": {"name": "A", "kind": "k"}, "predicate": "p",
            "supersede": random_uuid(),
        }]}),
        json!({"scope": "agent:a", "source": "s", "content": "c", "facts": [{
            "statement": "s", "subject": {"name": "A", "kind": "k"}, "predicate": "p",
            "supersedes": random_uuid(), "corroborates": random_uuid(),
        }]}),
    ];
    for body in bad_bodies {
        let (status, reply) = server
            .call("POST", "/v1/commits", Some(&root), Some(body))
            .await;
        assert_error(status, &reply, 400, "invalid_input");
    }
    let (status, reply) = server
        .call(
            "POST",
            "/v1/commits",
            Some(&root),
            Some(json!({"scope": "agent:a", "source": "s", "content": "c",
                        "idempotency_key": "k1", "facts": [statement("Ada sails", "Ada")]})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let first = uuid_at(&reply, "/facts/0/id");
    let (status, replay) = server
        .call(
            "POST",
            "/v1/commits",
            Some(&root),
            Some(json!({"scope": "agent:a", "source": "s", "content": "c",
                        "idempotency_key": "k1", "facts": [statement("Ada sails", "Ada")]})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    assert_eq!(uuid_at(&replay, "/facts/0/id"), first);
}
