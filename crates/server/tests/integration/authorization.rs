//! The authorization matrix of SPEC section 6.

use memoryos_core::{Scope, Trust};
use memoryos_sdk::{Error, MemoryOs, Options};
use memoryos_server::routes::served_routes;
use reqwest::header::WWW_AUTHENTICATE;
use reqwest::StatusCode;
use serde_json::json;
use uuid::Uuid;

use crate::common::{assert_error, random_uuid, statement, uuid_at, Key, Server};

fn concrete(path: &str) -> String {
    path.replace("{id}", &random_uuid().to_string())
        .replace("{agent_id}", "a1")
        .replace("{user_id}", "u1")
        .replace("{label}", "l1")
}

#[tokio::test(flavor = "multi_thread")]
async fn no_token_and_bad_tokens_get_401_on_every_protected_route() {
    let server = Server::host().await;
    let bad = Key {
        id: random_uuid(),
        token: "mos_notARealTokenAtAllButTheRightLengthForOne1234".to_owned(),
    };
    let protected: Vec<_> = served_routes()
        .into_iter()
        .filter(|(_, _, public)| !public)
        .collect();
    assert!(protected.len() > 20);
    for (method, path, _) in protected {
        let target = concrete(path);
        for key in [None, Some(&bad)] {
            let (status, headers, body) = server.call_full(method, &target, key, None).await;
            assert_error(status, &body, 401, "unauthorized");
            assert_eq!(headers[WWW_AUTHENTICATE], "Bearer", "{method} {path}");
        }
    }
    // A scheme other than Bearer is not a token.
    let response = server
        .raw_client()
        .post(format!("{}/v1/recall", server.url))
        .header("authorization", "Basic abc")
        .send()
        .await
        .expect("send");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test(flavor = "multi_thread")]
async fn writing_a_block_needs_agent_trust_and_max_chars_is_capped() {
    let server = Server::host().await;
    let path = "/v1/agents/a1/blocks/notes";
    let body = json!({"content": "x"});

    let external = server
        .key("blocks", &["agent:a1"], Trust::External, false)
        .await;
    let (status, reply) = server
        .call("PUT", path, Some(&external), Some(body.clone()))
        .await;
    assert_error(status, &reply, 403, "forbidden");

    let agent = server
        .key("blocks", &["agent:a1"], Trust::Agent, false)
        .await;
    let (status, reply) = server.call("PUT", path, Some(&agent), Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    // Reading stays open to the key.
    let (status, _) = server.call("GET", path, Some(&external), None).await;
    assert_eq!(status, StatusCode::OK);

    let over = json!({"content": "x", "max_chars": 100_001});
    let (status, reply) = server
        .call("PUT", "/v1/agents/a1/blocks/big", Some(&agent), Some(over))
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    let at = json!({"content": "x", "max_chars": 100_000});
    let (status, reply) = server
        .call("PUT", "/v1/agents/a1/blocks/big", Some(&agent), Some(at))
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
}

#[tokio::test(flavor = "multi_thread")]
async fn admin_rights_stop_at_the_key_max_trust() {
    let server = Server::host().await;
    let tenant = "admintrust";
    let root = server.root(tenant).await;
    let low_admin = server.key(tenant, &["*"], Trust::Agent, true).await;
    let scope = Scope::user("u1");
    let fact = server
        .seed_fact(tenant, &scope, Trust::User, "user said so", None)
        .await;
    let episode = uuid_at(
        &server.fact_json(&root, fact).await,
        "/source_episode_ids/0",
    );

    for action in ["approve", "reject"] {
        let path = format!("/v1/quarantine/{fact}/{action}");
        let (status, reply) = server.call("POST", &path, Some(&low_admin), None).await;
        assert_error(status, &reply, 403, "forbidden");
        // Past the trust check, the same call fails on the fact's status instead.
        let (status, reply) = server.call("POST", &path, Some(&root), None).await;
        assert_error(status, &reply, 400, "invalid_input");
    }
    let purge = format!("/v1/episodes/{episode}/purge");
    let (status, reply) = server.call("POST", &purge, Some(&low_admin), None).await;
    assert_error(status, &reply, 403, "forbidden");
    let (status, reply) = server
        .call("DELETE", "/v1/users/u1/memory", Some(&low_admin), None)
        .await;
    assert_error(status, &reply, 403, "forbidden");
    assert_eq!(server.fact_json(&root, fact).await["status"], "active");

    // At its own level the same key works, and a held agent fact is what it is for.
    let (held, status, _) = server
        .commit_one(&low_admin, "user:u2", "agent claim")
        .await;
    assert_eq!(status, "quarantined");
    let (status, reply) = server
        .call(
            "POST",
            &format!("/v1/quarantine/{held}/reject"),
            Some(&low_admin),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let (status, reply) = server
        .call("DELETE", "/v1/users/u2/memory", Some(&low_admin), None)
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");

    let (status, reply) = server.call("POST", &purge, Some(&root), None).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
}

#[tokio::test(flavor = "multi_thread")]
async fn only_user_trust_may_invalidate_in_user_and_shared_scopes() {
    let server = Server::host().await;
    let tenant = "retract";
    let agent = server.key(tenant, &["*"], Trust::Agent, false).await;
    let user = server.key(tenant, &["*"], Trust::User, false).await;
    let body = json!({"reason": "wrong"});
    let invalidate = |id: Uuid| format!("/v1/facts/{id}/invalidate");

    for scope in ["user:u1", "shared:s1"] {
        // Equal trust: the agent wrote this one itself.
        let (id, _, _) = server.commit_one(&agent, scope, "agent claim").await;
        let (status, reply) = server
            .call("POST", &invalidate(id), Some(&agent), Some(body.clone()))
            .await;
        assert_error(status, &reply, 403, "forbidden");
        let (status, reply) = server
            .call("POST", &invalidate(id), Some(&user), Some(body.clone()))
            .await;
        assert_eq!(status, StatusCode::OK, "{reply}");
    }
    let (id, _, _) = server.commit_one(&agent, "agent:a1", "own note").await;
    let (status, reply) = server
        .call("POST", &invalidate(id), Some(&agent), Some(body))
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_revoked_key_stops_working_on_the_next_request() {
    use memoryos_core::MemoryStore;
    let server = Server::host().await;
    let key = server.root("revoke").await;
    let (status, _) = server
        .call("GET", "/v1/agents/a1/blocks", Some(&key), None)
        .await;
    assert_eq!(status, StatusCode::OK);
    server.store.revoke_api_key(key.id).await.expect("revoke");
    let (status, body) = server
        .call("GET", "/v1/agents/a1/blocks", Some(&key), None)
        .await;
    assert_error(status, &body, 401, "unauthorized");
}

/// One tenant's facts, seen below by a key with only `user:a`.
struct Seeded {
    root: Key,
    narrow: Key,
    fact: Uuid,
    entity: Uuid,
    episode: Uuid,
}

async fn seed_user_b(server: &Server) -> Seeded {
    let root = server.root("scoped").await;
    let narrow = server.key("scoped", &["user:a"], Trust::Agent, false).await;
    let (fact, _, episode) = server.commit_one(&root, "user:b", "Ada owns a boat").await;
    let body = server.fact_json(&root, fact).await;
    Seeded {
        entity: uuid_at(&body, "/subject_id"),
        root,
        narrow,
        fact,
        episode,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn named_scopes_outside_the_key_get_403() {
    let server = Server::host().await;
    let s = seed_user_b(&server).await;
    let cases = [
        ("/v1/recall", json!({"scopes": ["user:b"], "query": "boat"})),
        (
            "/v1/candidates",
            json!({"scopes": ["user:b"], "statement": "boat"}),
        ),
        (
            "/v1/commits",
            json!({"scope": "user:b", "source": "t", "content": "c",
                   "facts": [statement("x", "Ada")]}),
        ),
    ];
    for (path, body) in cases {
        let (status, reply) = server.call("POST", path, Some(&s.narrow), Some(body)).await;
        assert_error(status, &reply, 403, "forbidden");
    }

    // One forbidden scope among allowed ones refuses the whole request.
    let (status, reply) = server
        .call(
            "POST",
            "/v1/recall",
            Some(&s.narrow),
            Some(json!({"scopes": ["user:a", "user:b"], "query": "boat"})),
        )
        .await;
    assert_error(status, &reply, 403, "forbidden");

    let path = format!("/v1/entities/{}/neighbors?scopes=user:b", s.entity);
    let (status, reply) = server.call("GET", &path, Some(&s.narrow), None).await;
    assert_error(status, &reply, 403, "forbidden");
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_read_by_id_across_scopes_get_404_with_the_same_body_as_absent_rows() {
    let server = Server::host().await;
    let s = seed_user_b(&server).await;
    let reads = [
        ("GET", format!("/v1/facts/{}", s.fact), None),
        ("GET", format!("/v1/facts/{}/history", s.fact), None),
        (
            "POST",
            format!("/v1/facts/{}/invalidate", s.fact),
            Some(json!({"reason": "r"})),
        ),
        ("GET", format!("/v1/entities/{}", s.entity), None),
        (
            "GET",
            format!("/v1/entities/{}/neighbors?scopes=user:a", s.entity),
            None,
        ),
        ("GET", format!("/v1/episodes/{}", s.episode), None),
    ];
    let (_, absent) = server
        .call(
            "GET",
            &format!("/v1/facts/{}", random_uuid()),
            Some(&s.narrow),
            None,
        )
        .await;
    for (method, path, body) in reads {
        let (status, reply) = server.call(method, &path, Some(&s.narrow), body).await;
        assert_error(status, &reply, 404, "not_found");
        assert_eq!(reply, absent, "{method} {path} differs from an absent row");
    }
    // The invalidate above must not have gone through.
    let fact = server.fact_json(&s.root, s.fact).await;
    assert!(fact["invalidated_at"].is_null());

    // The same key reads the same rows in a scope it has.
    let wide = server.key("scoped", &["user:*"], Trust::Agent, false).await;
    let (status, _) = server
        .call("GET", &format!("/v1/facts/{}", s.fact), Some(&wide), None)
        .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn blocks_require_the_agent_scope() {
    let server = Server::host().await;
    let root = server.root("blocks").await;
    let narrow = server
        .key("blocks", &["agent:mine", "user:*"], Trust::Agent, false)
        .await;
    let (status, _) = server
        .call(
            "PUT",
            "/v1/agents/theirs/blocks/notes",
            Some(&root),
            Some(json!({"content": "x"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    let cases = [
        ("GET", "/v1/agents/theirs/blocks", None),
        ("GET", "/v1/agents/theirs/blocks/notes", None),
        ("GET", "/v1/agents/theirs/blocks/notes/versions", None),
        (
            "PUT",
            "/v1/agents/theirs/blocks/notes",
            Some(json!({"content": "y", "expected_version": 1})),
        ),
    ];
    for (method, path, body) in cases {
        let (status, reply) = server.call(method, path, Some(&narrow), body).await;
        assert_error(status, &reply, 403, "forbidden");
    }
    let (status, _) = server
        .call("GET", "/v1/agents/mine/blocks", Some(&narrow), None)
        .await;
    assert_eq!(status, StatusCode::OK);

    // An agent id that cannot form a scope is refused, not treated as a miss.
    let (status, reply) = server
        .call("GET", "/v1/agents/bad%20id/blocks", Some(&root), None)
        .await;
    assert_error(status, &reply, 400, "invalid_input");
}

#[tokio::test(flavor = "multi_thread")]
async fn non_admins_are_blocked_from_admin_routes() {
    let server = Server::host().await;
    let root = server.root("admin").await;
    let plain = server.key("admin", &["*"], Trust::System, false).await;
    let (fact, status_text, episode) = server.commit_one(&root, "user:u", "Ada sails").await;
    assert_eq!(status_text, "quarantined");

    let cases = [
        ("GET", "/v1/quarantine?scopes=user:u".to_owned()),
        ("POST", format!("/v1/quarantine/{fact}/approve")),
        ("POST", format!("/v1/quarantine/{fact}/reject")),
        ("POST", format!("/v1/episodes/{episode}/purge")),
        ("DELETE", "/v1/users/u/memory".to_owned()),
    ];
    for (method, path) in cases {
        let (status, reply) = server.call(method, &path, Some(&plain), None).await;
        assert_error(status, &reply, 403, "forbidden");
    }
    assert_eq!(
        server.fact_json(&root, fact).await["status"],
        "quarantined",
        "a refused approval must change nothing"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn admin_keys_still_respect_scope_patterns() {
    let server = Server::host().await;
    let root = server.root("admin2").await;
    let admin_a = server.key("admin2", &["user:a"], Trust::System, true).await;
    let (fact, _, episode) = server.commit_one(&root, "user:b", "Ada sails").await;

    let (status, reply) = server
        .call("GET", "/v1/quarantine?scopes=user:b", Some(&admin_a), None)
        .await;
    assert_error(status, &reply, 403, "forbidden");
    let (status, reply) = server
        .call("DELETE", "/v1/users/b/memory", Some(&admin_a), None)
        .await;
    assert_error(status, &reply, 403, "forbidden");
    for path in [
        format!("/v1/quarantine/{fact}/approve"),
        format!("/v1/quarantine/{fact}/reject"),
        format!("/v1/episodes/{episode}/purge"),
    ] {
        let (status, reply) = server.call("POST", &path, Some(&admin_a), None).await;
        assert_error(status, &reply, 404, "not_found");
    }
    assert_eq!(server.fact_json(&root, fact).await["status"], "quarantined");
}

/// The ids `GET /v1/quarantine` returns, sorted.
async fn quarantine_ids(server: &Server, key: &Key, scopes: &str) -> Vec<Uuid> {
    let path = format!("/v1/quarantine?scopes={scopes}");
    let (status, reply) = server.call("GET", &path, Some(key), None).await;
    assert_eq!(status, StatusCode::OK, "{scopes}: {reply}");
    let mut ids: Vec<Uuid> = reply["facts"]
        .as_array()
        .expect("facts")
        .iter()
        .map(|fact| fact["id"].as_str().expect("id").parse().expect("uuid"))
        .collect();
    ids.sort();
    ids
}

#[tokio::test(flavor = "multi_thread")]
async fn a_quarantine_wildcard_returns_only_scopes_the_key_may_read() {
    let server = Server::host().await;
    let root = server.root("wild").await;
    let (in_a, status_text, _) = server.commit_one(&root, "user:a", "Ada sails").await;
    assert_eq!(status_text, "quarantined");
    let (in_ab, _, _) = server.commit_one(&root, "user:ab", "Abe sails").await;
    let (in_b, _, _) = server.commit_one(&root, "user:b", "Bob sails").await;
    let (in_shared, _, _) = server.commit_one(&root, "shared:a", "All sail").await;
    let sorted = |mut ids: Vec<Uuid>| {
        ids.sort();
        ids
    };

    let only_a = server.key("wild", &["user:a"], Trust::System, true).await;
    for scopes in ["user:*", "*", "user:a*", "user:a,user:*"] {
        assert_eq!(
            quarantine_ids(&server, &only_a, scopes).await,
            [in_a],
            "{scopes}"
        );
    }
    for scopes in ["user:b*", "shared:*", "user:b"] {
        let path = format!("/v1/quarantine?scopes={scopes}");
        let (status, reply) = server.call("GET", &path, Some(&only_a), None).await;
        assert_error(status, &reply, 403, "forbidden");
    }

    let a_prefix = server.key("wild", &["user:a*"], Trust::System, true).await;
    assert_eq!(
        quarantine_ids(&server, &a_prefix, "user:*").await,
        sorted(vec![in_a, in_ab])
    );

    assert_eq!(
        quarantine_ids(&server, &root, "user:*").await,
        sorted(vec![in_a, in_ab, in_b])
    );
    assert_eq!(
        quarantine_ids(&server, &root, "*").await,
        sorted(vec![in_a, in_ab, in_b, in_shared])
    );

    // Row-level security still applies to a key that allows everything.
    let other_tenant = server.root("wild_other").await;
    assert!(quarantine_ids(&server, &other_tenant, "*").await.is_empty());

    let (status, reply) = server
        .call("GET", "/v1/quarantine?scopes=team:*", Some(&root), None)
        .await;
    assert_error(status, &reply, 400, "invalid_input");
}

#[tokio::test(flavor = "multi_thread")]
async fn one_tenant_sees_nothing_of_another() {
    let server = Server::host().await;
    let alice = server.root("tenant_a").await;
    let bob = server.root("tenant_b").await;
    let (quarantined, _, episode) = server.commit_one(&alice, "user:u", "Ada sails boats").await;
    let (active, status_text, _) = server
        .commit_one(&alice, "agent:a1", "Ada sails ships")
        .await;
    assert_eq!(status_text, "active");
    let entity = uuid_at(&server.fact_json(&alice, active).await, "/subject_id");
    let (status, _) = server
        .call(
            "PUT",
            "/v1/agents/a1/blocks/notes",
            Some(&alice),
            Some(json!({"content": "private"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    // Bob's key has `*`, so only row-level security stands between him and these.
    for (method, path, body) in [
        ("GET", format!("/v1/facts/{active}"), None),
        ("GET", format!("/v1/facts/{active}/history"), None),
        (
            "POST",
            format!("/v1/facts/{active}/invalidate"),
            Some(json!({"reason": "r"})),
        ),
        ("GET", format!("/v1/episodes/{episode}"), None),
        ("POST", format!("/v1/episodes/{episode}/purge"), None),
        ("GET", format!("/v1/entities/{entity}"), None),
        (
            "GET",
            format!("/v1/entities/{entity}/neighbors?scopes=agent:a1"),
            None,
        ),
        (
            "POST",
            format!("/v1/quarantine/{quarantined}/approve"),
            None,
        ),
        ("POST", format!("/v1/quarantine/{quarantined}/reject"), None),
        ("GET", "/v1/agents/a1/blocks/notes".to_owned(), None),
    ] {
        let (status, reply) = server.call(method, &path, Some(&bob), body).await;
        assert_error(status, &reply, 404, "not_found");
    }

    assert!(server
        .recall_statements(&bob, "agent:a1", "sails")
        .await
        .is_empty());
    let (_, listed) = server
        .call("GET", "/v1/quarantine?scopes=user:u", Some(&bob), None)
        .await;
    assert_eq!(listed["facts"], json!([]));
    let (_, blocks) = server
        .call("GET", "/v1/agents/a1/blocks", Some(&bob), None)
        .await;
    assert_eq!(blocks["blocks"], json!([]));

    // A commit cannot point at another tenant's fact, and the error does not say it exists.
    let (status, reply) = server
        .call(
            "POST",
            "/v1/commits",
            Some(&bob),
            Some(json!({
                "scope": "agent:a1", "source": "t", "content": "c",
                "facts": [{"statement": "x", "subject": {"name": "A", "kind": "person"},
                           "predicate": "p", "supersedes": active}],
            })),
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");

    // Deleting a user in one tenant leaves the same user id in another.
    let (status, _) = server
        .call("DELETE", "/v1/users/u/memory", Some(&bob), None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        server.fact_json(&alice, quarantined).await["status"],
        "quarantined"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn episode_trust_above_the_key_maximum_gets_403() {
    let server = Server::worker_without_loop().await;
    let key = server.key("trust", &["*"], Trust::Agent, false).await;
    let new = |trust: &str| json!({"scope": "agent:a1", "source": "t", "trust": trust, "content": "text"});
    for trust in ["user", "system"] {
        let (status, reply) = server
            .call("POST", "/v1/episodes", Some(&key), Some(new(trust)))
            .await;
        assert_error(status, &reply, 403, "forbidden");
    }
    for trust in ["external", "agent"] {
        let (status, reply) = server
            .call("POST", "/v1/episodes", Some(&key), Some(new(trust)))
            .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{reply}");
    }
    // A scope outside the key is refused the same way.
    let narrow = server.key("trust", &["user:a"], Trust::System, false).await;
    let (status, reply) = server
        .call("POST", "/v1/episodes", Some(&narrow), Some(new("agent")))
        .await;
    assert_error(status, &reply, 403, "forbidden");
}

#[tokio::test(flavor = "multi_thread")]
async fn commit_trust_follows_the_key_and_a_request_cannot_raise_it() {
    let server = Server::host().await;
    let system = server.key("trust2", &["*"], Trust::System, false).await;
    let external = server.key("trust2", &["*"], Trust::External, false).await;

    let (fact, status_text, _) = server.commit_one(&system, "agent:a1", "Ada reads").await;
    assert_eq!(status_text, "active");
    assert_eq!(server.fact_json(&system, fact).await["trust"], "agent");

    let (fact, status_text, _) = server.commit_one(&external, "agent:a1", "Ada writes").await;
    assert_eq!(status_text, "quarantined");
    assert_eq!(server.fact_json(&system, fact).await["trust"], "external");

    let (status, reply) = server
        .call(
            "POST",
            "/v1/commits",
            Some(&system),
            Some(
                json!({"scope": "agent:a1", "source": "t", "content": "c", "trust": "system",
                        "facts": [statement("x", "Ada")]}),
            ),
        )
        .await;
    assert_error(status, &reply, 400, "invalid_input");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_idempotency_replay_in_another_scope_gets_400_and_reveals_nothing() {
    let server = Server::worker_without_loop().await;
    let root = server.root("replay").await;
    let secret = "private text about user b";
    let new = |scope: &str| {
        json!({"scope": scope, "source": "t", "trust": "agent", "content": secret,
               "idempotency_key": "shared-key"})
    };
    let (status, original) = server
        .call("POST", "/v1/episodes", Some(&root), Some(new("user:b")))
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);

    let narrow = server.key("replay", &["user:a"], Trust::Agent, false).await;
    let (status, reply) = server
        .call("POST", "/v1/episodes", Some(&narrow), Some(new("user:a")))
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    let text = reply.to_string();
    assert!(!text.contains(secret), "{text}");
    assert!(!text.contains("user:b"), "{text}");
    assert!(
        !text.contains(original["id"].as_str().expect("id")),
        "{text}"
    );

    // In its own scope a replay returns the first episode.
    let (status, replay) = server
        .call("POST", "/v1/episodes", Some(&root), Some(new("user:b")))
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(replay["id"], original["id"]);

    // The same rule through a commit, where the key reaches another scope's episode.
    let commit = |scope: &str| {
        json!({"scope": scope, "source": "t", "content": secret,
               "idempotency_key": "shared-key", "facts": [statement("x", "Ada")]})
    };
    let (status, reply) = server
        .call("POST", "/v1/commits", Some(&narrow), Some(commit("user:a")))
        .await;
    assert_error(status, &reply, 400, "invalid_input");
    assert!(!reply.to_string().contains(secret));
}

#[tokio::test(flavor = "multi_thread")]
async fn sessions_need_both_the_agent_scope_and_the_memory_scope() {
    let server = Server::worker_without_loop().await;
    let agent_only = server
        .key("sessions", &["agent:a1"], Trust::Agent, false)
        .await;
    let memory_only = server
        .key("sessions", &["user:u"], Trust::Agent, false)
        .await;
    let both = server
        .key("sessions", &["agent:a1", "user:u"], Trust::Agent, false)
        .await;
    let body = json!({"scope": "user:u"});
    for key in [&agent_only, &memory_only] {
        let (status, reply) = server
            .call(
                "POST",
                "/v1/agents/a1/sessions",
                Some(key),
                Some(body.clone()),
            )
            .await;
        assert_error(status, &reply, 403, "forbidden");
    }
    let (status, session) = server
        .call("POST", "/v1/agents/a1/sessions", Some(&both), Some(body))
        .await;
    assert_eq!(status, StatusCode::CREATED, "{session}");
    let id = uuid_at(&session, "/id");

    let messages = format!("/v1/sessions/{id}/messages");
    for key in [&agent_only, &memory_only] {
        let (status, reply) = server.call("GET", &messages, Some(key), None).await;
        assert_error(status, &reply, 404, "not_found");
        let (status, reply) = server
            .call("POST", &messages, Some(key), Some(json!({"text": "hi"})))
            .await;
        assert_error(status, &reply, 404, "not_found");
    }
    let (status, _) = server.call("GET", &messages, Some(&both), None).await;
    assert_eq!(status, StatusCode::OK);

    // A turn writes agent trust, so a key limited to external cannot run one.
    let external = server
        .key("sessions", &["agent:a1", "user:u"], Trust::External, false)
        .await;
    let (status, reply) = server
        .call(
            "POST",
            &messages,
            Some(&external),
            Some(json!({"text": "hi"})),
        )
        .await;
    assert_error(status, &reply, 403, "forbidden");
    assert_eq!(server.mock.llm_calls().await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn sdk_errors_are_typed() {
    let server = Server::host().await;
    let narrow = server.key("sdk", &["user:a"], Trust::Agent, false).await;
    let mem: MemoryOs = server.sdk(&narrow);

    let forbidden = mem.recall(Scope::user("b"), "x").await;
    assert!(
        matches!(forbidden, Err(Error::Forbidden { .. })),
        "{forbidden:?}"
    );
    let missing = mem.get_fact(random_uuid()).await;
    assert!(
        matches!(missing, Err(Error::NotFound { .. })),
        "{missing:?}"
    );

    let bad = MemoryOs::new(&server.url, "mos_wrong", Options::default()).expect("client");
    let denied = bad.get_fact(random_uuid()).await;
    assert!(
        matches!(denied, Err(Error::Unauthorized { .. })),
        "{denied:?}"
    );

    let unreachable = MemoryOs::connect("http://127.0.0.1:1", "k").await;
    assert!(matches!(unreachable, Err(Error::Transport(_))));
}
