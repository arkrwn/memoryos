//! `memoryos tui` against a real server: the keys a person presses, through to the server and back.

use memoryos_server::tui::{apply, Api, App, Effect, Tab};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use reqwest::StatusCode;
use serde_json::{json, Value};

use crate::common::{Key, Server};

async fn commit(server: &Server, key: &Key, scope: &str, statement: &str) {
    let body = json!({
        "scope": scope, "source": "test", "content": "said in a test",
        "facts": [{"statement": statement, "subject": {"name": "Ada", "kind": "person"}, "predicate": "note"}],
    });
    let (status, value): (StatusCode, Value) = server
        .call("POST", "/v1/commits", Some(key), Some(body))
        .await;
    assert_eq!(status, StatusCode::OK, "{value}");
}

/// Moves to `scope` with `]`, as a person would.
async fn choose_scope(app: &mut App, api: &Api, scope: &str) {
    for _ in 0..app.scopes.len() {
        if app.scope().as_deref() == Some(scope) {
            return;
        }
        press(app, api, KeyCode::Char(']')).await;
    }
    panic!("no scope {scope}");
}

/// Presses a key and performs what it asks for, as the event loop does.
async fn press(app: &mut App, api: &Api, code: KeyCode) {
    let effect = app.key(KeyEvent::new(code, KeyModifiers::NONE));
    assert!(apply(app, api, effect).await.expect("effect"));
}

#[tokio::test]
async fn facts_ask_detail_and_review_work_against_the_server() {
    let server = Server::start().await;
    let root = server.root("acme").await;
    commit(&server, &root, "agent:a", "Ada lives in Bandung").await;
    commit(&server, &root, "user:u", "Ada wants dark mode").await;
    let api = Api::new(server.url.clone(), root.token.clone()).expect("api");
    let mut app = App::new();

    apply(&mut app, &api, Effect::LoadScopes)
        .await
        .expect("load");
    choose_scope(&mut app, &api, "agent:a").await;
    assert_eq!(app.facts.len(), 1);
    assert_eq!(app.facts[0].fact.statement, "Ada lives in Bandung");

    press(&mut app, &api, KeyCode::Enter).await;
    let detail = app.detail.as_ref().expect("detail open");
    assert_eq!(detail.fact.statement, "Ada lives in Bandung");
    assert_eq!(detail.sources.len(), 1, "the source episode is shown");
    press(&mut app, &api, KeyCode::Esc).await;

    press(&mut app, &api, KeyCode::Char('3')).await;
    press(&mut app, &api, KeyCode::Char('/')).await;
    for c in "where does Ada live".chars() {
        press(&mut app, &api, KeyCode::Char(c)).await;
    }
    press(&mut app, &api, KeyCode::Enter).await;
    assert_eq!(app.tab(), Tab::Ask);
    assert_eq!(app.answers[0].fact.statement, "Ada lives in Bandung");

    // The user scope holds a fact waiting for review. Approving it makes it current.
    choose_scope(&mut app, &api, "user:u").await;
    press(&mut app, &api, KeyCode::Char('2')).await;
    assert_eq!(app.review.len(), 1);
    press(&mut app, &api, KeyCode::Char('a')).await;
    assert!(app.review.is_empty());
    press(&mut app, &api, KeyCode::Char('1')).await;
    assert_eq!(app.facts[0].fact.statement, "Ada wants dark mode");
}

#[tokio::test]
async fn a_server_error_is_reported_not_hidden() {
    let server = Server::start().await;
    let api = Api::new(server.url.clone(), "not-a-key".into()).expect("api");
    let mut app = App::new();
    let error = apply(&mut app, &api, Effect::LoadScopes)
        .await
        .expect_err("a bad key fails");
    assert!(error.contains("401"), "{error}");
}
