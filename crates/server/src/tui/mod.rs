//! `memoryos tui`: the `memoryos ui` page's Facts, Review, Ask and Timeline views, in the terminal.
//!
//! It talks to the server with the same key from `credentials.json`, and every request goes
//! through the page's allowlist ([`crate::ui::upstream_path`]), so it can read and review but never
//! purge, retract, delete or write facts.

mod app;

use std::time::Duration;

use memoryos_core::{Episode, Fact, FactView, ScopeSummary, ScoredFact};
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use reqwest::Method;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

pub use app::{App, Decision, Detail, Effect, Tab};

const TIMEOUT: Duration = Duration::from_secs(30);
/// How many facts a list view loads. The web page uses the same order of magnitude.
const LIST_LIMIT: &str = "500";

/// The server, reached only through the page's allowlist.
pub struct Api {
    base: String,
    key: String,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct Scopes {
    scopes: Vec<ScopeSummary>,
}

#[derive(Deserialize)]
struct Facts<T> {
    facts: Vec<T>,
}

impl Api {
    pub fn new(base: String, key: String) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| format!("cannot start the HTTP client: {e}"))?;
        Ok(Self {
            base: base.trim_end_matches('/').to_owned(),
            key,
            client,
        })
    }

    /// `target` is a page path such as `facts` or `quarantine/{id}/approve`.
    async fn send<T: DeserializeOwned>(
        &self,
        method: Method,
        target: &str,
        query: &[(&str, &str)],
        body: Option<Value>,
    ) -> Result<T, String> {
        let path = crate::ui::upstream_path(&method, target).ok_or_else(|| {
            format!("the terminal view cannot make the request {method} {target}")
        })?;
        let mut request = self
            .client
            .request(method, format!("{}{path}", self.base))
            .bearer_auth(&self.key)
            .query(query);
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.map_err(|_| {
            "cannot reach the MemoryOS server. Start it with `memoryos setup`".to_owned()
        })?;
        let status = response.status();
        if !status.is_success() {
            let body: Value = response.json().await.unwrap_or(Value::Null);
            let message = body["error"]["message"].as_str().unwrap_or("no message");
            return Err(format!(
                "the server answered {status}: {}",
                app::clean(message)
            ));
        }
        response
            .json()
            .await
            .map_err(|e| format!("the server's answer could not be read: {e}"))
    }

    async fn get<T: DeserializeOwned>(
        &self,
        target: &str,
        query: &[(&str, &str)],
    ) -> Result<T, String> {
        self.send(Method::GET, target, query, None).await
    }
}

/// Performs one effect and puts its result into `app`. Returns false when the app should stop.
pub async fn apply(app: &mut App, api: &Api, effect: Effect) -> Result<bool, String> {
    match effect {
        Effect::None => {}
        Effect::Quit => return Ok(false),
        Effect::LoadScopes => {
            let scopes: Scopes = api.get("scopes", &[]).await?;
            app.set_scopes(scopes.scopes);
            load_tab(app, api).await?;
        }
        Effect::LoadTab => load_tab(app, api).await?,
        Effect::Recall(question) => {
            let Some(scope) = app.scope() else {
                return Ok(true);
            };
            let body = json!({"scopes": [scope], "query": question, "limit": 10});
            let answer: Facts<ScoredFact> =
                api.send(Method::POST, "recall", &[], Some(body)).await?;
            app.set_answers(question, answer.facts);
        }
        Effect::Open(id) => {
            let fact: Fact = api.get(&format!("facts/{id}"), &[]).await?;
            let mut history: Facts<Fact> = api.get(&format!("facts/{id}/history"), &[]).await?;
            history.facts.sort_by_key(|item| item.valid_from);
            let mut sources = Vec::new();
            for episode in fact.source_episode_ids.iter().take(3) {
                // A source that is gone or out of reach leaves the rest of the detail intact.
                if let Ok(found) = api
                    .get::<Episode>(&format!("episodes/{episode}"), &[])
                    .await
                {
                    sources.push(found);
                }
            }
            if let Some(replaces) = fact.pending_supersede_of {
                if let Ok(old) = api.get::<Fact>(&format!("facts/{replaces}"), &[]).await {
                    history.facts.insert(0, old);
                }
            }
            app.open(Detail {
                fact,
                history: history.facts,
                sources,
            });
        }
        Effect::Decide(id, decision) => {
            let _: Value = api
                .send(
                    Method::POST,
                    &format!("quarantine/{id}/{}", decision.action()),
                    &[],
                    None,
                )
                .await?;
            let scopes: Scopes = api.get("scopes", &[]).await?;
            app.set_scopes(scopes.scopes);
            load_tab(app, api).await?;
            app.say(match decision {
                Decision::Approve => "Approved: your agent can recall it now.",
                Decision::Reject => "Rejected: it will not be recalled.",
            });
        }
    }
    Ok(true)
}

async fn load_tab(app: &mut App, api: &Api) -> Result<(), String> {
    let Some(scope) = app.scope() else {
        return Ok(());
    };
    match app.tab() {
        Tab::Facts => {
            let facts: Facts<FactView> = api
                .get(
                    "facts",
                    &[
                        ("scope", &scope),
                        ("state", "current"),
                        ("limit", LIST_LIMIT),
                    ],
                )
                .await?;
            app.facts = facts.facts;
        }
        Tab::Timeline => {
            let facts: Facts<FactView> = api
                .get(
                    "facts",
                    &[("scope", &scope), ("state", "all"), ("limit", LIST_LIMIT)],
                )
                .await?;
            app.timeline = facts.facts;
        }
        Tab::Review => {
            let facts: Facts<Fact> = api
                .get("quarantine", &[("scopes", &scope), ("limit", "200")])
                .await?;
            app.review = facts.facts;
        }
        Tab::Ask => {}
    }
    app.loaded();
    Ok(())
}

/// Runs until `q` or Ctrl+C. The terminal is restored on every exit, including a panic.
pub fn run() -> Result<(), String> {
    let (base, key) = memoryos_hooks::ui_connection()?;
    let api = Api::new(base, key)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the runtime: {e}"))?;
    let mut app = App::new();
    let mut terminal = ratatui::init();
    let result = (|| {
        let mut effect = Effect::LoadScopes;
        loop {
            match runtime.block_on(apply(&mut app, &api, effect)) {
                Ok(true) => {}
                Ok(false) => return Ok(()),
                Err(message) => app.say(message),
            }
            terminal
                .draw(|frame| app.draw(frame))
                .map_err(|e| format!("cannot draw: {e}"))?;
            effect = Effect::None;
            match event::read().map_err(|e| format!("cannot read the keyboard: {e}"))? {
                Event::Key(key) if key.kind == KeyEventKind::Press => effect = app.key(key),
                _ => {}
            }
        }
    })();
    ratatui::restore();
    result
}
