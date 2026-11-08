//! What the terminal view holds, how keys change it, and how it is drawn. No network here: a key
//! returns an [`Effect`], and `mod.rs` performs it and hands the result back.

use chrono::{DateTime, Utc};
use memoryos_core::{Episode, Fact, FactStatus, FactView, ScopeSummary, ScoredFact, Trust};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Tabs, Wrap};
use ratatui::Frame;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Facts,
    Review,
    Ask,
    Timeline,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Facts, Tab::Review, Tab::Ask, Tab::Timeline];

    fn title(self) -> &'static str {
        match self {
            Tab::Facts => "Facts",
            Tab::Review => "Review",
            Tab::Ask => "Ask",
            Tab::Timeline => "Timeline",
        }
    }

    fn index(self) -> usize {
        Tab::ALL.iter().position(|tab| *tab == self).unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Approve,
    Reject,
}

impl Decision {
    pub fn action(self) -> &'static str {
        match self {
            Decision::Approve => "approve",
            Decision::Reject => "reject",
        }
    }
}

/// Work a key asks for that needs the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    None,
    Quit,
    /// The scope list, then the current tab.
    LoadScopes,
    LoadTab,
    Recall(String),
    Open(Uuid),
    Decide(Uuid, Decision),
}

/// One fact opened in the detail panel.
#[derive(Debug, Clone)]
pub struct Detail {
    pub fact: Fact,
    pub history: Vec<Fact>,
    pub sources: Vec<Episode>,
}

/// A fact as one tab shows it.
struct Row {
    id: Uuid,
    title: String,
    meta: String,
}

#[derive(Default)]
pub struct App {
    pub scopes: Vec<ScopeSummary>,
    scope: usize,
    tab: TabState,
    pub facts: Vec<FactView>,
    pub timeline: Vec<FactView>,
    pub review: Vec<Fact>,
    pub answers: Vec<ScoredFact>,
    question: String,
    selected: usize,
    /// Text being typed after `/`: a question on Ask, a filter elsewhere.
    input: Option<String>,
    filter: String,
    pub detail: Option<Detail>,
    detail_scroll: u16,
    message: Option<String>,
}

/// `Tab` with a default, so `App` can derive one.
#[derive(Clone, Copy)]
struct TabState(Tab);

impl Default for TabState {
    fn default() -> Self {
        Self(Tab::Facts)
    }
}

impl App {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tab(&self) -> Tab {
        self.tab.0
    }

    pub fn scope(&self) -> Option<String> {
        self.scopes
            .get(self.scope)
            .map(|item| item.scope.to_string())
    }

    /// Keeps the chosen scope when it still exists.
    pub fn set_scopes(&mut self, scopes: Vec<ScopeSummary>) {
        let chosen = self.scope();
        self.scopes = scopes;
        self.scope = chosen
            .and_then(|name| {
                self.scopes
                    .iter()
                    .position(|item| item.scope.to_string() == name)
            })
            .unwrap_or(0);
    }

    pub fn set_answers(&mut self, question: String, answers: Vec<ScoredFact>) {
        self.question = question;
        self.answers = answers;
        self.selected = 0;
    }

    /// Called after a tab's data is replaced, so the selection stays on a row.
    pub fn loaded(&mut self) {
        let count = self.rows().len();
        self.selected = self.selected.min(count.saturating_sub(1));
    }

    pub fn open(&mut self, detail: Detail) {
        self.detail = Some(detail);
        self.detail_scroll = 0;
    }

    pub fn say(&mut self, message: impl Into<String>) {
        self.message = Some(message.into());
    }

    pub fn key(&mut self, key: KeyEvent) -> Effect {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Effect::Quit;
        }
        self.message = None;
        if let Some(input) = self.input.as_mut() {
            match key.code {
                KeyCode::Esc => self.input = None,
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Char(c) => input.push(c),
                KeyCode::Enter => {
                    let text = self.input.take().unwrap_or_default();
                    self.selected = 0;
                    if self.tab() == Tab::Ask {
                        let question = text.trim().to_owned();
                        return if question.is_empty() || self.scope().is_none() {
                            Effect::None
                        } else {
                            Effect::Recall(question)
                        };
                    }
                    self.filter = text;
                }
                _ => {}
            }
            return Effect::None;
        }
        if self.detail.is_some() {
            match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => self.detail = None,
                KeyCode::Down | KeyCode::Char('j') => {
                    self.detail_scroll = self.detail_scroll.saturating_add(1)
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.detail_scroll = self.detail_scroll.saturating_sub(1)
                }
                _ => {}
            }
            return Effect::None;
        }
        match key.code {
            KeyCode::Char('q') => Effect::Quit,
            KeyCode::Esc => {
                self.filter.clear();
                Effect::None
            }
            KeyCode::Char(c @ '1'..='4') => self.switch(Tab::ALL[c as usize - '1' as usize]),
            KeyCode::Tab => self.switch(Tab::ALL[(self.tab().index() + 1) % Tab::ALL.len()]),
            KeyCode::Char(']') => self.move_scope(1),
            KeyCode::Char('[') => self.move_scope(self.scopes.len().saturating_sub(1)),
            KeyCode::Char('R') => Effect::LoadScopes,
            KeyCode::Char('/') => {
                self.input = Some(String::new());
                Effect::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let last = self.rows().len().saturating_sub(1);
                self.selected = (self.selected + 1).min(last);
                Effect::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = self.selected.saturating_sub(1);
                Effect::None
            }
            KeyCode::Enter => self.chosen().map_or(Effect::None, Effect::Open),
            KeyCode::Char(c @ ('a' | 'r')) if self.tab() == Tab::Review => {
                let decision = if c == 'a' {
                    Decision::Approve
                } else {
                    Decision::Reject
                };
                self.chosen()
                    .map_or(Effect::None, |id| Effect::Decide(id, decision))
            }
            _ => Effect::None,
        }
    }

    fn switch(&mut self, tab: Tab) -> Effect {
        self.tab = TabState(tab);
        self.selected = 0;
        self.filter.clear();
        if tab == Tab::Ask {
            Effect::None
        } else {
            Effect::LoadTab
        }
    }

    fn move_scope(&mut self, step: usize) -> Effect {
        if self.scopes.is_empty() {
            return Effect::None;
        }
        self.scope = (self.scope + step) % self.scopes.len();
        self.selected = 0;
        self.answers.clear();
        self.question.clear();
        Effect::LoadTab
    }

    fn chosen(&self) -> Option<Uuid> {
        self.rows().get(self.selected).map(|row| row.id)
    }

    fn rows(&self) -> Vec<Row> {
        let now = Utc::now();
        let needle = self.filter.to_lowercase();
        let keep = |fact: &Fact, subject: &str| {
            needle.is_empty()
                || fact.statement.to_lowercase().contains(&needle)
                || subject.to_lowercase().contains(&needle)
        };
        match self.tab() {
            Tab::Facts => {
                let mut facts: Vec<&FactView> = self
                    .facts
                    .iter()
                    .filter(|view| keep(&view.fact, &view.subject_name))
                    .collect();
                facts.sort_by(|a, b| {
                    a.subject_name
                        .cmp(&b.subject_name)
                        .then(b.fact.valid_from.cmp(&a.fact.valid_from))
                });
                facts
                    .into_iter()
                    .map(|view| Row {
                        id: view.fact.id,
                        title: clean(&view.fact.statement),
                        meta: format!(
                            "{} · {} · since {} · {}",
                            clean(&view.subject_name),
                            humanize(&view.fact.predicate),
                            date(view.fact.valid_from),
                            trust(view.fact.trust)
                        ),
                    })
                    .collect()
            }
            Tab::Timeline => {
                let mut facts: Vec<&FactView> = self
                    .timeline
                    .iter()
                    .filter(|view| keep(&view.fact, &view.subject_name))
                    .collect();
                facts.sort_by(|a, b| {
                    (&a.subject_name, &a.fact.predicate, a.fact.valid_from).cmp(&(
                        &b.subject_name,
                        &b.fact.predicate,
                        b.fact.valid_from,
                    ))
                });
                facts
                    .into_iter()
                    .map(|view| Row {
                        id: view.fact.id,
                        title: clean(&view.fact.statement),
                        meta: format!(
                            "{} · {} · {} to {} · {}",
                            clean(&view.subject_name),
                            humanize(&view.fact.predicate),
                            date(view.fact.valid_from),
                            view.fact
                                .valid_to
                                .or(view.fact.invalidated_at)
                                .map_or("now".to_owned(), date),
                            status(&view.fact, now)
                        ),
                    })
                    .collect()
            }
            Tab::Review => self
                .review
                .iter()
                .filter(|fact| keep(fact, ""))
                .map(|fact| Row {
                    id: fact.id,
                    title: clean(&fact.statement),
                    meta: format!(
                        "{} · {} · added {}{}",
                        humanize(&fact.predicate),
                        trust(fact.trust),
                        date(fact.recorded_at),
                        if fact.pending_supersede_of.is_some() {
                            " · would replace a fact, Enter to see it"
                        } else {
                            ""
                        }
                    ),
                })
                .collect(),
            Tab::Ask => self
                .answers
                .iter()
                .enumerate()
                .map(|(rank, scored)| Row {
                    id: scored.fact.id,
                    title: format!("{}. {}", rank + 1, clean(&scored.fact.statement)),
                    meta: format!(
                        "{} · since {} · {}",
                        humanize(&scored.fact.predicate),
                        date(scored.fact.valid_from),
                        trust(scored.fact.trust)
                    ),
                })
                .collect(),
        }
    }

    pub fn draw(&self, frame: &mut Frame) {
        let [header, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        self.draw_header(frame, header);
        self.draw_body(frame, body);
        self.draw_footer(frame, footer);
        if let Some(detail) = &self.detail {
            draw_detail(frame, detail, self.detail_scroll);
        }
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let scope = match self.scope() {
            Some(scope) => format!(
                "[ {} ({}/{}) ] ",
                clean(&scope),
                self.scope + 1,
                self.scopes.len()
            ),
            None => "no scopes ".to_owned(),
        };
        let [tabs, scope_area] = Layout::horizontal([
            Constraint::Min(0),
            Constraint::Length(scope.chars().count() as u16),
        ])
        .areas(area);
        let titles = Tab::ALL
            .iter()
            .enumerate()
            .map(|(at, tab)| format!("{} {}", at + 1, tab.title()));
        frame.render_widget(
            Tabs::new(titles)
                .select(self.tab().index())
                .highlight_style(Style::new().add_modifier(Modifier::BOLD | Modifier::REVERSED)),
            tabs,
        );
        frame.render_widget(Paragraph::new(scope).dim(), scope_area);
    }

    fn draw_body(&self, frame: &mut Frame, area: Rect) {
        let rows = self.rows();
        let mut title = match self.tab() {
            Tab::Facts => format!(" True now · {} ", rows.len()),
            Tab::Review => format!(" Waiting for review · {} ", rows.len()),
            Tab::Ask => match self.question.as_str() {
                "" => " Ask: press / and type a question ".to_owned(),
                question => format!(" Ask: {} ", clean(question)),
            },
            Tab::Timeline => format!(" Everything, oldest first per subject · {} ", rows.len()),
        };
        if !self.filter.is_empty() {
            title.push_str(&format!("· filter \"{}\" ", clean(&self.filter)));
        }
        let block = Block::default().borders(Borders::ALL).title(title);
        if rows.is_empty() {
            let empty = match (self.scope(), self.tab()) {
                (None, _) => "Nothing is remembered yet.",
                (_, Tab::Ask) if self.question.is_empty() => "The facts your agent would get.",
                (_, Tab::Review) => "Nothing is waiting for review.",
                _ => "Nothing here.",
            };
            frame.render_widget(Paragraph::new(empty).dim().block(block), area);
            return;
        }
        let items: Vec<ListItem> = rows
            .into_iter()
            .map(|row| {
                ListItem::new(Text::from(vec![
                    Line::from(row.title),
                    Line::from(Span::styled(
                        format!("   {}", row.meta),
                        Style::new().fg(Color::DarkGray),
                    )),
                ]))
            })
            .collect();
        let mut state = ListState::default().with_selected(Some(self.selected));
        frame.render_stateful_widget(
            List::new(items)
                .block(block)
                .highlight_style(Style::new().add_modifier(Modifier::BOLD))
                .highlight_symbol("> "),
            area,
            &mut state,
        );
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let line = if let Some(input) = &self.input {
            let label = if self.tab() == Tab::Ask {
                "Question"
            } else {
                "Filter"
            };
            Line::from(format!("{label}: {}_", clean(input)))
        } else if let Some(message) = &self.message {
            Line::from(Span::styled(
                message.clone(),
                Style::new().fg(Color::Yellow),
            ))
        } else if self.detail.is_some() {
            Line::from("↑↓ scroll · Esc close").dim()
        } else {
            let review = if self.tab() == Tab::Review {
                " · a approve · r reject"
            } else {
                ""
            };
            Line::from(format!(
                "1-4 view · [ ] scope · ↑↓ move · Enter open · / {}{review} · R reload · q quit",
                if self.tab() == Tab::Ask {
                    "ask"
                } else {
                    "filter"
                }
            ))
            .dim()
        };
        frame.render_widget(Paragraph::new(line), area);
    }
}

fn draw_detail(frame: &mut Frame, detail: &Detail, scroll: u16) {
    let area = centered(frame.area(), 80, 80);
    let fact = &detail.fact;
    let now = Utc::now();
    let field = |name: &str, value: String| {
        Line::from(vec![
            Span::styled(format!("{name:<12}"), Style::new().fg(Color::DarkGray)),
            Span::raw(value),
        ])
    };
    let mut lines = vec![
        Line::from(clean(&fact.statement)).bold(),
        Line::default(),
        field("Status", status(fact, now).to_owned()),
        field("Means", humanize(&fact.predicate)),
        field("True from", date(fact.valid_from)),
        field(
            "True until",
            fact.valid_to.map_or("not set".to_owned(), date),
        ),
        field("Learned", date(fact.recorded_at)),
        field("Trust", trust(fact.trust).to_owned()),
        field("Scope", clean(&fact.scope.to_string())),
        field("Id", fact.id.to_string()),
    ];
    if detail.history.len() > 1 {
        lines.push(Line::default());
        lines.push(Line::from("History").bold());
        for item in &detail.history {
            lines.push(Line::from(format!(
                "  {} to {}  {}",
                date(item.valid_from),
                item.valid_to.map_or("now".to_owned(), date),
                clean(&item.statement)
            )));
        }
    }
    if !detail.sources.is_empty() {
        lines.push(Line::default());
        lines.push(Line::from("Where it came from").bold());
        for episode in &detail.sources {
            lines.push(
                Line::from(format!(
                    "  {}, {}",
                    clean(&episode.source),
                    date(episode.occurred_at)
                ))
                .dim(),
            );
            for text in episode.content.lines() {
                lines.push(Line::from(format!("  {}", clean(text))));
            }
        }
    }
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
            .block(Block::default().borders(Borders::ALL).title(" Fact ")),
        area,
    );
}

fn centered(area: Rect, width_percent: u16, height_percent: u16) -> Rect {
    let width = area.width * width_percent / 100;
    let height = area.height * height_percent / 100;
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

/// Server text with control characters replaced. Fact text is untrusted: agents write it and it can
/// quote hostile pages, and an escape sequence written to the terminal could move the cursor,
/// rewrite earlier lines or change the window title.
pub fn clean(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

fn date(at: DateTime<Utc>) -> String {
    at.format("%Y-%m-%d").to_string()
}

fn humanize(predicate: &str) -> String {
    clean(&predicate.replace('_', " "))
}

fn trust(trust: Trust) -> &'static str {
    match trust {
        Trust::System => "system",
        Trust::User => "from the user",
        Trust::Agent => "from an agent",
        Trust::External => "outside source",
    }
}

/// The same states the web page shows, in the same order of precedence.
fn status(fact: &Fact, now: DateTime<Utc>) -> &'static str {
    match fact.status {
        FactStatus::Quarantined => return "waiting for review",
        FactStatus::Rejected => return "rejected",
        FactStatus::Active => {}
    }
    if fact.invalidated_at.is_some_and(|at| at <= now) {
        "retracted"
    } else if fact.valid_from > now {
        "starts later"
    } else if fact.valid_to.is_some_and(|at| at <= now) {
        if fact.superseded_by.is_some() {
            "replaced"
        } else {
            "ended"
        }
    } else if fact.expired_at(now) {
        "expired"
    } else {
        "true now"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memoryos_core::Scope;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn fact(statement: &str) -> Fact {
        let at = DateTime::parse_from_rfc3339("2026-11-01T00:00:00Z")
            .expect("date")
            .with_timezone(&Utc);
        Fact {
            id: Uuid::now_v7(),
            scope: "agent:coding-agent".parse::<Scope>().expect("scope"),
            subject_id: Uuid::now_v7(),
            predicate: "lives_in".into(),
            object_id: None,
            statement: statement.into(),
            trust: Trust::Agent,
            status: FactStatus::Active,
            valid_from: at,
            valid_to: None,
            expires_at: None,
            recorded_at: at,
            invalidated_at: None,
            superseded_by: None,
            pending_supersede_of: None,
            source_episode_ids: vec![],
        }
    }

    fn view(statement: &str, subject: &str) -> FactView {
        FactView {
            fact: fact(statement),
            subject_name: subject.into(),
            subject_kind: "person".into(),
            object_name: None,
            object_kind: None,
        }
    }

    fn summary(scope: &str) -> ScopeSummary {
        ScopeSummary {
            scope: scope.parse::<Scope>().expect("scope"),
            current: 1,
            past: 0,
            quarantined: 0,
            entities: 1,
            blocks: 0,
            last_recorded_at: None,
        }
    }

    fn press(app: &mut App, code: KeyCode) -> Effect {
        app.key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn typed(app: &mut App, text: &str) {
        for c in text.chars() {
            press(app, KeyCode::Char(c));
        }
    }

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(100, 20)).expect("terminal");
        terminal.draw(|frame| app.draw(frame)).expect("draw");
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn with_facts() -> App {
        let mut app = App::new();
        app.set_scopes(vec![summary("agent:coding-agent"), summary("user:ada")]);
        app.facts = vec![
            view("Ada lives in Bandung", "Ada"),
            view("The CI runs on GitHub Actions", "CI"),
        ];
        app.loaded();
        app
    }

    #[test]
    fn number_keys_switch_views_and_ask_loads_nothing_until_asked() {
        let mut app = with_facts();
        assert_eq!(press(&mut app, KeyCode::Char('2')), Effect::LoadTab);
        assert_eq!(app.tab(), Tab::Review);
        assert_eq!(press(&mut app, KeyCode::Char('3')), Effect::None);
        assert_eq!(app.tab(), Tab::Ask);
        assert_eq!(press(&mut app, KeyCode::Tab), Effect::LoadTab);
        assert_eq!(app.tab(), Tab::Timeline);
    }

    #[test]
    fn a_question_is_typed_after_slash_and_sent_on_enter() {
        let mut app = with_facts();
        press(&mut app, KeyCode::Char('3'));
        press(&mut app, KeyCode::Char('/'));
        typed(&mut app, "where does Ada live? q");
        // `q` was text, not quit, because a question was being typed.
        assert_eq!(
            press(&mut app, KeyCode::Enter),
            Effect::Recall("where does Ada live? q".into())
        );
    }

    #[test]
    fn a_filter_keeps_matching_facts_and_esc_clears_it() {
        let mut app = with_facts();
        press(&mut app, KeyCode::Char('/'));
        typed(&mut app, "bandung");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.rows().len(), 1);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.rows().len(), 2);
    }

    #[test]
    fn enter_opens_the_selected_fact_and_review_keys_decide_it() {
        let mut app = with_facts();
        press(&mut app, KeyCode::Down);
        let second = app.rows()[1].id;
        assert_eq!(press(&mut app, KeyCode::Enter), Effect::Open(second));

        press(&mut app, KeyCode::Char('2'));
        app.review = vec![fact("Ada wants dark mode")];
        let id = app.review[0].id;
        assert_eq!(
            press(&mut app, KeyCode::Char('a')),
            Effect::Decide(id, Decision::Approve)
        );
        assert_eq!(
            press(&mut app, KeyCode::Char('r')),
            Effect::Decide(id, Decision::Reject)
        );
    }

    #[test]
    fn approve_and_reject_keys_do_nothing_outside_review() {
        let mut app = with_facts();
        assert_eq!(press(&mut app, KeyCode::Char('a')), Effect::None);
        assert_eq!(press(&mut app, KeyCode::Char('r')), Effect::None);
    }

    #[test]
    fn brackets_cycle_scopes_and_reload() {
        let mut app = with_facts();
        assert_eq!(press(&mut app, KeyCode::Char(']')), Effect::LoadTab);
        assert_eq!(app.scope().as_deref(), Some("user:ada"));
        press(&mut app, KeyCode::Char(']'));
        assert_eq!(app.scope().as_deref(), Some("agent:coding-agent"));
        press(&mut app, KeyCode::Char('['));
        assert_eq!(app.scope().as_deref(), Some("user:ada"));
        // A refreshed list keeps the scope that was chosen.
        app.set_scopes(vec![summary("user:ada"), summary("agent:coding-agent")]);
        assert_eq!(app.scope().as_deref(), Some("user:ada"));
    }

    #[test]
    fn the_screen_shows_views_scope_and_facts() {
        let app = with_facts();
        let text = screen(&app);
        assert!(text.contains("1 Facts"), "{text}");
        assert!(text.contains("agent:coding-agent (1/2)"), "{text}");
        assert!(text.contains("Ada lives in Bandung"), "{text}");
        assert!(text.contains("since 2026-11-01"), "{text}");
    }

    #[test]
    fn the_detail_panel_shows_history_and_source() {
        let mut app = with_facts();
        let old = fact("Ada lives in Jakarta");
        let current = fact("Ada lives in Bandung");
        app.open(Detail {
            fact: current.clone(),
            history: vec![old, current],
            sources: vec![],
        });
        let text = screen(&app);
        assert!(text.contains("History"), "{text}");
        assert!(text.contains("Ada lives in Jakarta"), "{text}");
        assert_eq!(press(&mut app, KeyCode::Esc), Effect::None);
        assert!(app.detail.is_none());
    }

    #[test]
    fn control_characters_in_fact_text_never_reach_the_terminal() {
        assert_eq!(clean("a\u{1b}]0;owned\u{7}b\nc"), "a ]0;owned b c");
        let mut app = with_facts();
        app.facts = vec![view("evil\u{1b}[2J text", "x")];
        assert!(!screen(&app).contains('\u{1b}'));
    }
}
