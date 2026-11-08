//! What the terminal view holds, how keys change it, and how it is drawn. No network here: a key
//! returns an [`Effect`], and `mod.rs` performs it and hands the result back.

use chrono::{DateTime, Utc};
use memoryos_core::{Episode, Fact, FactStatus, FactView, ScopeSummary, ScoredFact, Trust};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Tabs, Wrap};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
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
struct Row<'a> {
    fact: &'a Fact,
    /// The subject heading it sits under, on views grouped by subject.
    subject: Option<&'a str>,
    /// Its place in recall's answer, on Ask.
    rank: Option<usize>,
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
        self.rows().get(self.selected).map(|row| row.fact.id)
    }

    fn rows<'a>(&'a self) -> Vec<Row<'a>> {
        let needle = self.filter.to_lowercase();
        let keep = |fact: &Fact, subject: &str| {
            needle.is_empty()
                || fact.statement.to_lowercase().contains(&needle)
                || subject.to_lowercase().contains(&needle)
        };
        let grouped = |views: &'a [FactView]| -> Vec<Row<'a>> {
            views
                .iter()
                .filter(|view| keep(&view.fact, &view.subject_name))
                .map(|view| Row {
                    fact: &view.fact,
                    subject: Some(view.subject_name.as_str()),
                    rank: None,
                })
                .collect()
        };
        match self.tab() {
            Tab::Facts => {
                let mut rows = grouped(&self.facts);
                // Subjects in name order, each subject's newest fact first.
                rows.sort_by(|a, b| {
                    a.subject
                        .cmp(&b.subject)
                        .then(b.fact.valid_from.cmp(&a.fact.valid_from))
                });
                rows
            }
            Tab::Timeline => {
                let mut rows = grouped(&self.timeline);
                rows.sort_by(|a, b| {
                    (a.subject, &a.fact.predicate, a.fact.valid_from).cmp(&(
                        b.subject,
                        &b.fact.predicate,
                        b.fact.valid_from,
                    ))
                });
                rows
            }
            Tab::Review => self
                .review
                .iter()
                .filter(|fact| keep(fact, ""))
                .map(|fact| Row {
                    fact,
                    subject: None,
                    rank: None,
                })
                .collect(),
            Tab::Ask => self
                .answers
                .iter()
                .enumerate()
                .map(|(rank, scored)| Row {
                    fact: &scored.fact,
                    subject: None,
                    rank: Some(rank + 1),
                })
                .collect(),
        }
    }

    pub fn draw(&self, frame: &mut Frame) {
        let [header, body, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        self.draw_header(frame, header);
        let rows = self.rows();
        if body.width >= PREVIEW_MIN_WIDTH && !rows.is_empty() {
            let [list, preview] =
                Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
                    .areas(body);
            self.draw_list(frame, list, &rows);
            draw_preview(frame, preview, &rows[self.selected.min(rows.len() - 1)]);
        } else {
            self.draw_list(frame, body, &rows);
        }
        self.draw_footer(frame, footer);
        if let Some(detail) = &self.detail {
            draw_detail(frame, detail, self.detail_scroll);
        }
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let summary = self.scopes.get(self.scope);
        let count = |pick: fn(&ScopeSummary) -> i64| summary.map(|s| format!(" {}", pick(s)));
        let titles = [
            (Tab::Facts, count(|s| s.current)),
            (Tab::Review, count(|s| s.quarantined)),
            (Tab::Ask, None),
            (Tab::Timeline, count(|s| s.current + s.past)),
        ]
        .into_iter()
        .enumerate()
        .map(|(at, (tab, count))| {
            let selected = tab == self.tab();
            let name = if selected {
                Style::new().fg(ACCENT).bold()
            } else {
                Style::new()
            };
            Line::from(vec![
                Span::styled(format!("{} ", at + 1), Style::new().fg(MUTED)),
                Span::styled(tab.title(), name),
                Span::styled(count.unwrap_or_default(), Style::new().fg(MUTED)),
            ])
        });
        let [line, rule] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);
        let [brand, tabs] =
            Layout::horizontal([Constraint::Length(11), Constraint::Min(0)]).areas(line);
        frame.render_widget(Paragraph::new(" MemoryOS").bold(), brand);
        frame.render_widget(
            Tabs::new(titles)
                .select(self.tab().index())
                .divider(Span::styled(" │ ", Style::new().fg(MUTED)))
                .highlight_style(Style::new().add_modifier(Modifier::UNDERLINED)),
            tabs,
        );
        frame.render_widget(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::new().fg(MUTED)),
            rule,
        );
    }

    /// The scope and where it sits in the list, for the list frame's top right corner.
    fn scope_line(&self) -> Line<'static> {
        match self.scope() {
            Some(scope) => Line::from(vec![
                Span::styled(" scope ", Style::new().fg(MUTED)),
                Span::styled(clean(&scope), Style::new().fg(ACCENT)),
                Span::styled(
                    format!(" {}/{} ", self.scope + 1, self.scopes.len()),
                    Style::new().fg(MUTED),
                ),
            ]),
            None => Line::from(" no scopes ").fg(MUTED),
        }
        .right_aligned()
    }

    fn draw_list(&self, frame: &mut Frame, area: Rect, rows: &[Row]) {
        let mut title = match self.tab() {
            Tab::Facts => " True now ".to_owned(),
            Tab::Review => " Waiting for review ".to_owned(),
            Tab::Ask => match self.question.as_str() {
                "" => " Ask ".to_owned(),
                question => format!(" Ask: {} ", clean(question)),
            },
            Tab::Timeline => " Everything, oldest first ".to_owned(),
        };
        if !self.filter.is_empty() {
            title.push_str(&format!("· filter \"{}\" ", clean(&self.filter)));
        }
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::new().fg(MUTED))
            .title(Span::styled(title, Style::new().bold()))
            .title_top(self.scope_line());
        if rows.is_empty() {
            let empty = match (self.scope(), self.tab()) {
                (None, _) => "Nothing is remembered yet.",
                (_, Tab::Ask) if self.question.is_empty() => {
                    "Press / and type a question to see the facts your agent would get."
                }
                (_, Tab::Review) => "Nothing is waiting for review.",
                _ if !self.filter.is_empty() => "Nothing matches the filter. Esc clears it.",
                _ => "Nothing here.",
            };
            frame.render_widget(
                Paragraph::new(format!("\n  {empty}"))
                    .fg(MUTED)
                    .block(block),
                area,
            );
            return;
        }
        // Inside the borders, less the selection marker.
        let width = area.width.saturating_sub(4) as usize;
        let mut items = Vec::new();
        let mut selected_item = 0;
        let mut group: Option<&str> = None;
        for (at, row) in rows.iter().enumerate() {
            if row.subject.is_some() && row.subject != group {
                group = row.subject;
                if !items.is_empty() {
                    items.push(ListItem::new(""));
                }
                let size = rows.iter().filter(|r| r.subject == group).count();
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(
                        clean(group.unwrap_or_default()),
                        Style::new().fg(ACCENT).bold(),
                    ),
                    Span::styled(format!("  {size}"), Style::new().fg(MUTED)),
                ])));
            }
            if at == self.selected {
                selected_item = items.len();
            }
            items.push(ListItem::new(self.row_line(row, width)));
        }
        let mut state = ListState::default().with_selected(Some(selected_item));
        frame.render_stateful_widget(
            List::new(items)
                .block(block)
                .highlight_style(Style::new().bg(SELECTED).bold())
                .highlight_symbol("▌ ")
                .highlight_spacing(ratatui::widgets::HighlightSpacing::Always),
            area,
            &mut state,
        );
    }

    /// One line per fact: the statement, cut to fit, and a short column on the right.
    fn row_line(&self, row: &Row, width: usize) -> Line<'static> {
        let now = Utc::now();
        let prefix = match (row.rank, row.subject) {
            (Some(rank), _) => format!("{rank:>2}. "),
            (None, Some(_)) => "  ".to_owned(),
            (None, None) => String::new(),
        };
        let (right, right_style) = match self.tab() {
            Tab::Timeline => {
                let (word, color) = status(row.fact, now);
                let until = row
                    .fact
                    .valid_to
                    .or(row.fact.invalidated_at)
                    .map_or("now".to_owned(), date);
                (
                    format!("{} → {until}  {word:<9}", date(row.fact.valid_from)),
                    Style::new().fg(color),
                )
            }
            Tab::Review => (
                format!("{}  {}", trust(row.fact.trust), date(row.fact.recorded_at)),
                Style::new().fg(MUTED),
            ),
            Tab::Facts | Tab::Ask => (date(row.fact.valid_from), Style::new().fg(MUTED)),
        };
        let room = width.saturating_sub(prefix.width() + right.width() + 2);
        let text = fit(&clean(&row.fact.statement), room);
        let gap = width.saturating_sub(prefix.width() + text.width() + right.width());
        Line::from(vec![
            Span::styled(prefix, Style::new().fg(MUTED)),
            Span::raw(text),
            Span::raw(" ".repeat(gap)),
            Span::styled(right, right_style),
        ])
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let line = if let Some(input) = &self.input {
            let label = if self.tab() == Tab::Ask {
                " Question: "
            } else {
                " Filter: "
            };
            Line::from(vec![
                Span::styled(label, Style::new().fg(ACCENT).bold()),
                Span::raw(clean(input)),
                Span::styled("▏", Style::new().fg(ACCENT)),
                Span::styled("   Enter done · Esc cancel", Style::new().fg(MUTED)),
            ])
        } else if let Some(message) = &self.message {
            Line::from(Span::styled(
                format!(" {message}"),
                Style::new().fg(Color::Yellow),
            ))
        } else if self.detail.is_some() {
            hints(&[("↑↓", "scroll"), ("Esc", "close")])
        } else {
            let slash = if self.tab() == Tab::Ask {
                "ask"
            } else {
                "filter"
            };
            let mut keys = vec![
                ("1-4", "view"),
                ("[ ]", "scope"),
                ("↑↓", "move"),
                ("Enter", "open"),
                ("/", slash),
            ];
            if self.tab() == Tab::Review {
                keys.extend([("a", "approve"), ("r", "reject")]);
            }
            keys.extend([("R", "reload"), ("q", "quit")]);
            hints(&keys)
        };
        frame.render_widget(Paragraph::new(line), area);
    }
}

/// Lines terminals below this width give the list alone, without the preview.
const PREVIEW_MIN_WIDTH: u16 = 110;
const ACCENT: Color = Color::Cyan;
const MUTED: Color = Color::DarkGray;
const SELECTED: Color = Color::Indexed(237);

fn hints(keys: &[(&str, &str)]) -> Line<'static> {
    let mut spans = vec![Span::raw(" ")];
    for (at, (key, what)) in keys.iter().enumerate() {
        if at > 0 {
            spans.push(Span::styled("  ", Style::new()));
        }
        spans.push(Span::styled(
            key.to_string(),
            Style::new().fg(ACCENT).bold(),
        ));
        spans.push(Span::styled(format!(" {what}"), Style::new().fg(MUTED)));
    }
    Line::from(spans)
}

/// `text` cut to `room` columns, with an ellipsis when cut. Wide characters count as two.
fn fit(text: &str, room: usize) -> String {
    if text.width() <= room {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used + w + 1 > room {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');
    out
}

/// The fields of one fact, shared by the preview pane and the detail panel.
fn fact_lines(fact: &Fact, subject: Option<&str>, rank: Option<usize>) -> Vec<Line<'static>> {
    let field = |name: &str, value: Span<'static>| {
        Line::from(vec![
            Span::styled(format!("{name:<12}"), Style::new().fg(MUTED)),
            value,
        ])
    };
    let (word, color) = status(fact, Utc::now());
    let mut lines = vec![Line::from(clean(&fact.statement)).bold(), Line::default()];
    if let Some(rank) = rank {
        lines.push(field("Rank", Span::raw(rank.to_string())));
    }
    if let Some(subject) = subject {
        lines.push(field(
            "About",
            Span::styled(clean(subject), Style::new().fg(ACCENT)),
        ));
    }
    lines.extend([
        field("Means", Span::raw(humanize(&fact.predicate))),
        field("Status", Span::styled(word, Style::new().fg(color))),
        field("True from", Span::raw(date(fact.valid_from))),
        field(
            "True until",
            Span::raw(fact.valid_to.map_or("not set".to_owned(), date)),
        ),
        field("Learned", Span::raw(date(fact.recorded_at))),
        field("Trust", Span::raw(trust(fact.trust))),
        field("Scope", Span::raw(clean(&fact.scope.to_string()))),
    ]);
    lines
}

fn draw_preview(frame: &mut Frame, area: Rect, row: &Row) {
    let mut lines = fact_lines(row.fact, row.subject, row.rank);
    lines.push(Line::default());
    lines.push(Line::from("Enter shows history and sources").fg(MUTED));
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().fg(MUTED))
                .padding(ratatui::widgets::Padding::horizontal(1))
                .title(" Selected "),
        ),
        area,
    );
}

fn draw_detail(frame: &mut Frame, detail: &Detail, scroll: u16) {
    let area = centered(frame.area(), 80, 80);
    let mut lines = fact_lines(&detail.fact, None, None);
    lines.push(field_line("Id", detail.fact.id.to_string()));
    if detail.history.len() > 1 {
        lines.push(Line::default());
        lines.push(Line::from("History").fg(ACCENT).bold());
        for item in &detail.history {
            lines.push(Line::from(vec![
                Span::styled(
                    format!(
                        "  {} → {}  ",
                        date(item.valid_from),
                        item.valid_to.map_or("now".to_owned(), date)
                    ),
                    Style::new().fg(MUTED),
                ),
                Span::raw(clean(&item.statement)),
            ]));
        }
    }
    if !detail.sources.is_empty() {
        lines.push(Line::default());
        lines.push(Line::from("Where it came from").fg(ACCENT).bold());
        for episode in &detail.sources {
            lines.push(
                Line::from(format!(
                    "  {}, {}",
                    clean(&episode.source),
                    date(episode.occurred_at)
                ))
                .fg(MUTED),
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
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::new().fg(ACCENT))
                    .padding(ratatui::widgets::Padding::horizontal(1))
                    .title(" Fact "),
            ),
        area,
    );
}

fn field_line(name: &str, value: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{name:<12}"), Style::new().fg(MUTED)),
        Span::raw(value),
    ])
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

/// The same states the web page shows, in the same order of precedence, with a color for each.
fn status(fact: &Fact, now: DateTime<Utc>) -> (&'static str, Color) {
    match fact.status {
        FactStatus::Quarantined => return ("waiting", Color::Yellow),
        FactStatus::Rejected => return ("rejected", Color::Red),
        FactStatus::Active => {}
    }
    if fact.invalidated_at.is_some_and(|at| at <= now) {
        ("retracted", Color::Red)
    } else if fact.valid_from > now {
        ("later", MUTED)
    } else if fact.valid_to.is_some_and(|at| at <= now) {
        if fact.superseded_by.is_some() {
            ("replaced", MUTED)
        } else {
            ("ended", MUTED)
        }
    } else if fact.expired_at(now) {
        ("expired", MUTED)
    } else {
        ("true now", Color::Green)
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

    fn screen(app: &App, width: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, 20)).expect("terminal");
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
        let second = app.rows()[1].fact.id;
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
    fn the_screen_shows_views_scope_and_facts_under_their_subject() {
        let app = with_facts();
        let text = screen(&app, 100);
        assert!(text.contains("1 Facts 1"), "{text}");
        assert!(text.contains("scope agent:coding-agent 1/2"), "{text}");
        let lines: Vec<&str> = text.lines().collect();
        let heading = lines
            .iter()
            .position(|line| line.contains("Ada  1"))
            .expect("subject heading");
        assert!(
            lines[heading + 1].contains("Ada lives in Bandung"),
            "{text}"
        );
        assert!(lines[heading + 1].contains("2026-11-01"), "{text}");
        assert!(text.contains("CI  1"), "{text}");
    }

    #[test]
    fn a_wide_terminal_shows_the_selected_fact_in_full_beside_the_list() {
        let mut app = with_facts();
        let long = format!("Ada keeps notes {}", "about many things ".repeat(12));
        app.facts = vec![view(&long, "Ada")];
        let text = screen(&app, 140);
        assert!(text.contains("Selected"), "{text}");
        assert!(text.contains('…'), "the list line is cut: {text}");
        assert!(text.contains("About       Ada"), "{text}");
        assert!(
            !screen(&app, 90).contains("Selected"),
            "no room for it at 90 columns"
        );
    }

    #[test]
    fn fit_cuts_by_display_width_with_an_ellipsis() {
        assert_eq!(fit("short", 10), "short");
        assert_eq!(fit("abcdefghij", 5), "abcd…");
        // Each of these characters takes two columns.
        assert_eq!(fit("兴来大人兴来大人", 7), "兴来大…");
        assert!(fit("兴来大人兴来大人", 7).width() <= 7);
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
        let text = screen(&app, 100);
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
        assert!(!screen(&app, 100).contains('\u{1b}'));
    }
}
