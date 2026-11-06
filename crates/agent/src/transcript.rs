//! Pure logic over stored messages: decoding, repair of interrupted turns, and where compaction
//! may cut. Providers reject a `tool_use` not directly followed by its `tool_result`, so
//! everything here keeps those pairs intact.

use std::collections::{HashMap, HashSet, VecDeque};

use memoryos_core::{ContentBlock, Message, Role, SessionMessage};

pub(crate) const INTERRUPTED_RESULT: &str = "This tool call was interrupted before a result was \
recorded. It may or may not have taken effect. Check the current state before retrying.";

/// Keeps a window that starts mid-turn valid: Anthropic requires the first message to be `user`.
const WINDOW_START: &str = "[Earlier conversation is in the summary.]";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StoredMessage {
    pub(crate) seq: i64,
    pub(crate) message: Message,
}

pub(crate) fn role_name(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// A row that cannot be decoded is skipped, not fatal: it would otherwise fail every later turn
/// of the session. The log carries only its sequence number.
pub(crate) fn decode(rows: Vec<SessionMessage>) -> Vec<StoredMessage> {
    rows.into_iter()
        .filter_map(|row| {
            let role = match row.role.as_str() {
                "user" => Some(Role::User),
                "assistant" => Some(Role::Assistant),
                _ => None,
            };
            let content = serde_json::from_value::<Vec<ContentBlock>>(row.content).ok();
            match (role, content) {
                (Some(role), Some(content)) => Some(StoredMessage {
                    seq: row.seq,
                    message: Message { role, content },
                }),
                _ => {
                    tracing::warn!(seq = row.seq, "skipping undecodable session message");
                    None
                }
            }
        })
        .collect()
}

/// Error results for every stored `tool_use` that has none. The caller appends them.
pub(crate) fn missing_results(stored: &[StoredMessage]) -> Vec<ContentBlock> {
    pair_tool_blocks(stored).1
}

/// The provider-ready sequence: each `tool_use` directly followed by its result, orphan results
/// dropped, and consecutive messages of one role merged.
pub(crate) fn build_messages(stored: &[StoredMessage]) -> Vec<Message> {
    pair_tool_blocks(stored).0
}

/// A repair row lands after whatever was already stored, and two server instances can interleave,
/// so a result is looked up by id rather than expected in the next row.
fn pair_tool_blocks(stored: &[StoredMessage]) -> (Vec<Message>, Vec<ContentBlock>) {
    let mut results = ResultQueues::new(stored);
    let mut out: Vec<Message> = Vec::with_capacity(stored.len());
    let mut missing = Vec::new();

    for (index, row) in stored.iter().enumerate() {
        match row.message.role {
            Role::Assistant => {
                push_merged(&mut out, Role::Assistant, row.message.content.clone());
                let mut paired = Vec::new();
                for block in &row.message.content {
                    let ContentBlock::ToolUse { id, .. } = block else {
                        continue;
                    };
                    match results.take(id, index) {
                        Some(result) => paired.push(result),
                        None => {
                            let repair = ContentBlock::ToolResult {
                                tool_use_id: id.clone(),
                                content: INTERRUPTED_RESULT.to_owned(),
                                is_error: true,
                            };
                            paired.push(repair.clone());
                            missing.push(repair);
                        }
                    }
                }
                push_merged(&mut out, Role::User, paired);
            }
            Role::User => {
                let text: Vec<ContentBlock> = row
                    .message
                    .content
                    .iter()
                    .filter(|block| !matches!(block, ContentBlock::ToolResult { .. }))
                    .cloned()
                    .collect();
                push_merged(&mut out, Role::User, text);
            }
        }
    }

    if out.first().is_some_and(|m| m.role == Role::Assistant) {
        out.insert(
            0,
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: WINDOW_START.to_owned(),
                }],
            },
        );
    }
    (out, missing)
}

/// Results by `tool_use_id`, each with the index of its row.
struct ResultQueues(HashMap<String, VecDeque<(usize, ContentBlock)>>);

impl ResultQueues {
    fn new(stored: &[StoredMessage]) -> Self {
        let mut queues: HashMap<String, VecDeque<(usize, ContentBlock)>> = HashMap::new();
        for (index, row) in stored.iter().enumerate() {
            for block in &row.message.content {
                if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                    queues
                        .entry(tool_use_id.clone())
                        .or_default()
                        .push_back((index, block.clone()));
                }
            }
        }
        Self(queues)
    }

    /// The first result stored after the call at `call_index`. Earlier ones are orphans.
    fn take(&mut self, id: &str, call_index: usize) -> Option<ContentBlock> {
        let queue = self.0.get_mut(id)?;
        while let Some((index, block)) = queue.pop_front() {
            if index > call_index {
                return Some(block);
            }
        }
        None
    }
}

/// Appends `content`, or extends the last message when it has the same role. Empty text and empty
/// messages are dropped: Anthropic rejects both.
pub(crate) fn push_merged(out: &mut Vec<Message>, role: Role, content: Vec<ContentBlock>) {
    let content: Vec<ContentBlock> = content
        .into_iter()
        .filter(|block| !matches!(block, ContentBlock::Text { text } if text.is_empty()))
        .collect();
    if content.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(last) if last.role == role => last.content.extend(content),
        _ => out.push(Message { role, content }),
    }
}

/// Size of a message as stored.
pub(crate) fn stored_chars(message: &Message) -> usize {
    serde_json::to_string(&message.content)
        .map(|json| json.chars().count())
        .unwrap_or(0)
}

/// Index of the first message to keep, or `None` if nothing can be compacted. The cut lands only
/// where a user turn starts and no tool pair crosses it. It keeps at most `keep_chars` when such a
/// point exists, and otherwise the shortest tail available.
pub(crate) fn compaction_split(stored: &[StoredMessage], keep_chars: usize) -> Option<usize> {
    let mut wanted = stored.len();
    let mut kept = 0;
    for (index, row) in stored.iter().enumerate().rev() {
        kept += stored_chars(&row.message);
        if kept > keep_chars {
            break;
        }
        wanted = index;
    }

    let cuts = turn_starts(stored);
    cuts.iter()
        .copied()
        .find(|cut| *cut >= wanted)
        .or_else(|| cuts.last().copied())
}

/// Indexes above 0 where a cut leaves every tool pair whole and the tail starts with user text.
fn turn_starts(stored: &[StoredMessage]) -> Vec<usize> {
    let mut open: HashSet<&str> = HashSet::new();
    let mut cuts = Vec::new();
    for (index, row) in stored.iter().enumerate() {
        let starts_turn = row.message.role == Role::User
            && row
                .message
                .content
                .iter()
                .all(|block| matches!(block, ContentBlock::Text { .. }));
        if index > 0 && open.is_empty() && starts_turn {
            cuts.push(index);
        }
        for block in &row.message.content {
            match block {
                ContentBlock::ToolUse { id, .. } => {
                    open.insert(id);
                }
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    open.remove(tool_use_id.as_str());
                }
                ContentBlock::Text { .. } => {}
            }
        }
    }
    cuts
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;
    use serde_json::json;

    use super::*;

    fn text(t: &str) -> ContentBlock {
        ContentBlock::Text { text: t.into() }
    }
    fn call(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.into(),
            name: "memory_recall".into(),
            input: json!({"query": "q"}),
        }
    }
    fn result(id: &str, content: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: content.into(),
            is_error: false,
        }
    }
    fn user(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::User,
            content,
        }
    }
    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
        }
    }
    fn stored(messages: Vec<Message>) -> Vec<StoredMessage> {
        messages
            .into_iter()
            .enumerate()
            .map(|(i, message)| StoredMessage {
                seq: i as i64 + 1,
                message,
            })
            .collect()
    }

    fn assert_valid_for_provider(messages: &[Message]) {
        assert_eq!(messages.first().map(|m| m.role), Some(Role::User));
        for (i, message) in messages.iter().enumerate() {
            if i > 0 {
                assert_ne!(messages[i - 1].role, message.role, "roles must alternate");
            }
            let calls: Vec<&str> = message
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                    _ => None,
                })
                .collect();
            if calls.is_empty() {
                continue;
            }
            let next = &messages[i + 1];
            let answers: Vec<&str> = next
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(calls, answers, "results must directly follow their calls");
            let first_text = next
                .content
                .iter()
                .position(|b| matches!(b, ContentBlock::Text { .. }));
            if let Some(first_text) = first_text {
                assert!(first_text >= answers.len(), "results must precede text");
            }
        }
    }

    #[test]
    fn decode_skips_bad_rows_and_keeps_order() {
        let row = |seq, role: &str, content| SessionMessage {
            seq,
            role: role.into(),
            content,
            created_at: DateTime::UNIX_EPOCH,
        };
        let decoded = decode(vec![
            row(1, "user", json!([{"type": "text", "text": "hi"}])),
            row(2, "user", json!("not an array")),
            row(3, "system", json!([])),
            row(4, "assistant", json!([{"type": "text", "text": "yo"}])),
        ]);
        let seqs: Vec<i64> = decoded.iter().map(|m| m.seq).collect();
        assert_eq!(seqs, vec![1, 4]);
    }

    #[test]
    fn clean_history_needs_no_repair() {
        let rows = stored(vec![
            user(vec![text("q")]),
            assistant(vec![call("a")]),
            user(vec![result("a", "r")]),
            assistant(vec![text("done")]),
        ]);
        assert!(missing_results(&rows).is_empty());
        assert_valid_for_provider(&build_messages(&rows));
    }

    #[test]
    fn dangling_call_gets_an_error_result() {
        let rows = stored(vec![
            user(vec![text("q")]),
            assistant(vec![call("a"), call("b")]),
        ]);
        let missing = missing_results(&rows);
        assert_eq!(missing.len(), 2);
        for (block, id) in missing.iter().zip(["a", "b"]) {
            let ContentBlock::ToolResult {
                tool_use_id,
                is_error,
                content,
            } = block
            else {
                panic!("expected a result, got {block:?}");
            };
            assert_eq!(tool_use_id, id);
            assert!(*is_error);
            assert_eq!(content, INTERRUPTED_RESULT);
        }
        assert_valid_for_provider(&build_messages(&rows));
    }

    #[test]
    fn only_the_unanswered_call_of_a_pair_is_repaired() {
        let rows = stored(vec![
            user(vec![text("q")]),
            assistant(vec![call("a"), call("b")]),
            user(vec![result("a", "r")]),
        ]);
        let missing = missing_results(&rows);
        assert_eq!(missing.len(), 1);
        assert!(matches!(
            &missing[0],
            ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "b"
        ));
        assert_valid_for_provider(&build_messages(&rows));
    }

    #[test]
    fn repair_row_after_later_messages_moves_up_and_is_not_repeated() {
        let rows = stored(vec![
            user(vec![text("q1")]),
            assistant(vec![call("a")]),
            user(vec![text("q2")]),
            user(vec![result("a", "repaired")]),
            user(vec![text("q3")]),
        ]);
        assert!(missing_results(&rows).is_empty(), "repair must be stable");
        let messages = build_messages(&rows);
        assert_valid_for_provider(&messages);
        assert_eq!(
            messages[2].content,
            vec![result("a", "repaired"), text("q2"), text("q3")]
        );
    }

    #[test]
    fn orphan_and_duplicate_results_are_dropped_from_the_context() {
        let rows = stored(vec![
            user(vec![result("gone", "orphan"), text("q")]),
            assistant(vec![call("a")]),
            user(vec![result("a", "first")]),
            user(vec![result("a", "late duplicate")]),
        ]);
        let messages = build_messages(&rows);
        assert_valid_for_provider(&messages);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, vec![text("q")]);
        assert_eq!(messages[2].content, vec![result("a", "first")]);
    }

    #[test]
    fn reused_tool_id_pairs_each_call_with_its_own_result() {
        let rows = stored(vec![
            user(vec![text("q")]),
            assistant(vec![call("a")]),
            user(vec![result("a", "one")]),
            assistant(vec![call("a")]),
            user(vec![result("a", "two")]),
        ]);
        let messages = build_messages(&rows);
        assert_eq!(messages[2].content, vec![result("a", "one")]);
        assert_eq!(messages[4].content, vec![result("a", "two")]);
    }

    #[test]
    fn window_starting_with_assistant_gets_a_user_opening() {
        let rows = stored(vec![assistant(vec![text("hello")]), user(vec![text("q")])]);
        let messages = build_messages(&rows);
        assert_valid_for_provider(&messages);
        assert_eq!(messages[0].content, vec![text(WINDOW_START)]);
    }

    #[test]
    fn empty_text_and_empty_rows_are_dropped() {
        let rows = stored(vec![
            user(vec![text("q")]),
            assistant(vec![text("")]),
            user(vec![]),
            assistant(vec![text("a")]),
        ]);
        let messages = build_messages(&rows);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content, vec![text("a")]);
    }

    /// A turn of `q` then a padded tool pair, then a reply.
    fn tool_turn(padding: usize) -> Vec<Message> {
        vec![
            user(vec![text("q")]),
            assistant(vec![call("a")]),
            user(vec![result("a", &"r".repeat(padding))]),
            assistant(vec![text("done")]),
        ]
    }

    #[test]
    fn split_never_separates_a_tool_pair() {
        let mut messages = tool_turn(100);
        messages.push(user(vec![text("next")]));
        let rows = stored(messages);
        // The budget fits from the tool result onward, which is where a naive cut would land.
        let tail: usize = rows[2..].iter().map(|r| stored_chars(&r.message)).sum();
        assert_eq!(compaction_split(&rows, tail), Some(4));
    }

    #[test]
    fn split_at_every_budget_keeps_pairs_whole() {
        let mut messages = tool_turn(50);
        messages.extend(tool_turn(50));
        messages.extend(tool_turn(50));
        messages.push(user(vec![text("next")]));
        let rows = stored(messages);
        for budget in 0..800 {
            let cut = compaction_split(&rows, budget).expect("a turn start exists");
            assert!(cut >= 1);
            assert_eq!(rows[cut].message.role, Role::User);
            assert!(!matches!(
                rows[cut].message.content[0],
                ContentBlock::ToolResult { .. }
            ));
            assert!(missing_results(&rows[cut..]).is_empty(), "budget {budget}");
            assert_valid_for_provider(&build_messages(&rows[cut..]));
        }
    }

    #[test]
    fn split_keeps_recent_turns_that_fit() {
        let mut messages = tool_turn(1000);
        messages.extend(tool_turn(10));
        messages.push(user(vec![text("next")]));
        let rows = stored(messages);
        let tail: usize = rows[4..].iter().map(|r| stored_chars(&r.message)).sum();
        assert_eq!(compaction_split(&rows, tail), Some(4));
    }

    #[test]
    fn split_keeps_the_last_turn_even_when_it_exceeds_the_budget() {
        let rows = stored(vec![
            user(vec![text("old")]),
            assistant(vec![text("reply")]),
            user(vec![text(&"x".repeat(5000))]),
        ]);
        assert_eq!(compaction_split(&rows, 10), Some(2));
    }

    #[test]
    fn split_is_none_without_a_later_turn() {
        assert_eq!(compaction_split(&stored(vec![]), 10), None);
        assert_eq!(
            compaction_split(&stored(vec![user(vec![text("only")])]), 0),
            None
        );
        // Mid-turn with no later user text: nothing safe to cut.
        let rows = stored(vec![
            user(vec![text("q")]),
            assistant(vec![call("a")]),
            user(vec![result("a", "r")]),
        ]);
        assert_eq!(compaction_split(&rows, 0), None);
    }

    #[test]
    fn split_does_not_cut_between_a_call_and_a_result_stored_late() {
        let rows = stored(vec![
            user(vec![text("q1")]),
            assistant(vec![call("a")]),
            user(vec![text("q2")]),
            user(vec![result("a", "late")]),
            user(vec![text("q3")]),
        ]);
        assert_eq!(compaction_split(&rows, 0), Some(4));
    }
}
