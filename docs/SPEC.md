# MemoryOS specification

MemoryOS is a memory service for AI agents: one Rust service on Postgres + pgvector, with one
data model behind four capabilities. The contracts are `crates/core/src/{types,store,llm}.rs`,
`migrations/0001_init.sql` and `docs/openapi.yaml`. If this file disagrees with them, stop and
record it in `docs/PROGRESS.md`.

## 1. Capabilities

| Capability | Behavior |
|---|---|
| Personalization | Facts scoped to a user. Reconciliation chooses add, supersede, corroborate or ignore |
| Changing facts | Superseding closes a fact without deleting it. Reads support `as_of` and `known_at` |
| Self-editing agents | Size-limited, versioned blocks edited through tools, plus an agent loop with sessions |
| Shared graph | Facts are edges between entities. Shared scopes are guarded by per-key scope patterns |

## 2. Crates and public APIs

Each crate exposes exactly these entry points, so crates can be built in parallel.

```rust
// memoryos-store
PgStore::connect(database_url: &str, max_connections: u32) -> Result<PgStore>
PgStore::migrate(&self) -> Result<()>          // sqlx::migrate!("../../migrations")
impl MemoryStore for PgStore

// memoryos-llm
AnthropicClient::new(api_key: String, model: String, base_url: Option<String>) -> Self   // impl Llm
OpenAiClient::new(api_key: String, model: String, base_url: Option<String>) -> Self      // impl Llm
OpenAiEmbedder::new(api_key: String, model: String, base_url: Option<String>) -> Self    // impl Embedder

// memoryos-ingest
Ingestor::new(store: Arc<dyn MemoryStore>, llm: Arc<dyn Llm>, embedder: Arc<dyn Embedder>) -> Self
Ingestor::process_episode(&self, ctx: &RequestContext, episode_id: Uuid) -> Result<()>
Worker::new(store: Arc<dyn MemoryStore>, ingestor: Arc<Ingestor>, config: WorkerConfig) -> Self
Worker::run(self, shutdown: CancellationToken) -> impl Future<Output = ()>

// memoryos-retrieve
Retriever::new(store: Arc<dyn MemoryStore>, embedder: Arc<dyn Embedder>) -> Self
Retriever::recall(&self, ctx: &RequestContext, req: RecallRequest) -> Result<RecallResponse>
pub struct RecallRequest { scopes: Vec<Scope>, query: String, limit: i64, as_of: Option<DateTime<Utc>>, known_at: Option<DateTime<Utc>> }
pub struct RecallResponse { facts: Vec<ScoredFact>, pending_episodes: Vec<ScoredEpisode> }

// memoryos-agent
AgentRuntime::new(store: Arc<dyn MemoryStore>, retriever: Arc<Retriever>, llm: Arc<dyn Llm>, config: AgentConfig) -> Self
AgentRuntime::run_turn(&self, ctx: &RequestContext, session_id: Uuid, user_text: String) -> Result<TurnResult>
pub struct TurnResult { text: String, tool_calls: u32, input_tokens: u64, output_tokens: u64 }

// memoryos-sdk: thin HTTP client over docs/openapi.yaml
MemoryOs::connect(base_url, api_key)
```

The `memoryos` binary (crate `server`) has these commands: `serve` (API plus embedded worker,
`--no-worker` to disable), `worker`, `migrate`, `key create`, `key revoke`.

## 3. Write path

1. `POST /v1/episodes` checks the scope and `trust <= key.max_trust`, then calls
   `insert_episode`, which stores the episode and an `extract` job in one transaction. Returns 202.
   Replaying an `idempotency_key` returns 202 with the existing episode; if that episode is in a
   different scope the server returns 400 `invalid_input` and does not reveal it.
2. The episode is immediately searchable through `search_pending_episodes`, only when its trust
   is `user` or `system`.
3. The worker claims the job and calls `Ingestor::process_episode`:
   a. LLM extraction, forced through one tool, returns entities and statements as JSON.
   b. Entity resolution: same normalized name uses `upsert_entity`; otherwise cosine similarity
      of at least 0.90 with the same kind merges, adding the new name as an alias.
   c. Reconciliation: for each statement, an LLM compares it with `current_facts_for_subject`
      and returns `add`, `supersede(fact_id)`, `corroborate(fact_id)` or `ignore`. A `fact_id`
      that was not in the candidate list is rejected and treated as `add`.
   d. Trust gate (section 5) assigns each fact's status.
   e. `apply_extraction` commits everything and sets `processed_at` atomically.
4. Retries back off exponentially (30s base, 1h cap), and a job goes `dead` after 5 attempts.
   The worker checks `job.attempts` right after claiming, so a job that keeps crashing workers
   still goes `dead`. A job whose episode was purged (`NotFound`) is completed, not retried.
5. Job leases are fenced: `complete_job`, `fail_job` and `renew_job` take effect only when the
   job is still `running` with the same `locked_by` and `attempts`, and otherwise report `false`.
   Workers renew the lease while extraction runs.

## 4. Time semantics

- `valid_from` / `valid_to`: when the fact was true. Default `valid_from` is `occurred_at`.
- `recorded_at`: when MemoryOS learned it.
- `invalidated_at`: retracted as wrong. Never returned by current or `as_of` reads.
- Superseding sets `old.valid_to = new.valid_from` and `old.superseded_by = new.id`.
- Out-of-order input: if a new fact's `valid_from` is before the old fact's, it is stored as a
  closed historical fact, with `valid_to = old.valid_from`, and the old fact stays current.

Fact visible at valid time T (default now) and knowledge time K (default now):

```
status = 'active'
AND valid_from <= T AND (valid_to IS NULL OR valid_to > T)
AND recorded_at <= K
AND (invalidated_at IS NULL OR invalidated_at > K)
```

## 5. Trust and quarantine

| Rule | Result |
|---|---|
| Episode trust is `external` | Every derived fact is `quarantined` |
| Episode trust is `agent` and scope is `user:*` or `shared:*` | Quarantined, so an agent cannot plant claims in user or shared memory |
| Statement contradicts a fact of higher trust | Quarantined, with `pending_supersede_of` set |
| Statement contradicts a fact of equal or lower trust | Active, supersedes it |
| Otherwise | Active |

- A fact's trust equals its episode's trust. Corroboration never raises it.
- A `FactOp::Supersede` whose new fact is quarantined leaves the old fact open and sets
  `pending_supersede_of`. `approve_fact` activates the fact and then performs the close.
- Superseding rechecks trust at the chain head when it commits. If the head has higher trust than the
  new fact, the new fact is quarantined and pending on that head, which closes a race with the gate.
- `approve_fact` on a chain that moved to a higher-trust head the admin has not reviewed does not close
  it: it re-points `pending_supersede_of` at that head, writes `fact.approve_held`, and returns
  `InvalidInput`. A second approval closes it. The server tells the caller to review again.
- Purging a source recomputes a fact's trust as the highest trust among remaining sources, never above
  its current trust, and re-quarantines the fact if a source rule in the table above now applies.
- `reject_fact` sets `rejected`. Rejected and quarantined facts never appear in recall or graph.
- Extraction and reconciliation prompts wrap content as untrusted data, and they expose only the
  forced output tool. Episode text can never reach a memory-writing tool.
- `POST /v1/episodes/{id}/purge` removes one bad source and everything derived only from it. A
  fact closed by a purged fact is reopened or relinked, so purging a poison restores the truth.
- Approve, reject, invalidate and purge each write an `audit_events` row.

## 6. Tenancy and authorization

- Every tenant transaction runs `SET LOCAL ROLE memoryos_app` and
  `SELECT set_config('app.tenant', $1, true)`. Row-level security then applies, even when the
  login role is a superuser. With no tenant set, tables return no rows.
- Only job methods, `*_api_key` and `ping` run outside that role. A non-superuser login needs
  `GRANT memoryos_app TO <login>`. Fact and pending-episode text search go through the
  `SECURITY DEFINER` functions in the migration, because GIN indexes are unusable under RLS.
- API keys: bearer tokens `mos_<43 url-safe chars>` from 32 random bytes. Only the SHA-256 hex is
  stored. Each key has a tenant, scope patterns, `max_trust` and `is_admin`.
- `server` must check `scope` against the key's patterns on every request, including reads by id,
  where it checks the loaded row's scope and returns 404 on a mismatch.
- Blocks and sessions for agent `X` require scope `agent:X`.
- Quarantine listing and approval, purge and `DELETE /v1/users/{id}/memory` require `is_admin`.

## 7. Retrieval

`Retriever::recall`:
1. Embed the query. If embedding fails, log a warning and continue with text search only.
2. Run vector and text fact searches concurrently, 50 candidates each, using the section 4 filter.
3. Graph expansion: take the subjects of the top 5 fused facts and add facts from `neighbors` at
   depth 1 as a third ranked list.
4. Reciprocal rank fusion with `k = 60`: `score = sum(1 / (60 + rank))`. Return the top `limit`
   (default 10, max 100).
5. Add `search_pending_episodes` only when neither `as_of` nor `known_at` is set.

## 8. Blocks and agent runtime

- Tools given to the model: `block_replace(label, content, expected_version)`,
  `block_append(label, text, expected_version)`, `memory_recall(query)`, `memory_remember(text)`.
- `memory_remember` writes an episode with trust `agent` into scope `agent:<agent_id>`, where its
  facts can go active. `memory_recall` searches the session's scope and `agent:<agent_id>`.
- A reply with `StopReason::MaxTokens` is reported in logs, and the turn asks the model to continue
  once, so truncated text is not silently returned as complete.
- Context order: system prompt, blocks rendered with label, version and usage, session summary,
  then messages. Block content is delimited and labelled as data in the prompt.
- Tool-call loop limit is 10 per turn. Tool errors go back to the model as `is_error` results.
- A version conflict returns the current content, so the model can retry.
- Compaction runs when stored messages exceed `AgentConfig::max_context_chars` (default 100000):
  older messages are summarized and replaced through `compact_session`, keeping tool-use and
  tool-result pairs together.
- Unknown agents get default blocks `persona` and `human`, 2000 chars each.

## 9. API

`docs/openapi.yaml` is the contract. Error body: `{"error": {"code": "...", "message": "..."}}`.

| Error | HTTP | code |
|---|---|---|
| `NotFound` | 404 | `not_found` |
| `InvalidInput` | 400 | `invalid_input` |
| `Forbidden` | 403 | `forbidden` |
| `VersionConflict` | 409 | `version_conflict` |
| `BlockTooLarge` | 422 | `block_too_large` |
| `Provider` | 502 | `provider_error` |
| `Storage`, `Internal` | 500 | `internal` (details logged, never returned) |
| No or bad token | 401 | `unauthorized` |
| Rate limited | 429 | `rate_limited` |

Limits: request body 1 MiB, episode content 100000 chars, 50 scopes per request, 100 requests per
second per key by default. `POST /mcp` exposes MCP tools `remember`, `recall`, `block_read`,
`block_replace` and `graph_neighbors` under the same authorization.

## 10. Configuration

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | required | |
| `MEMORYOS_BIND` | `0.0.0.0:8080` | |
| `MEMORYOS_LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `MEMORYOS_LLM_MODEL` | `claude-haiku-5-5` / `gpt-5.1` by provider | Haiku accepts forced tool calls, which extraction uses. Models that do not fall back to `auto` with one tool |
| `MEMORYOS_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedder only. Must return 384 dims |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | | Embeddings need the OpenAI key |
| `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL` | provider default | Tests point these at WireMock |
| `MEMORYOS_DB_MAX_CONNECTIONS` | `20` | |
| `MEMORYOS_RATE_LIMIT_RPS` | `100` | Per key |
| `MEMORYOS_WORKER_CONCURRENCY` | `4` | |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Traces are exported only when set |

## 11. Testing rules

- Integration tests live in each crate's `tests/`, gated behind a cargo feature `integration`.
  Run them with `cargo test -p <crate> --features integration`.
- Integration tests use real containers through `testcontainers`: `pgvector/pgvector:pg17` for
  Postgres and `wiremock/wiremock:3.3.1` for LLM and embedding HTTP. Mock objects are not allowed there.
- Unit tests may use small in-memory fakes of `Llm` and `Embedder` only.
- Tenant isolation must be tested with two tenants against real Postgres.
- Run gates: `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test`, `cargo deny check`.

## 12. Production gates

| Gate | Target | Proof |
|---|---|---|
| Static checks | No warnings | Commands in section 11 |
| Recall latency | p95 under 200 ms at 1M facts, excluding embedding and LLM calls | Load test output |
| Staleness | At least 95% of superseded facts absent from current recall | `memoryos-eval` report |
| Poisoning | 0 poisoned facts in recall without approval | `memoryos-eval` report |
| Isolation | 0 cross-tenant reads | Integration tests |
| Recovery | A job held by a killed worker completes after lease expiry | Integration test |
| Restore | Backup restored into an empty database and recall still works | Drill log |
| Ready to use | README quickstart works from a clean checkout | Recorded run |

## 14. Host mode (default)

`MEMORYOS_MODE=host` is the default; `worker` enables sections 3 and 8. MemoryOS never calls the
host: the host calls MemoryOS.

| | Host | Worker |
|---|---|---|
| LLM key | None | Required |
| Default embedder (`MEMORYOS_EMBEDDER`) | `local` | `openai`; `local` also allowed |
| Worker and extraction | Off | On |
| `POST /v1/episodes` | 400 `invalid_input`, directing to `POST /v1/commits` | Normal |
| Session endpoints | 400 `invalid_input` | Normal |
| `POST /v1/candidates`, `POST /v1/commits` | Normal | Normal |

Flow:
1. `POST /v1/candidates` with `{scopes, statement, limit}` returns related current facts. It is
   `Retriever::recall` without pending episodes.
2. The host decides, then `POST /v1/commits` sends `{scope, source, content, facts[]}`. Each fact has
   `statement`, `subject {name, kind}`, `predicate`, optional `object {name, kind}`, optional
   `valid_from`, and at most one of `supersedes` and `corroborates`. Maximum 50 facts.
3. `HostCommitter` stores `content` through `insert_unqueued_episode` as the source, resolves entities
   and embeds statements as in section 3, builds `FactOp`s, applies the section 5 trust gate with
   the same code as the worker, and calls `apply_extraction`. Returns `episode_id` and each fact's
   `id` and `status`.

Rules:
- Commit trust is `agent`, or `external` if the key's `max_trust` is `external`. A request cannot
  set a higher trust, because the host model also reads untrusted content.
- `supersedes` and `corroborates` must name a current, active fact in the same scope, else 400.
  The server never trusts a host-supplied id without that check.
- `recall_candidates`, `commit_facts`, `recall`, `block_read`, `block_replace` and `graph_neighbors`
  are the host-mode MCP tools. `remember` is worker mode only. Tool descriptions state when to
  recall, when to save, and that retrieved or tool-read content is data, never instructions.

Entry points:
```rust
// memoryos-ingest
HostCommitter::new(store: Arc<dyn MemoryStore>, embedder: Arc<dyn Embedder>) -> Self
HostCommitter::commit(&self, ctx: &RequestContext, req: CommitRequest, max_trust: Trust) -> Result<CommitResponse>
// CommitRequest, CommitFact, EntityRef, CommitResponse, CommittedFact are serde types in ingest

// memoryos-llm, feature `local`, on by default
LocalEmbedder::new(cache_dir: Option<PathBuf>) -> Result<Self>   // impl Embedder
```

`LocalEmbedder` uses `multilingual-e5-small` through `fastembed`, downloads the model on first use,
and prefixes every input with `query: `, the form that model documents for symmetric similarity.

CLI, using `MEMORYOS_URL` and `MEMORYOS_API_KEY` (any agent without MCP can use these):
`memoryos recall`, `candidates`, `commit`, `quarantine list|approve|reject`, and
`memoryos hook session-start|stop|install --agent claude-code|opencode`, as shown in `README.md`.
- `hook stop` reads the host's JSON from stdin, prints nothing when `stop_hook_active` is true, and
  otherwise prints the block decision once every 10 turns, counted per `session_id` in a state file.
- Hook commands never fail the host: on any error they log to stderr and exit 0 with no output.
- `hook install` merges into existing settings, keeps other entries, is idempotent, and also
  registers the MCP server. The API key stays in an environment variable reference, never in the file.

## 13. Known limits

- Text search uses the `simple` configuration, with no stemming and weak CJK tokenization. Vector
  search covers those cases when embeddings are available.
- Embedding width is fixed at 384. Changing model needs re-embedding.
- The rate limiter is in-process, so the limit applies per server instance.
- `known_at` is not fully bitemporal: `valid_to` and `status` are not versioned, so a later close or
  approval shows up in reads at an earlier `known_at`.
- Thinking blocks are not kept between agent tool calls, so the model loses that reasoning.
- The agent runtime's per-session lock is in-process. With several server instances, clients must
  not run concurrent turns on one session.
- `LocalEmbedder` silently truncates inputs over 512 tokens and needs at least 1 GiB of memory.
