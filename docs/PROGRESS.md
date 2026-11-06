# MemoryOS progress

Only the lead edits this file. A task is `done` only with proof from a command or test that was
actually run. States: `todo`, `doing`, `done`, `blocked`, `removed`.

| Task | State | Proof |
|---|---|---|
| T0 Spec, core, schema, OpenAPI | done | clippy clean, 2 core tests pass, migration applied on pgvector:pg17, psql check: no tenant sees 0 rows, cross-tenant insert rejected by RLS, app role denied on `jobs` |
| T1 Postgres store | done | Lead run on `develop`: 4 unit and 51 integration tests pass against real Postgres. Open risk: common-word text search in a 1M-fact scope is estimated at about 450 ms, over the latency gate; T9 must measure it |
| T2 LLM and embedding clients | removed | Removed by T15, except `LocalEmbedder` (T11). In git history |
| T3 Eval harness | done | Lead run on `develop`: 51 unit and 6 WireMock tests pass. Not yet run against a real server |
| T4 Ingest and worker | removed | Worker and extraction removed by T15. `HostCommitter`, the trust gate and entity resolution stay |
| T5 Hybrid retrieval | done | Lead run on merged `develop`: 18 unit and 22 integration tests pass |
| T6 Agent runtime | removed | Removed by T15. In git history |
| T11 Local embedder | done | Lead run: 54 llm unit tests pass. Real-model test (6 pass) reported by the agent, not re-run by lead. Needs at least 1 GiB memory and glibc 2.38+ |
| T13 Hardening after Wave 2 | done | Lead run on merged `develop`: 387 tests pass, fmt and clippy clean |
| T14 Purge fixes | merged | Agent reports 3 unit and 73 integration store tests pass, 35 unit and 44 integration ingest tests pass. Lead workspace run at T7 merge passes |
| T7 Server, MCP, SDK | done | Lead workspace run on `develop`: 506 tests pass, fmt and clippy clean. MCP checked against spec text and tests, not a real host. No live providers |
| T12 CLI commands and hooks | doing | |
| T8 Deploy, CI, docs | doing | |
| T9 Integration and gates | todo | |
| T10 Independent review | todo | |

## Production gates (SPEC section 12)

| Gate | State | Proof |
|---|---|---|
| Static checks | todo | |
| Recall latency | todo | |
| Staleness | todo | |
| Poisoning | todo | |
| Isolation | todo | |
| Restore | todo | |
| Ready to use | todo | |

## Contract changes

Host mode only, by the user's decision (T15):

- MemoryOS never calls an LLM. Removed: `crates/agent`, the LLM clients, `Llm` and its types, `Job`,
  sessions, the worker, `POST /v1/episodes`, `/v1/sessions/*`, `POST /v1/agents/{id}/sessions`, MCP
  `remember`, `memoryos worker`, `--no-worker`, and the mode, provider and embedder settings.
- `memoryos-llm` is now `memoryos-embed`. `apply_extraction` is `apply_commit`, and
  `insert_unqueued_episode` is `insert_episode`. `RecallResponse` has only `facts`, and `/readyz` no
  longer reports `not_required`.
- The migration lost `jobs`, `sessions`, `session_messages`, `enqueue_job`,
  `search_pending_episode_ids`, `episodes.content_tsv` and the episode search indexes. `episodes` is
  back to `FORCE ROW LEVEL SECURITY`. `episodes.processed_at` stays: it makes a replayed commit a no-op.
- The Recovery gate is gone, because no job exists to recover. Entries below describe the removed
  design where they mention jobs, workers or Anthropic.

After Wave 1, from agent findings:

- `Fact.pending_supersede_of` added; `NewEpisode.metadata` defaults to `{}`.
- `CompletionResponse.stop_reason` added.
- Job methods fenced by worker and attempts; `renew_job` added.
- `agent` trust facts quarantined in `user:*` and `shared:*` scopes; pending episodes limited to
  `user` and `system` trust.
- Text search moved to `SECURITY DEFINER` functions because GIN is unusable under RLS.
- Indexes added on `superseded_by` and `pending_supersede_of`.
- Default Anthropic model changed to `claude-haiku-5-5` for forced tool calls. Not verified against
  the live API because no Anthropic key is available.
- Work merges to `develop`, not `main`.

No paid provider calls, by the user's decision:

- Staleness and poisoning gates run through `memoryos-eval --mode host`, with no LLM.
- Worker mode stays available but is unproven against real models.

Host mode, chosen by the user:

- Host mode is the default (ADR 0005, SPEC section 14). `insert_unqueued_episode` added.
- Embedding width changed from 1536 to 384 (ADR 0003).

After Wave 2, from T13:

- `MemoryStore` gains `find_entity`, `facts_for_episode`, `release_job`, and `neighbors` gains
  `edge_limit`. Core exports `normalize_name` and `source_requires_quarantine`.
- Migration gains `entities.alias_keys`, `facts_subject_recent_idx` and `facts_object_recent_idx`.
- Test Postgres runs on tmpfs, because each run was leaking an anonymous Docker volume.
