# MemoryOS progress

Only the lead edits this file. A task is `done` only with proof from a command or test that was
actually run. States: `todo`, `doing`, `done`, `blocked`.

| Task | State | Proof |
|---|---|---|
| T0 Spec, core, schema, OpenAPI | done | clippy clean, 2 core tests pass, migration applied on pgvector:pg17, psql check: no tenant sees 0 rows, cross-tenant insert rejected by RLS, app role denied on `jobs` |
| T1 Postgres store | done | Lead run on `develop`: 4 unit and 51 integration tests pass against real Postgres. Open risk: common-word text search in a 1M-fact scope is estimated at about 450 ms, over the latency gate; T9 must measure it |
| T2 LLM and embedding clients | done | Lead run on `develop`: 36 unit and 29 WireMock tests pass. Not run against live provider APIs |
| T3 Eval harness | done | Lead run on `develop`: 51 unit and 6 WireMock tests pass. Not yet run against a real server |
| T4 Ingest and worker | todo | |
| T5 Hybrid retrieval | todo | |
| T6 Agent runtime | todo | |
| T7 Server, MCP, SDK | todo | |
| T8 Deploy, CI, docs | todo | |
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
| Recovery | todo | |
| Restore | todo | |
| Ready to use | todo | |

## Contract changes

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
