# MemoryOS progress

Only the lead edits this file. A task is `done` only with proof from a command or test that was
actually run. States: `todo`, `doing`, `done`, `blocked`.

| Task | State | Proof |
|---|---|---|
| T0 Spec, core, schema, OpenAPI | done | clippy clean, 2 core tests pass, migration applied on pgvector:pg17, psql check: no tenant sees 0 rows, cross-tenant insert rejected by RLS, app role denied on `jobs` |
| T1 Postgres store | doing | |
| T2 LLM and embedding clients | doing | |
| T3 Eval harness | doing | |
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

None yet.
