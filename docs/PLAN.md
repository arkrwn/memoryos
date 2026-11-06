# MemoryOS development plan

Goal: production-grade and ready to use. `docs/SPEC.md` section 12 defines done.

## Rules for every agent

1. Read `docs/SPEC.md`, `crates/core/src/*` and `migrations/0001_init.sql` before writing code.
2. Change only the paths your task owns. Add dependencies in your own crate's `Cargo.toml`, not
   the workspace root. Do not edit `core`, the migration, the OpenAPI file or `PROGRESS.md`.
3. If a contract is wrong, do not work around it silently. Report it in your final message.
4. Prove your work: run the gates from SPEC section 11 for your crate and report the actual output.
5. Commit to your worktree branch with a `feat: <crate> | <what>` message. Never merge or push.

## Waves

| Wave | Task | Owner | Paths | Depends on |
|---|---|---|---|---|
| 0 | Spec, core, schema, OpenAPI | Lead | `docs/`, `crates/core`, `migrations/` | |
| 1 | T1 Postgres store | Agent | `crates/store` | Wave 0 |
| 1 | T2 LLM and embedding clients | Agent | `crates/llm` | Wave 0 |
| 1 | T3 Eval harness, HTTP black box | Agent | `crates/eval` | Wave 0 |
| 2 | T4 Ingest and worker | Agent | `crates/ingest` | T1, T2 |
| 2 | T5 Hybrid retrieval | Agent | `crates/retrieve` | T1, T2 |
| 2 | T6 Agent runtime | Agent | `crates/agent` | T1, T2 |
| 2 | T11 Local embedder | Agent | `crates/llm` | T2 |
| 3 | T7 Server, MCP, SDK, both modes | Agent | `crates/server`, `crates/sdk` | T4, T5, T6, T11 |
| 3b | T12 CLI commands and hooks | Agent | `crates/hooks` | T7 |
| 3b | T8 Deploy, CI, docs | Agent | `Dockerfile`, `deploy/`, `.github/`, `README.md`, `deny.toml`, `docs/runbook.md` | T7 |
| 4 | T9 Integration and gates | Lead | Whole repo | T7, T8 |
| 4 | T10 Independent review | Fresh agent | Read-only | T7, T8 |

The lead merges each wave into `develop` and runs the workspace gates before the next wave starts.
`main` holds only the approved baseline until the user confirms a merge.
