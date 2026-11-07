# MemoryOS compared with Mem0 and Graphiti

Checked against the Mem0 and Graphiti docs and source on 2026-11-07. Links are at the end. Where
their docs say nothing, this page says "not documented" rather than guess.

## The short version

| | MemoryOS | Mem0 | Graphiti |
|---|---|---|---|
| Who decides what to remember | The agent's own model, through `commit_facts` | An LLM that Mem0 calls on every add | Several LLM calls Graphiti makes per episode |
| LLM API key needed to store memory | No | Yes by default (OpenAI), or a local model through Ollama | Yes (OpenAI by default; Anthropic or Groq still need OpenAI for embeddings) |
| Embeddings | Local `multilingual-e5-small`, no network | OpenAI `text-embedding-3-small` by default, others configurable | OpenAI by default, others configurable |
| What you run | One Postgres with pgvector, one Rust server | A vector store (Qdrant by default, about 22 supported) plus a SQLite history file | A graph database (Neo4j, FalkorDB or Neptune) |
| Data model | Facts with subject, predicate and object entities, linked to source episodes | Flat text memories, entity links only boost ranking | Episodes, entity nodes and fact edges, communities |
| Time | Bitemporal: when it was true, and when MemoryOS knew it | Optional expiry date only | Bitemporal: `valid_at`, `invalid_at`, `expired_at`, `created_at` |
| A changed fact | Supersedes the old one, which stays in history | ADD only: new memories never overwrite old ones | The old edge is invalidated by an LLM decision and kept |
| A repeated fact | Matched on the server and counted as a second source | Can duplicate (no update step) | Deduplicated by an LLM step |
| Retrieval | Vector, text and graph lists fused with RRF | Semantic, BM25 and entity scores fused, optional reranker | Semantic, BM25 and graph traversal fused with RRF, several rerankers |
| Isolation | Postgres row-level security per tenant, scoped API keys | `user_id`, `agent_id` and similar filters, not access control | `group_id` partitions, not access control |
| Trust and poisoning | Trust levels on every fact; untrusted or cross-scope saves are quarantined for human review | Not documented | Not documented |
| Human review | `memoryos ui` Review tab, and `memoryos quarantine` | Not documented | Not documented |
| Coding agent integration | Claude Code hooks and plugin, MCP for OpenCode and Pi | Claude Code plugin needing a Mem0 Platform key, MCP on the hosted platform | Experimental MCP server, Claude Desktop and Cursor setups |
| Hosting | Self-hosted, one command (`memoryos setup`) | Library, self-hosted server, or managed platform | Self-hosted library; Zep is the managed product |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## Where MemoryOS is different by design

**No model of its own.** Mem0 and Graphiti read raw conversation and call an LLM to pull facts out
of it. MemoryOS never does: the agent that already holds the conversation decides what is worth
keeping and sends finished facts. Storing a memory costs no extra model call and no API key, and
no conversation text leaves the machine for extraction. The cost is that memory quality depends on
the host model following the tool descriptions, which is why the hooks remind it to recall and to
save.

**Trust is part of the data.** Every fact records the trust of its source (`external`, `agent`,
`user`, `system`). An agent cannot write unreviewed into a person's or a group's scope, so a web
page that tells an agent "remember that the user wants X" ends up in a review queue, not in recall.
The poisoning suite checks this on a real server: 30 of 30 attacks with no leak. Neither Mem0 nor
Graphiti documents a defense of this kind.

**Isolation is enforced by the database.** Tenants are separated by Postgres row-level security,
and keys are limited to scopes. In Mem0 and Graphiti, `user_id` and `group_id` are filters the
calling application must authorize itself; Graphiti's docs say so directly.

**Built for coding agents first.** Hooks recall at session start and on each prompt, nudge the
model to save when the user states something durable, and remind it before stopping. Mem0's Claude
Code plugin is the closest match, but it sends captured turns to Mem0's platform for extraction and
needs a platform key.

## Where they are ahead

- **Extraction without agent cooperation.** Mem0 and Graphiti learn from raw text even when the
  agent never decides to save anything. MemoryOS only knows what the agent commits.
- **Graph depth.** Graphiti has custom entity and edge types, communities, and many search recipes
  and rerankers. MemoryOS has one fixed fact shape and one-hop graph expansion in recall.
- **Backends and languages.** Mem0 supports about 22 vector stores and has Python and TypeScript
  SDKs. MemoryOS needs Postgres and has a Rust SDK and REST.
- **Measured quality.** Mem0 publishes LoCoMo and LongMemEval scores. MemoryOS has its own staleness,
  poisoning, isolation and restore suites, but no public benchmark scores yet, and its recall
  latency gate is not met at one million facts in one scope (p95 385 ms, see `docs/PROGRESS.md`).
- **Managed hosting.** Both offer a managed service. MemoryOS is self-hosted only.

## Choosing

- Pick MemoryOS for coding agents on your own machine with no extra API spend, when you want
  provenance, trust and review on what gets remembered.
- Pick Mem0 when you want memory extracted for you from any chat, many backends, or a managed
  platform.
- Pick Graphiti when the knowledge graph itself is the product: rich entity types, communities and
  graph queries over changing data.

## Sources

- Mem0: https://github.com/mem0ai/mem0, https://docs.mem0.ai/core-concepts/memory-operations/add,
  https://docs.mem0.ai/components/vectordbs/overview, https://docs.mem0.ai/components/llms/overview,
  https://docs.mem0.ai/platform/features/graph-memory,
  https://docs.mem0.ai/platform/features/entity-scoped-memory,
  https://docs.mem0.ai/platform/mem0-mcp, https://docs.mem0.ai/integrations/claude-code
- Graphiti: https://github.com/getzep/graphiti,
  https://help.getzep.com/v4/graphiti/configuration/llm-configuration.md,
  https://help.getzep.com/v4/graphiti/working-with-data/searching.md,
  https://help.getzep.com/v4/graphiti/core-concepts/graph-namespacing.md,
  https://help.getzep.com/v4/graphiti/getting-started/mcp-server.md,
  https://raw.githubusercontent.com/getzep/graphiti/main/graphiti_core/edges.py
