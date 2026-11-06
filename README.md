<div align="center">

# 🧠 MemoryOS

**Long-term memory for AI agents that stays current, traceable and hard to poison.**

![Status: pre-release](https://img.shields.io/badge/status-pre--release-orange)
![Rust 1.92+](https://img.shields.io/badge/rust-1.92%2B-b7410e?logo=rust)
![Postgres + pgvector](https://img.shields.io/badge/postgres-pgvector-336791?logo=postgresql&logoColor=white)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

[Why MemoryOS](#-why-memoryos) · [Quickstart](#-quickstart) · [How it works](#-how-it-works) · [Security](#-security) · [Agent hooks](#-agent-hooks) · [Roadmap](#-roadmap) · [Docs](#-documentation)

</div>

> [!NOTE]
> **Pre-release.** The server, REST API, MCP endpoint and Rust SDK run, and the Docker stack below is
> tested end to end in host mode. Not done yet: integration gates, a load test and independent review.
> It has not run against a live LLM provider or a real agent host. Track the work in
> [`docs/PROGRESS.md`](docs/PROGRESS.md).

## ✨ Why MemoryOS

Most agent memory stops at "store text, search it later". Real memory also has to handle facts that
change and sources that lie.

| Capability | What you get |
|---|---|
| 👤 **Personal memory** | Facts scoped per user, reconciled so new information adds to, replaces or repeats what is known |
| 🕰️ **Memory over time** | A replaced fact is closed, not deleted, so you can ask what is true now or what was true then |
| ✍️ **Self-editing agents** | Size-limited, versioned blocks that agents edit through tools, with conflict detection |
| 🕸️ **Shared knowledge graph** | Every fact is also an edge between entities, shared across agents through scoped access |
| 🛡️ **Poisoning defense** | Low-trust content is quarantined, and every fact traces back to its source |

## 🚀 Quickstart

Needs Docker, and at least 1 GiB of memory for the server container. The first image build takes a
few minutes, and the first start downloads a 481 MB embedding model into a named volume.

```bash
cp deploy/.env.example deploy/.env
sed -i.bak "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 32)/" deploy/.env && rm deploy/.env.bak
docker compose -f deploy/docker-compose.yml up -d --build

# Ready once the model has loaded, about 2 minutes on a first start.
until curl -sf http://localhost:8080/readyz; do sleep 5; done

TOKEN=$(docker compose -f deploy/docker-compose.yml exec -T memoryos memoryos key create \
  --tenant demo --name quickstart --scopes 'user:*,agent:*' --max-trust agent --admin)
```

Save a fact, in the agent's own scope, then recall it:

```bash
curl -sS http://localhost:8080/v1/commits -H "Authorization: Bearer $TOKEN" -d '{
  "scope": "agent:demo",
  "source": "quickstart",
  "content": "The team deploys to Hetzner.",
  "facts": [{
    "statement": "The team deploys to Hetzner",
    "subject": {"name": "the team", "kind": "group"},
    "predicate": "deploys_to",
    "object": {"name": "Hetzner", "kind": "organization"}
  }]
}'
# {"episode_id":"...","facts":[{"id":"...","status":"active"}]}

curl -sS http://localhost:8080/v1/recall -H "Authorization: Bearer $TOKEN" \
  -d '{"scopes": ["agent:demo"], "query": "where does the team deploy?"}'
```

The same write to `user:alice` comes back `"status":"quarantined"`, and recall leaves it out until an
admin approves it. MCP clients connect to `POST /mcp` with the same token:

```bash
curl -sS http://localhost:8080/mcp -H "Authorization: Bearer $TOKEN" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'
```

Backups, upgrades, key rotation, worker mode and incidents are in the
[runbook](docs/runbook.md).

## 🔍 Example

This uses the Rust SDK against a server in worker mode, where MemoryOS extracts facts itself. The user
says *"I moved to Bandung"* after previously living in Jakarta:

```rust
let mem = MemoryOs::connect(url, api_key).await?;
let user = Scope::user("u_123");

mem.remember(user.clone(), Episode::message("I moved to Bandung")).await?;

// Current answer: Bandung
let now = mem.recall(user.clone(), "where does the user live?").await?;

// Answer as of January 2025: Jakarta
let then = mem.recall(user, "where does the user live?")
    .as_of(date(2025, 1, 1))
    .await?;
```

## ⚙️ How it works

```mermaid
flowchart TD
    A[Agents and apps] -->|REST or MCP| S[memoryos-server<br/>auth, scopes, limits, telemetry]
    S -->|host mode: commit| G[Trust gate]
    S -->|worker mode: episode| E[Episode + job<br/>one transaction]
    S -->|read| R[Hybrid retrieval<br/>vector + text + graph]
    E --> W[Worker<br/>extract, resolve entities, reconcile]
    W --> G
    G --> DB[(Postgres + pgvector)]
    R --> DB
```

Everything reads from **one data model**, so text search, vector search and the graph cannot
disagree:

- **Episodes:** immutable raw input, the source of everything else.
- **Entities:** the people, places and things that facts connect.
- **Facts:** statements with two timelines, when they were true and when we learned them.
- **Blocks:** versioned working notes that an agent edits directly.

### Two ways to run

| | 🖥️ Host mode (default) | ☁️ Worker mode |
|---|---|---|
| **Who extracts facts** | The coding agent's own model | A background worker calling an LLM API |
| **API key needed** | None | OpenAI, or Anthropic plus OpenAI |
| **Embeddings** | Local model, downloaded on first use | OpenAI, or the local model |
| **Runs when the agent is closed** | No | Yes |
| **Best for** | Claude Code, OpenCode and similar agents | Servers and multi-user apps |

## 🛡️ Security

Research shows a poisoned memory survives across sessions, unlike ordinary prompt injection. So:

| Source | What happens to its facts |
|---|---|
| `external` content, such as web pages and documents | Quarantined until approved |
| `agent` writes to user or shared memory | Quarantined until approved |
| `agent` writes to its own scope | Active |
| Lower-trust statement contradicting higher-trust fact | Quarantined, and the original stays in place |
| `user` or `system` statements otherwise | Active |

- **Tenant isolation:** Postgres row-level security enforces it, so a missed filter in the code
  cannot leak another tenant's data.
- **Purging:** removing one bad source deletes what was derived only from it and restores any fact
  it had replaced.
- **Audit trail:** approvals, rejections, retractions and purges are all recorded.

## 🪝 Agent hooks

In host mode, the agent reads and saves memories through MCP tools. Hooks are optional. They only
load memories and remind the agent to save, and they cannot bypass validation or quarantine.

```mermaid
flowchart LR
    A[Session start] -->|hook: load memories| B[During work]
    B -->|recall_candidates, commit_facts| B
    B --> C[Before stop]
    C -->|hook: reminder to save| D[Done]
```

Without hooks, memory still works, but saving depends on the agent choosing to do it. A hook may
not run if the agent crashes or is killed, so agents should also save during work.

<details>
<summary><b>Claude Code setup</b></summary>

`.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|compact",
        "hooks": [{ "type": "command", "command": "memoryos hook session-start" }]
      }
    ],
    "Stop": [
      {
        "hooks": [{ "type": "command", "command": "memoryos hook stop" }]
      }
    ]
  }
}
```

When a save is due, `memoryos hook stop` prints:

```json
{
  "decision": "block",
  "reason": "Before stopping, call commit_facts for durable facts learned in this session. If there are none, stop."
}
```

`Stop` fires at the end of every turn, so the command limits itself to once every 10 turns per
session. `SessionEnd` cannot be used for this, because it has no decision control and cannot make the
agent act.

</details>

<details>
<summary><b>OpenCode setup</b></summary>

`.opencode/plugins/memoryos.ts` (v1 plugin API):

```ts
export const MemoryOS = async ({ $ }) => ({
  "experimental.session.compacting": async (_input, output) => {
    const memories = await $`memoryos hook session-start`.text();
    output.context.push(memories);
  },
});
```

This keeps memories through compaction but does not remind the agent to save, which needs a separate
`session.idle` handler. OpenCode v2 renames this hook to `ctx.session.hook("compaction", ...)`.

</details>

<details>
<summary><b>One-command install</b></summary>

Adds its entries, keeps your existing hooks, and registers the MCP server. The API key stays an
environment variable reference, never a value in the file:

```bash
memoryos hook install --agent claude-code
memoryos hook install --agent opencode
```

</details>

The `memoryos hook` commands read `MEMORYOS_URL` and `MEMORYOS_API_KEY`, and never fail the host: on
an error they log to stderr and exit 0. These setups have not been tested in a real Claude Code or
OpenCode session. References:
[Claude Code hooks](https://code.claude.com/docs/en/hooks),
[OpenCode plugins](https://opencode.ai/docs/plugins/),
[OpenCode v1 migration](https://opencode.ai/v2/docs/build/plugins/migrate-v1/).

## 🗺️ Roadmap

- [x] **Wave 0:** specification, core contracts, database schema, API contract
- [x] **Wave 1:** Postgres store, LLM and embedding clients, eval harness
- [x] **Wave 2:** ingest and worker, hybrid retrieval, agent runtime
- [ ] **Wave 3:** server, MCP, SDK, deployment
- [ ] **Wave 4:** integration, load test, restore drill, independent review

### Release targets

These are goals, not measured results:

| Gate | Target |
|---|---|
| ⚡ Recall latency | p95 under 200 ms at 1M facts, excluding embedding and LLM calls |
| 🕰️ Staleness | At least 95% of replaced facts absent from current recall |
| 🛡️ Poisoning | 0 poisoned facts in recall without approval |
| 🔒 Isolation | 0 cross-tenant reads |
| ♻️ Recovery | Jobs from a killed worker complete after the lease expires |

## 📚 Documentation

| Document | Contents |
|---|---|
| [`docs/runbook.md`](docs/runbook.md) | Install, upgrades, backup and restore, keys, monitoring, incidents |
| [`docs/SPEC.md`](docs/SPEC.md) | Behavior and contracts |
| [`docs/PLAN.md`](docs/PLAN.md) | Development waves and ownership |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Status, with proof for each task |
| [`docs/openapi.yaml`](docs/openapi.yaml) | REST API contract |
| [`docs/adr/`](docs/adr/) | Architecture decisions |

## 🧪 Development

Requires Rust 1.92+ and Docker. Integration tests start real Postgres and WireMock containers. Do not
use `cargo test --all-features`, which downloads the embedding model. The test feature list is in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --features memoryos-store/integration,memoryos-llm/integration,memoryos-ingest/integration,memoryos-retrieve/integration,memoryos-agent/integration,memoryos-eval/integration,memoryos-server/integration
cargo deny check
```

## 📄 License

Apache-2.0, as declared in `Cargo.toml`.
