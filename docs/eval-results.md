# Real-host results

Date: 2026-11-06. Commit under test: `89e81e0`. Task: T9b.

**Status: not completed. Nothing here counts as a pass, and it says nothing about the staleness and
poisoning gates.**

## Real agent hosts

Server: real local embedder, compose project `memoryos-t9b-host`, image built from `89e81e0`.

### Claude Code 2.1.319

The MCP server, hooks and tool descriptions were **not tested through Claude Code**.

- `memoryos hook install --agent claude-code --project` wrote `.claude/settings.json` and `.mcp.json`,
  kept the key as `${MEMORYOS_API_KEY}`, and a second run reported `unchanged`.
- With `CLAUDE_CONFIG_DIR` inside the temp dir: `Not logged in · Please run /login`. Credentials were
  not copied in.
- With the normal login, the `claude -p` run was denied by the permission classifier of the session
  running this task, so it never started. No SessionStart, Stop, MCP or tool-description result exists.

### OpenCode 2.2.5

Project-scoped install, with config, data and cache directories inside the temp dir.

| Check | Result |
|---|---|
| MCP connection | Works. Log: `mcp connected server=memoryos tools=6` |
| `opencode.json` format | Accepted, and normalized to `mcp.servers.memoryos` by `opencode debug config` |
| `opencode mcp list` | Prints `No MCP servers configured` while the same config connects at run time. Also printed for a hand-written v2 shape, so this is not specific to the installed file. Cause not traced |
| **Installed plugin** | **Fails to load.** See below |
| Model turn | Not reached: the account behind OpenCode's own model had no credit. No tool was called by a model |

The plugin error, from `.opencode/plugins/memoryos.ts`:

```
failed to load plugin ... PluginModule.LoadError: Plugin must export a default definition with an id
and an effect or setup function. (cause: SchemaError(Missing key at ["default"]))
```

`hook install` writes the v1 plugin form, and OpenCode 2.2.5 rejects it. A load failure is only logged
and the session carries on, so MCP still connects, but the plugin's compaction hook cannot run. That
follows from the load failure; a compaction was not run to observe it.

### Pi

Untested, out of scope.

### Hook commands, run by hand

No host consumed this output, so whether a real host accepts `hook stop` output is **still unverified**.

| Check | Result |
|---|---|
| `hook stop` with `MEMORYOS_HOOK_STOP_EVERY=1` | Valid JSON with keys `decision` and `reason` |
| `hook stop` with `stop_hook_active` true | 0 bytes |
| `hook stop` at the default interval, 11 calls | 144 bytes on call 10, 0 on the others |
| `hook session-start` | Prints the memories block, and includes a fact committed earlier |
| Either hook with the server down | One line on stderr, exit 0 |

### Server behaviour over REST, using curl

This is not a loop through a model.

| Write | Result |
|---|---|
| Commit to `agent:coding-agent` | `active`, trust `agent`, returned by recall |
| Commit to `user:alice` | `quarantined`, trust `agent`, absent from recall, listed in `/v1/quarantine` |

### Readiness after the outage

The container was unhealthy because of the outage, not a server fault. Four model load attempts failed
with network errors: a TLS close at 16:39 UTC, then DNS failures. It had 0 restarts and no OOM kill. The
retry that began at 18:22 UTC reached `ready` at 18:42 UTC with no intervention, and health went to
`healthy`. The download took about 20 minutes at about 25 MB/min.

`/readyz` kept reporting `"embedder":"failed"` during that successful retry, while logs said `the first
embed call has not returned, the model may still be downloading`. Timings from inside the outage are
not used here.
