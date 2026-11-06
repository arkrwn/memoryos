# ADR 0005: Host mode is the only mode

Status: accepted, chosen by the user

The coding agent's own model decides what to remember and sends structured facts over MCP. MemoryOS
never calls an LLM and needs no provider key, and any MCP-capable agent works. It never calls the host
either. Costs: nothing is processed while the host is closed, quality depends on the host model, and
because that model also reads untrusted content, host writes use `agent` trust and are quarantined
outside `agent:*` scopes. An earlier design also had a server-side extraction worker and agent runtime.
They were removed: they stay in git history and were never proven against a real model.
