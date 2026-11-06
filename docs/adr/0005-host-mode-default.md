# ADR 0005: Host mode is the default

Status: accepted, chosen by the user

The coding agent's own model decides what to remember and sends structured facts over MCP, so no LLM
key is needed and any MCP-capable agent works. MemoryOS never calls the host. Costs: nothing is
processed while the host is closed, quality depends on the host model, and because that model also
reads untrusted content, host writes use `agent` trust and are quarantined outside `agent:*` scopes.
Worker mode remains for servers.
