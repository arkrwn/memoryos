# MemoryOS runbook

Run commands from the repository root, with this helper:

```bash
dc() { docker compose -f deploy/docker-compose.yml "$@"; }
```

Every command here was run on Docker 29.8.2, Compose 5.6.0 and arm64 Linux, using throwaway project
names set through `COMPOSE_PROJECT_NAME`, unless it is marked **not run**. No real LLM provider or
production traffic was involved. Quoted outputs are from those runs.

## Install

Requirements:

| Resource | Need |
|---|---|
| Runtime memory | At least 1 GiB per container running the local embedder; 768m is OOM killed. Measured 980 MiB after load. The compose limit is `1536m` (`MEMORYOS_MEM_LIMIT`) |
| Disk | 144 MB image, 465 MiB model cache, plus the build cache and database |
| Network at build | `cdn.pyke.io`, for ONNX Runtime |
| Network at first start | Hugging Face, for the 481 MB model. Took about 2 minutes here |
| glibc | 2.38 or newer. `debian:trixie-slim`, `ubuntu:24.04` and `gcr.io/distroless/cc-debian13` work; bookworm and alpine do not |

```bash
cp deploy/.env.example deploy/.env
sed -i.bak "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 32)/" deploy/.env && rm deploy/.env.bak
dc up -d --build
until curl -sf http://localhost:8080/readyz; do sleep 5; done
```

The stack is `postgres` (not published on the host), `migrate` (one-shot) and `memoryos` (host mode,
published on `127.0.0.1:8080`). `/healthz`, `/readyz` and `/metrics` answer without a token, so
keep a TLS reverse proxy in front before setting `MEMORYOS_PUBLISH` to a public address.

Leave `HF_HOME` unset, because it overrides `MEMORYOS_MODEL_CACHE`.

### Worker mode

`deploy/.env` needs the provider keys, then:

```bash
dc up -d memoryos-worker
```

This starts `postgres`, `migrate` and the worker-mode server, without `memoryos`. Without a key the
server exits at startup and names the variable, for example `the openai embedder requires
OPENAI_API_KEY (or set MEMORYOS_EMBEDDER=local)`. `dc up -d memoryos-worker memoryos-job-worker` adds a
standalone job worker. It serves no HTTP, so its job counts never reach `/metrics`.

Pick one mode per database. Vectors from the local model and from OpenAI cannot be compared, and
changing embedder needs re-embedding, for which there is no tool yet.

### Database roles

The compose file logs in as the `POSTGRES_USER` superuser, so migrations can create the `vector`
extension and the `memoryos_app` role. To run with a non-superuser login, a superuser prepares first:

```sql
CREATE ROLE memoryos_app NOLOGIN;
CREATE ROLE memoryos_login LOGIN PASSWORD '...';
GRANT memoryos_app TO memoryos_login;
CREATE DATABASE memoryos OWNER memoryos_login;
\c memoryos
CREATE EXTENSION vector;
```

Then point `DATABASE_URL` at that login and run `memoryos migrate`. What the drill showed:

| Situation | Result |
|---|---|
| Extension not prepared | `permission denied to create extension "vector"` |
| Login without `GRANT memoryos_app` | Requests return 500, and logs say `permission denied to set role "memoryos_app"` |
| `pg_dump` as that login | Fails: `query would be affected by row-level security policy for table "audit_events"`. Back up as a superuser |

> **Warning: isolation applies to the server's transactions, not to everyone with the login.** `facts`
> and `episodes` are `NO FORCE`, so the owner login, outside `memoryos_app`, reads every tenant. The
> drill returned both tenants' facts from plain `psql`. A superuser does too. Do not give this login
> to people or to reporting tools, and to query as one tenant use
> `SET ROLE memoryos_app; SET app.tenant = '...';`.

## Upgrades and migrations

Take a backup first, as described below. Then:

```bash
dc up -d --build
dc logs migrate
```

The first command rebuilds, reruns `migrate` and replaces the server only if migration succeeds. The
log should end with `migrations applied`. Rerunning on a current schema changes nothing.

Migrations only go forward. To roll back a release, restore the backup taken before it and start the
old image. **Not run** across a real schema change, since only one migration exists so far.

## Backup and restore

The model cache needs no backup. It downloads again. `deploy/.env` is not in the database, so keep it
safely. API keys are in the database as hashes, and existing tokens keep working after a restore.

```bash
dc exec -T postgres pg_dump -U memoryos -d memoryos -Fc > memoryos.dump
```

Restore into a fresh, empty database, here a second project beside the old one:

```bash
export COMPOSE_PROJECT_NAME=memoryos-restore
dc up -d postgres
until dc exec -T postgres pg_isready -U memoryos -d memoryos; do sleep 1; done
dc exec -T postgres psql -U memoryos -d memoryos -c "CREATE ROLE memoryos_app NOLOGIN;"
dc exec -T postgres pg_restore -U memoryos -d memoryos --exit-on-error < memoryos.dump
dc up -d
until curl -sf http://localhost:8080/readyz; do sleep 5; done
```

The order matters:

- `pg_dump` does not carry roles, and the dump grants to `memoryos_app`, so create it first.
- Restore before `dc up -d`, because `migrate` on an empty database would create the tables the
  restore then collides with.
- Stop the old project's `memoryos` first, or set `MEMORYOS_PUBLISH` to another port.

Check that the row counts match the source, and that row-level security survived:

```bash
dc exec -T postgres psql -U memoryos -d memoryos -Atc "SELECT (SELECT count(*) FROM facts), (SELECT count(*) FROM episodes), (SELECT count(*) FROM api_keys);"
dc exec -T postgres psql -U memoryos -d memoryos -Atc "SELECT count(*) FROM pg_policies;"
```

The drill gave 9 policies, the same counts on both sides, `recall` returned both facts with the
pre-backup token, and a key in another tenant got `{"facts":[],"pending_episodes":[]}`.

To restore in place instead of beside (**not run**, and it deletes the current database):

```bash
dc down
docker volume rm memoryos_pgdata
```

Then continue from `dc up -d postgres` above, without setting `COMPOSE_PROJECT_NAME`.

## API keys

```bash
TOKEN=$(dc exec -T memoryos memoryos key create --tenant acme --name laptop --scopes 'user:*,agent:*' --max-trust agent --admin)
```

The token goes to stdout once, and the key id goes to stderr. Only the hash is stored.

| Flag | Choice |
|---|---|
| `--scopes` | `*`, `user:*`, or an exact scope like `agent:a1`. Give the narrowest set that works |
| `--max-trust` | `external`, `agent`, `user` or `system`. Agents get `agent`, which is also the most a commit can write |
| `--admin` | Quarantine review, purge and user deletion, within the key's scopes. Keep separate from agent keys |

There is no rotate command. Create the new key, move clients to it, then revoke the old one:

```bash
dc exec -T postgres psql -U memoryos -d memoryos -c "SELECT id, tenant, name, scopes, max_trust, is_admin, revoked_at FROM api_keys ORDER BY created_at;"
dc exec -T memoryos memoryos key revoke <id>
```

Revoking takes effect on the next request: the drill went from 200 to 401 without a restart.

## Monitoring

| Signal | Watch for |
|---|---|
| `GET /readyz` | 503. The body names `database` (`ok`, `unavailable`) and `embedder` (`not_required`, `loading`, `ready`, `failed`) |
| `GET /healthz` | Failure means the process is gone. It stays 200 through database and model failures |
| `memoryos_http_requests_total{status=~"5.."}` | 500 is usually the database, and 502 is the model or a provider |
| `memoryos_http_request_duration_seconds` | Latency by route |
| `memoryos_recall_duration_seconds` | Target is p95 under 200 ms, not yet measured at scale |
| `memoryos_jobs_processed_total{result="failed"}` | Worker mode, embedded worker only. A rising count means extraction fails |
| Logs, JSON on stdout | `not ready`, `retrying provider request`, `query embedding timed out` |
| Container memory | Near `MEMORYOS_MEM_LIMIT`. OOM shows as a restart with exit code 137 |

Job backlog, worker mode:

```bash
dc exec -T postgres psql -U memoryos -d memoryos -Atc "SELECT status, attempts, run_after > now() AS waiting, left(last_error,120) FROM jobs;"
```

Quarantine backlog is `GET /v1/quarantine`, per scope, below. Traces export only when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set (**not run**).

## Quarantine

An admin key reviews facts that the trust gate held back. Each scope is named exactly in the query.
Host commits to `user:*` or `shared:*` always land here.

```bash
curl -sS -H "Authorization: Bearer $TOKEN" "http://localhost:8080/v1/quarantine?scopes=user:alice"
curl -sS -X POST -H "Authorization: Bearer $TOKEN" "http://localhost:8080/v1/quarantine/$FACT_ID/approve"
curl -sS -X POST -H "Authorization: Bearer $TOKEN" "http://localhost:8080/v1/quarantine/$FACT_ID/reject"
```

- Quarantined and rejected facts never appear in recall. The drill returned no facts for
  `user:alice` before approval, and the fact after.
- An approval can answer `invalid_input` when the chain moved to a higher-trust fact since the
  commit (`fact.approve_held` in `audit_events`). Review that fact, then approve again. **Not run**
  over HTTP; covered by integration tests.
- To remove a poisoned source and everything derived only from it, where a fact it replaced is
  restored:

```bash
curl -sS -X POST -H "Authorization: Bearer $TOKEN" "http://localhost:8080/v1/episodes/$EPISODE_ID/purge"
```

- The trail is in `audit_events`:

```bash
dc exec -T postgres psql -U memoryos -d memoryos -c "SELECT created_at, tenant, actor, action, target_id FROM audit_events ORDER BY created_at DESC LIMIT 20;"
```

## Incidents

### Provider down (worker mode)

Symptoms: `memoryos_jobs_processed_total{result="failed"}` rises, and logs show `retrying provider
request`. Host mode has no LLM provider, so this does not apply there.

What keeps working, confirmed with base URLs pointed at a closed port:

- `POST /v1/episodes` still answers 202.
- `recall` still works, falling back to text search with the warning `query embedding timed out, using
  text search only`. The new episode shows under `pending_episodes` when its trust is `user` or
  `system`.
- Jobs retry with backoff from 30s to a 1h cap, and go `dead` after 5 attempts.

Steps:

1. Check the provider's status and the keys. Provider errors in logs are cut at the first colon, so
   they may not say why.
2. Look at the job states, using the query under Monitoring.
3. When the provider is back, requeue what went dead:

```bash
dc exec -T postgres psql -U memoryos -d memoryos -c "UPDATE jobs SET status = 'pending', attempts = 0, run_after = now(), locked_by = NULL, locked_until = NULL WHERE status = 'dead';"
```

The worker picks these up on its own, with no restart.

### Database down

Symptoms, from stopping Postgres under a running server: `/healthz` 200, `/readyz` 503 with
`"database":"unavailable"`, and API calls 500 with `internal`, within 10 ms.

1. `dc ps` and `dc logs postgres`. Check the disk first, since a full volume stops Postgres.
2. `dc start postgres`.
3. The server reconnects on its own, with no restart. The drill was ready again at once.

If the data is damaged, restore from backup, above. Writes during the outage were rejected, not
queued, so clients must retry them.

### Model download stall

Symptoms: `/readyz` 503 with `"embedder":"loading"` or `"failed"`, `/healthz` 200, and logs repeating
`not ready`. A healthy first start took about 2 minutes here; the container counts as `health:
starting` for 15 minutes.

What happens while the model is missing, confirmed on a network with no route out:

| Call | Result |
|---|---|
| `POST /v1/commits` | 502 `provider_error`, after about 3 seconds |
| `POST /v1/recall` | 200, text search only |
| Log | `local embedder model load failed: Failed to retrieve model file 'onnx/model.onnx': ... Temporary failure in name resolution` |

A failure retries on its own, with backoff up to 5 minutes. The drill covered a failing download, not
one that hangs. A hang keeps `loading` and logs a warning every 60 seconds (unit test only).

1. `dc logs memoryos`, looking for `not ready`.
2. Check DNS and the route to Hugging Face from the Docker network, and free space in the volume.
3. `dc restart memoryos` to retry now instead of waiting. **Not run** on a stalled download.
4. If the cache is corrupt (**not run**):

```bash
dc stop memoryos
docker volume rm memoryos_model-cache
dc up -d memoryos
```

For a host with no internet, fill the `model-cache` volume from another machine (**not run**).

## Docker disk growth

```bash
docker system df
docker system df -v
docker volume ls --filter dangling=true
```

| Source | Note |
|---|---|
| Build cache | Largest. A clean image build used about 3 GB here |
| Old images | Each `--build` leaves the previous `memoryos:local` untagged |
| `pgdata` | Normal growth. Closed facts are kept on purpose, so this does not shrink by itself |
| `model-cache` | Fixed at 465 MiB |
| Test containers | CI and `cargo test` start Postgres on tmpfs, so they should leave no volume. Check with the `dangling` listing |

Cleanup (**not run**, because they affect everything on the Docker host, not only MemoryOS, so read what they list first):

```bash
docker image prune
docker builder prune --keep-storage 5GB
```

Never use `docker volume prune` or `dc down -v` on a production host without a checked backup: they
delete `pgdata`.

## Known limits

- The full list is in `docs/SPEC.md` section 13. The ones that matter most in operation are below.
- The rate limiter and the agent session lock are per process, so with several instances the limit
  multiplies, and clients must not run concurrent turns on one session.
- A standalone `memoryos worker` serves no HTTP, so it has no metrics.
- `LocalEmbedder` silently truncates input over 512 tokens.
- Changing embedder needs re-embedding, and there is no tool for it.
- The compose stack is one Postgres and one server. There is no replication or scheduled backup; add
  those yourself.
- MCP has been checked by protocol requests and tests, not by a real agent host.
- No live LLM provider has been used in any test.
- Worker mode is proven to start, take episodes and survive provider failure. Extraction against a
  real provider has not run.
- The image takes prebuilt ONNX Runtime from `cdn.pyke.io` at build time. `cargo deny` does not cover it.
- The release targets in `README.md` are not yet measured.
