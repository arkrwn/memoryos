#!/bin/sh
# Runs MemoryOS natively on macOS: its own Postgres 18 + pgvector cluster and `memoryos serve`, both
# kept running by launchd agents. Run through `make setup`, which installs the binary first.
#
# Safe to run again: an existing cluster, database and server.env are kept, and the agents are
# rewritten and reloaded. On the first run, data from a Docker install (`memoryos setup`) is copied
# over when its Postgres container exists, and the Docker server container is stopped. Docker volumes
# are left alone.
set -eu

PG_PORT=${PG_PORT:-6432}
SERVER_ADDR=${SERVER_ADDR:-127.0.0.1:8080}
CONFIG_DIR=${CONFIG_DIR:-$HOME/.config/memoryos}
DATA_DIR=${DATA_DIR:-$HOME/.local/share/memoryos}
AGENT_PREFIX=${AGENT_PREFIX:-dev.memoryos}
DOCKER_PG=${DOCKER_PG:-memoryos-postgres-1}
DOCKER_SERVER=${DOCKER_SERVER:-memoryos-memoryos-1}
BIN=${BIN:-$HOME/.cargo/bin/memoryos}

PGDATA=$DATA_DIR/postgres
ENV_FILE=$CONFIG_DIR/server.env
AGENTS=$HOME/Library/LaunchAgents
DOMAIN=gui/$(id -u)
PG_AGENT=$AGENT_PREFIX.postgres
SERVER_AGENT=$AGENT_PREFIX.server

say() { printf '==> %s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || fail "this setup uses Homebrew and launchd, so it runs on macOS only"
command -v brew >/dev/null || fail "Homebrew is missing, see https://brew.sh"
[ -x "$BIN" ] || fail "$BIN is missing, run \`make install\` first"

has_docker_pg() {
	command -v docker >/dev/null && docker inspect "$DOCKER_PG" >/dev/null 2>&1
}

# --- Postgres 18 and pgvector -------------------------------------------------------------------
for formula in postgresql@18 pgvector; do
	brew list --versions "$formula" >/dev/null || { say "installing $formula"; brew install "$formula"; }
done
PGBIN=$(brew --prefix postgresql@18)/bin

# --- Password: kept from server.env, else taken from Docker, else generated -----------------------
password=
if [ -f "$ENV_FILE" ]; then
	password=$(sed -n 's|^DATABASE_URL=postgres://[^:]*:\([^@]*\)@.*|\1|p' "$ENV_FILE" |
		python3 -c 'import sys, urllib.parse; print(urllib.parse.unquote(sys.stdin.read().strip()))')
	[ -n "$password" ] || fail "$ENV_FILE has no DATABASE_URL with a password"
elif has_docker_pg; then
	password=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$DOCKER_PG" |
		sed -n 's/^POSTGRES_PASSWORD=//p')
fi
[ -n "$password" ] || password=$(openssl rand -hex 24)

# --- Cluster ------------------------------------------------------------------------------------
mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$AGENTS"
chmod 700 "$DATA_DIR" "$CONFIG_DIR"
if [ ! -f "$PGDATA/PG_VERSION" ]; then
	say "creating the Postgres cluster in $PGDATA"
	pwfile=$(mktemp)
	chmod 600 "$pwfile"
	printf '%s\n' "$password" >"$pwfile"
	"$PGBIN/initdb" -D "$PGDATA" -U memoryos --pwfile="$pwfile" --auth-local=scram-sha-256 \
		--auth-host=scram-sha-256 -E UTF8 --locale=C >/dev/null
	rm -f "$pwfile"
	cat >>"$PGDATA/postgresql.conf" <<EOF

# MemoryOS native cluster, written by deploy/native/setup.sh.
port = $PG_PORT
listen_addresses = 'localhost'
unix_socket_directories = '$PGDATA'
EOF
fi
PG_PORT=$(sed -n 's/^port = \([0-9]*\).*/\1/p' "$PGDATA/postgresql.conf" | tail -1)
[ -n "$PG_PORT" ] || fail "$PGDATA/postgresql.conf sets no port"

# --- server.env ---------------------------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
	say "writing $ENV_FILE"
	encoded=$(printf '%s' "$password" | python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read(), safe=""))')
	umask 077
	cat >"$ENV_FILE" <<EOF
DATABASE_URL=postgres://memoryos:$encoded@localhost:$PG_PORT/memoryos
MEMORYOS_BIND=$SERVER_ADDR
RUST_LOG=info,opentelemetry=warn,opentelemetry_sdk=warn,hyper=warn
EOF
	umask 022
fi
chmod 600 "$ENV_FILE"

# --- launchd agents -----------------------------------------------------------------------------
# Postgres exits with "postmaster became multithreaded during startup" when launchd starts it with
# no locale, so the agent sets LC_ALL.
cat >"$AGENTS/$PG_AGENT.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$PG_AGENT</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PGBIN/postgres</string>
        <string>-D</string><string>$PGDATA</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>LC_ALL</key><string>en_US.UTF-8</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$DATA_DIR/postgres.log</string>
    <key>StandardErrorPath</key><string>$DATA_DIR/postgres.log</string>
</dict>
</plist>
EOF

cat >"$AGENTS/$SERVER_AGENT.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$SERVER_AGENT</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string><string>-c</string>
        <string>set -a; . '$ENV_FILE'; set +a; until '$PGBIN/pg_isready' -q -h localhost -p $PG_PORT; do sleep 1; done; exec '$BIN' serve</string>
    </array>
    <key>WorkingDirectory</key><string>$DATA_DIR</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$DATA_DIR/server.log</string>
    <key>StandardErrorPath</key><string>$DATA_DIR/server.log</string>
</dict>
</plist>
EOF
plutil -lint "$AGENTS/$PG_AGENT.plist" "$AGENTS/$SERVER_AGENT.plist" >/dev/null

reload() {
	launchctl bootout "$DOMAIN/$1" 2>/dev/null || true
	launchctl bootstrap "$DOMAIN" "$AGENTS/$1.plist"
}

# The server is stopped first so nothing holds a connection while Postgres restarts.
launchctl bootout "$DOMAIN/$SERVER_AGENT" 2>/dev/null || true
pkill -f "$BIN serve" 2>/dev/null || true
launchctl bootout "$DOMAIN/$PG_AGENT" 2>/dev/null || true
"$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1 && "$PGBIN/pg_ctl" -D "$PGDATA" stop -m fast >/dev/null
say "starting Postgres on localhost:$PG_PORT"
reload "$PG_AGENT"
i=0
until "$PGBIN/pg_isready" -q -h localhost -p "$PG_PORT"; do
	i=$((i + 1))
	[ "$i" -lt 30 ] || fail "Postgres did not start, see $DATA_DIR/postgres.log"
	sleep 1
done

# --- Database, with the Docker data on the first run --------------------------------------------
export PGHOST=localhost PGPORT=$PG_PORT PGUSER=memoryos PGPASSWORD=$password
if [ -z "$("$PGBIN/psql" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = 'memoryos'")" ]; then
	"$PGBIN/psql" -d postgres -qc "CREATE DATABASE memoryos OWNER memoryos"
	if has_docker_pg; then
		say "copying the data from the Docker container $DOCKER_PG"
		docker start "$DOCKER_PG" >/dev/null
		docker stop "$DOCKER_SERVER" >/dev/null 2>&1 || true
		mkdir -p -m 700 "$DATA_DIR/backup"
		dump=$DATA_DIR/backup/memoryos-docker-$(date +%Y-%m-%d).dump
		docker exec "$DOCKER_PG" pg_dump -U memoryos -d memoryos -Fc >"$dump"
		chmod 600 "$dump"
		"$PGBIN/psql" -d postgres -qc "DO \$\$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'memoryos_app') THEN CREATE ROLE memoryos_app NOLOGIN; END IF; END \$\$"
		"$PGBIN/pg_restore" -d memoryos --exit-on-error "$dump"
		docker stop "$DOCKER_PG" >/dev/null
		say "copied $("$PGBIN/psql" -d memoryos -tAc 'SELECT count(*) FROM facts') facts, backup kept at $dump"
	fi
fi
unset PGPASSWORD
if command -v docker >/dev/null && docker inspect "$DOCKER_SERVER" >/dev/null 2>&1; then
	docker stop "$DOCKER_SERVER" >/dev/null 2>&1 || true
fi

say "applying migrations"
(set -a; . "$ENV_FILE"; set +a; "$BIN" migrate)

# --- Server -------------------------------------------------------------------------------------
say "starting the server on $SERVER_ADDR"
reload "$SERVER_AGENT"
addr=$(sed -n 's/^MEMORYOS_BIND=//p' "$ENV_FILE")
i=0
until curl -fs -o /dev/null "http://$addr/readyz"; do
	i=$((i + 1))
	[ "$i" -lt 120 ] || fail "the server was not ready after 2 minutes, see $DATA_DIR/server.log"
	sleep 1
done
say "MemoryOS is running on http://$addr"

if [ ! -f "$CONFIG_DIR/credentials.json" ]; then
	say "no $CONFIG_DIR/credentials.json yet: create the agent and admin keys with \`memoryos key create\`"
fi
