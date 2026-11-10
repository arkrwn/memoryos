# MemoryOS on macOS, native: Homebrew Postgres 18 + pgvector and `memoryos serve`.
#
#   make setup              once per machine: Postgres, the database and the binary
#   make import-docker      once, to copy the data of a Docker install (Docker must be running)
#   make start              start Postgres and the server, also after a reboot
#   make / make upgrade     after pulling new code
#
# Postgres listens on localhost only and trusts local connections, so no password is needed.

PG := $(shell brew --prefix postgresql@18 2>/dev/null)/bin
DIR := $(HOME)/.local/share/memoryos
PGDATA := $(DIR)/postgres
PORT ?= 6432
BIN := $(HOME)/.cargo/bin/memoryos
PSQL_ARGS := -h localhost -p $(PORT) -U memoryos

export DATABASE_URL := postgres://memoryos@localhost:$(PORT)/memoryos
export MEMORYOS_BIND ?= 127.0.0.1:8080
export RUST_LOG ?= info,opentelemetry=warn,opentelemetry_sdk=warn,hyper=warn

.PHONY: upgrade setup import-docker start start-db install

upgrade: install start-db
	$(BIN) migrate
	$(MAKE) start
	cargo clean

setup: install
	brew install postgresql@18 pgvector
	mkdir -p $(DIR)
	$(PG)/initdb -D $(PGDATA) -U memoryos --auth=trust -E UTF8 --locale=C
	$(MAKE) start-db
	$(PG)/createdb $(PSQL_ARGS) memoryos
	$(BIN) migrate
	cargo clean

# Replaces the native database with the Docker one, so run it before `make start`.
import-docker: start-db
	-docker stop memoryos-memoryos-1
	docker start memoryos-postgres-1
	until docker exec memoryos-postgres-1 pg_isready -q -U memoryos; do sleep 1; done
	$(PG)/dropdb $(PSQL_ARGS) --if-exists memoryos
	$(PG)/createdb $(PSQL_ARGS) memoryos
	$(BIN) migrate
	docker exec memoryos-postgres-1 pg_dump -U memoryos -Fc --data-only --exclude-table-data=_sqlx_migrations memoryos | \
		$(PG)/pg_restore $(PSQL_ARGS) -d memoryos --data-only --disable-triggers --exit-on-error
	docker stop memoryos-postgres-1
	$(PG)/psql $(PSQL_ARGS) -d memoryos -tAc "SELECT count(*) || ' facts copied' FROM facts"

start-db:
	$(PG)/pg_ctl -D $(PGDATA) status >/dev/null || $(PG)/pg_ctl -D $(PGDATA) -o "-p $(PORT)" -l $(DIR)/postgres.log start

# Starts the server, or restarts it when it is already running.
start: start-db
	-pkill -f "$(BIN) serve"
	nohup $(BIN) serve >> $(DIR)/server.log 2>&1 &
	until curl -fs -o /dev/null http://$(MEMORYOS_BIND)/readyz; do sleep 1; done
	@echo "memoryos is running on http://$(MEMORYOS_BIND)"

# The `memoryos` command the hooks, the MCP proxy and the server run. Copied, not symlinked, because
# a link into target/ would break on `cargo clean`.
install:
	cargo install --locked --force --path crates/server
