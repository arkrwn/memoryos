# Run MemoryOS natively on macOS. On a new machine, or one still on Docker: `make setup`.
# After pulling or changing code: `make` or `make upgrade`.
#
# `setup` (deploy/native/setup.sh) creates a Postgres 18 + pgvector cluster in $(DATA_DIR)/postgres,
# writes $(CONFIG_DIR)/server.env, copies the data over from a Docker install when there is one, and
# installs the launchd agents that keep Postgres and `memoryos serve` running. Run it again any time;
# it keeps the existing data and settings.
#
# The binary is copied into ~/.cargo/bin rather than symlinked, because a link into target/ would
# break on the `cargo clean` that ends the upgrade.

CONFIG_DIR ?= $(HOME)/.config/memoryos
DATA_DIR ?= $(HOME)/.local/share/memoryos
ENV_FILE := $(CONFIG_DIR)/server.env
AGENT := gui/$(shell id -u)/dev.memoryos.server
BIN := $(HOME)/.cargo/bin/memoryos

export CONFIG_DIR DATA_DIR

.PHONY: setup upgrade install check migrate server clean

setup: install
	deploy/native/setup.sh
	cargo clean

upgrade: check install migrate server clean

# The `memoryos` command the hooks, the MCP proxy and the server run.
install:
	cargo install --locked --force --path crates/server

check:
	@test -f $(ENV_FILE) || { echo "$(ENV_FILE) is missing: run \`make setup\` first"; exit 1; }
	@launchctl print $(AGENT) >/dev/null 2>&1 || { echo "the launchd agent $(AGENT) is not loaded: run \`make setup\` first"; exit 1; }

# `memoryos serve` does not migrate, so new migrations are applied before the restart.
migrate: check
	set -a; . $(ENV_FILE); set +a; $(BIN) migrate

# Restarts the server on the new binary and waits up to 60 s until it is ready.
server: check
	launchctl kickstart -k $(AGENT)
	@addr=$$(sed -n 's/^MEMORYOS_BIND=//p' $(ENV_FILE)); \
	for i in $$(seq 60); do \
		curl -fs -o /dev/null "http://$$addr/readyz" && echo "memoryos is ready on $$addr" && exit 0; \
		sleep 1; \
	done; \
	echo "memoryos did not become ready; see $(DATA_DIR)/server.log"; exit 1

clean:
	cargo clean
