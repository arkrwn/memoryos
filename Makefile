# Rebuild and restart a local MemoryOS after pulling or changing code: `make` or `make upgrade`.
# The binary is copied into ~/.cargo/bin rather than symlinked, because a link into target/ would
# break on the `cargo clean` that ends the upgrade.
#
# The server runs natively: `memoryos serve` with the settings in $(CONFIG_DIR)/server.env, under
# the launchd agent $(AGENT) when it is loaded, otherwise as a background process logging to
# $(LOG).

CONFIG_DIR ?= $(HOME)/.config/memoryos
DATA_DIR ?= $(HOME)/.local/share/memoryos
ENV_FILE := $(CONFIG_DIR)/server.env
LOG := $(DATA_DIR)/server.log
AGENT := gui/$(shell id -u)/dev.memoryos.server
BIN := $(HOME)/.cargo/bin/memoryos
READY_URL ?= http://127.0.0.1:8080/readyz

.PHONY: upgrade install migrate server clean

upgrade: install migrate server clean

# The `memoryos` command the hooks, the MCP proxy and the server run.
install:
	cargo install --locked --force --path crates/server

# `memoryos serve` does not migrate, so new migrations are applied before the restart.
migrate:
	set -a; . $(ENV_FILE); set +a; $(BIN) migrate

# Restarts the server on the new binary and waits up to 60 s until it is ready.
server:
	@if launchctl print $(AGENT) >/dev/null 2>&1; then \
		launchctl kickstart -k $(AGENT); \
	else \
		pkill -f "memoryos serve" || true; \
		sleep 1; \
		mkdir -p $(DATA_DIR); \
		( set -a; . $(ENV_FILE); set +a; nohup $(BIN) serve >>$(LOG) 2>&1 & ); \
	fi
	@for i in $$(seq 60); do \
		curl -fs -o /dev/null $(READY_URL) && echo "memoryos is ready" && exit 0; \
		sleep 1; \
	done; \
	echo "memoryos did not become ready; see $(LOG)"; exit 1

clean:
	cargo clean
