# Rebuild and restart a local MemoryOS after pulling or changing code: `make` or `make upgrade`.
# The binary is copied into ~/.cargo/bin rather than symlinked, because a link into target/ would
# break on the `cargo clean` that ends the upgrade.

CONFIG_DIR ?= $(HOME)/.config/memoryos
COMPOSE := docker compose -f $(CONFIG_DIR)/docker-compose.yml

.PHONY: upgrade install server clean

upgrade: install server clean

# The `memoryos` command the hooks and the MCP proxy run.
install:
	cargo install --locked --force --path crates/server

# Rebuilds the server image from this checkout and waits until the new container is healthy.
server:
	$(COMPOSE) up -d --build --wait memoryos

clean:
	cargo clean
