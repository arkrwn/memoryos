# syntax=docker/dockerfile:1

# The binary links ONNX Runtime statically, which sets three requirements:
# - the build needs g++ (libstdc++) and network access to cdn.pyke.io;
# - the runtime needs glibc 2.38 or newer, so bookworm and alpine do not work;
# - build and runtime stay on the same Debian release.

FROM rust:1.92-slim-trixie AS chef
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef --version 0.1.78 --locked
ENV CARGO_PROFILE_RELEASE_STRIP=symbols
WORKDIR /app

# Reduces the workspace to its manifests, so source edits keep the dependency layer cached.
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS dependencies
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --locked --package memoryos-server --bin memoryos --recipe-path recipe.json

FROM dependencies AS builder
COPY . .
RUN cargo build --release --locked --package memoryos-server --bin memoryos

FROM debian:trixie-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# The cache directory exists in the image, so a named volume mounted there inherits its owner.
RUN useradd --system --uid 10001 --user-group --home-dir /var/lib/memoryos --shell /usr/sbin/nologin memoryos \
    && mkdir -p /var/lib/memoryos/models \
    && chown -R memoryos:memoryos /var/lib/memoryos
COPY --from=builder /app/target/release/memoryos /usr/local/bin/memoryos
COPY --chmod=0755 deploy/healthcheck.sh /usr/local/bin/memoryos-healthcheck
ENV MEMORYOS_MODEL_CACHE=/var/lib/memoryos/models
USER 10001:10001
EXPOSE 8080
# The first start downloads a 481 MB embedding model, and /readyz fails until it has loaded. A 571 MB
# reranking model downloads after it, without holding readiness back.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --start-interval=5s --retries=3 \
    CMD ["/usr/local/bin/memoryos-healthcheck"]
ENTRYPOINT ["memoryos"]
CMD ["serve"]
