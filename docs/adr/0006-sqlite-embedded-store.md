# ADR 0006: SQLite + sqlite-vec as an embedded store

Status: accepted

MemoryOS supports PostgreSQL + pgvector and SQLite + sqlite-vec behind the same `MemoryStore`
contract. PostgreSQL remains the default and the choice for concurrent multi-instance deployments;
SQLite provides a single-process embedded deployment without a database server.

The SQLite implementation has its own schema and queries rather than translating PostgreSQL SQL at
runtime. It uses one connection to serialize writers, explicit tenant predicates and tenant-aware
foreign keys instead of row-level security, FTS5 for text, and exact cosine scans over sqlite-vec
embeddings. Exact scans preserve result counts under selective filters but are not an HNSW
performance equivalent. Both backends share the same 384-dimensional model and domain contract.

sqlite-vec 0.1.9 is used unmodified under its MIT/Apache-2.0 license and statically linked, so
startup does not depend on a writable extension path or SQLite's dynamic loader. On Unix, newly
created parent directories use mode 0700, the database uses 0600, and existing database/WAL/SHM
permissions are restricted at startup without changing the permissions of existing parent
directories. FTS ranking uses per-document match counts rather than corpus-wide BM25 statistics,
which would let one tenant's documents affect another tenant's scores.

`MEMORYOS_STORE=sqlite` and a `sqlite:` `DATABASE_URL` select the backend. There is no automatic
PostgreSQL-to-SQLite data migration; switching back to the unchanged PostgreSQL database is the
deployment rollback path.

libsqlite3-sys 0.39 bundles SQLite 3.54, which includes the fix for CVE-2026-119375. SQLx 0.9.0
predates that binding, so the build pins SQLx's upstream compatibility commit until the next release;
cargo-deny permits only that repository as a git source. This raises the workspace minimum Rust
version to 1.94, and the CI toolchain and release image use a compatible newer compiler.
