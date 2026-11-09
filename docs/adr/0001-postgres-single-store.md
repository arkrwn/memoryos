# ADR 0001: Postgres + pgvector as the only store

Status: superseded by ADR 0006

Facts must change atomically across text, vector and graph views, and they need tenant isolation.
One Postgres database gives transactions, row-level security, full-text search, HNSW vector search
and recursive graph queries. The cost is a required running database and no embedded mode.
