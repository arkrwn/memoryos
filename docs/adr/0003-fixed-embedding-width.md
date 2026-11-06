# ADR 0003: Fixed embedding width of 1536

Status: accepted

HNSW indexes need a fixed `vector(n)` column. v1 fixes 1536, matching `text-embedding-3-small`.
Models of another width need a migration and re-embedding. Clients validate the width and return
an error instead of storing a wrong-sized vector.
