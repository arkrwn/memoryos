# ADR 0003: Fixed embedding width of 384

Status: accepted

HNSW indexes need a fixed `vector(n)` column, and the SQLite `vec0` tables likewise declare a fixed
width. MemoryOS embeds on the machine with `multilingual-e5-small`, which needs no API key and
produces 384 dimensions. Vectors from different models are not comparable, so changing the model on
existing data needs re-embedding. The embedder validates the width and returns an error instead of
storing a wrong-sized vector.
