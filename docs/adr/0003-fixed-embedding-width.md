# ADR 0003: Fixed embedding width of 384

Status: accepted, revised for host mode

HNSW indexes need a fixed `vector(n)` column. Host mode needs a local model and no API key, and
`multilingual-e5-small` produces 384 dimensions. `text-embedding-3-small` can be shortened to 384
through its `dimensions` parameter, so worker mode shares one schema. Shortening may lower OpenAI
search quality compared with 1536. Vectors from different models are not comparable, so changing
embedder on existing data needs re-embedding. Clients validate the width and return an error
instead of storing a wrong-sized vector.
