# Load test results

Date: 2026-11-06. Stopped early by the user's decision, because 1M facts is a stress case for one
agent's memory. **The 200 ms gate was not met in what was measured, and it was not fully measured.**

Setup: 1M facts in one scope (`user:big`), pgvector:pg17 in Docker on an M1 Pro, in a 3.8 GiB Docker
VM with `shared_buffers=1GB`. The vector index is 1953 MB and the table is 1802 MB. Other builds ran on
the machine at the same time, so these numbers may read high. Embedding time is excluded.

| Query | Clients | p95 before | p95 after text cap |
|---|---|---:|---:|
| Text only, common word | 1 | 1770 ms | 6.6 ms |
| Recall, common word | 1 | 1868 ms | 385 ms |
| Recall, common word | 8 | 2106 ms | 466 ms |
| Recall, common word | 32 | 7449 ms | 1231 ms |
| Recall, rare word | 1 | 227 ms | not re-measured |
| Recall with `as_of`, common word | 1 | not read | 440 ms |
| Recall with `as_of`, common word | 32 | not read | 8566 ms |

Vector search only, one client, by `hnsw.ef_search`, with a further manual check on 12 queries:

| ef_search | Median | Median disk reads | Top-50 overlap with ef 1000 |
|---:|---:|---:|---:|
| 50 | 348 ms | 1720 | 0.80 |
| 100 | 232 ms | 1825 | 0.80 |
| 200 (current) | 346 ms | 1522 | 0.87 |
| 1000 | 2337 ms | 9724 | 1.00 |

The disk reads suggest the index does not fit in cache on this VM. This is a hypothesis: more memory was
not tried. Lowering `ef_search` lost quality with no clear speed gain, so it stays at 200.

Not measured: 1000 small scopes, the hub entity, results on a quiet machine, and results with more memory.

The text cap ranks only the newest 1000 matches, so older matches of a very common word can no longer
come from text search. Reproduce with `deploy/loadtest`.
