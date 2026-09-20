# ADR 0002: Local Qdrant and dense-only retrieval for v1

Status: accepted

## Decision

Persist one dense-vector collection in embedded, on-disk Qdrant. Keep retrieval behind the `Retriever` port.

## Consequences

Operators need no vector-database service. Concurrent CLI access is unsupported. BM25, hybrid, ensemble, and reranking remain later adapters rather than v1 behavior.

