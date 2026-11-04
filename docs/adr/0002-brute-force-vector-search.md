# ADR 0002: Brute-force numpy search instead of a vector database

- Status: accepted
- Date: 2026-11-04

## Context

The corpus has about 600 to 800 chunks per strategy. Dense retrieval needs nearest-neighbour
search over their embeddings.

## Decision

Keep embeddings in one numpy matrix with unit-length rows and score a query with a single matrix
product (`matrix @ query_vector`). BM25 uses a small in-memory inverted index written by hand.

## Consequences

- Search is exact, so evaluation numbers are not blurred by approximate-search recall loss.
- Search takes well under a millisecond per query at this size; no service to run or configure.
- BM25 and LSA are rebuilt at startup; MiniLM corpus embeddings are cached on disk by content
  hash (25 s cold, about 1 s warm). This stops scaling somewhere in the hundreds of thousands of
  chunks; at that point an ANN index (FAISS, pgvector) becomes worth its complexity.
