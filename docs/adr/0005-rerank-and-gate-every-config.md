# ADR 0005: Serve hybrid retrieval with reranking; gate recall@5 for every configuration

- Status: accepted
- Date: 2026-11-04

## Context

The ablation (16 configurations) shows a cross-encoder reranker adds 6 to 16 points of recall@5
to every first-stage retriever, while the top four reranked configurations are statistically tied.
Reranking costs about 0.5 to 1 second per query on a CPU.

## Decision

Serve `fixed/hybrid+rerank` by default (the highest recall@5, tied, and the cheapest of the tied
configurations to rerank because fixed chunks are shorter). Keep every configuration buildable
through the same `pipeline.build_retrievers` function, and gate CI on recall@5 for all of them:
the build fails if any configuration drops more than 3 points below `evals/baseline.json`.

## Consequences

- The latency budget is dominated by the reranker; a GPU or a smaller reranker depth would cut it.
- 3 points is 1.5 questions on a 50-question set: tight enough to catch a real regression, loose
  enough to absorb tiny floating-point differences between CPUs in the neural models.
- Intentional changes that move the numbers require updating the baseline in the same commit
  (`python -m evals.check_regression --update-baseline`), which makes the change visible in review.
