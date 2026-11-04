# Summary

## What was built

- A RAG question answering system over a pinned snapshot of the Model Context Protocol docs
  (37 MDX files, commit `3c4d91b`), with MDX cleaning and chunk metadata (path, heading path,
  chunk id, character offsets).
- Two chunkers (fixed-size with overlap, heading-aware markdown), four retrievers (hand-written
  BM25, dense LSA, dense MiniLM, hybrid with Reciprocal Rank Fusion) and an optional cross-encoder
  reranker: 16 configurations built through one function.
- Grounded answering with inline `[chunk_id]` citations and a "not in the docs" refusal decided
  from IDF-weighted query term coverage.
- An evaluation harness: a 60-question gold set (10 unanswerable) labelled with verbatim evidence
  quotes, metrics (recall@1/5/10, MRR, nDCG@10, citation accuracy, refusal accuracy,
  faithfulness), bootstrap confidence intervals, paired comparisons, a refusal threshold sweep,
  and a written analysis in `evals/results.md`.
- A CI regression gate that fails the build if any configuration's recall@5 drops more than
  3 points below `evals/baseline.json`.
- A FastAPI service (`/ask`, `/search`) with a small HTML page that shows the answer, linked
  citations and the retrieved chunks with scores; a Dockerfile; CI running lint, mypy, tests,
  evals and a container smoke test.
- Documentation: README, LEARNING.md (study guide), five ADRs, DECISIONS.md, SPEC.md, PLAN.md.

## Mocked vs real

| Part | Status |
| --- | --- |
| Corpus, cleaning, chunking | Real |
| BM25, LSA, MiniLM embeddings, RRF, cross-encoder reranking | Real (local models, CPU) |
| Retrieval metrics and the regression gate | Real, identical in mock and live mode |
| Refusal decision | Real (computed before any LLM call) |
| Answer generation | Mock by default: extractive, quotes the best-matching sentences with citations. Live Anthropic provider when `ANTHROPIC_API_KEY` is set |
| Faithfulness | Mock: token-overlap heuristic (100% by construction for extractive answers). Live: LLM-as-judge with structured JSON output |
| Live provider | Implemented and unit-tested with a fake client; never called against the real API (no key available) |
| "Recorded fixtures" | Golden snapshots of mock outputs (`tests/fixtures/mock_answers.json`), not live recordings |

## Known gaps

- The gold set was written by the builder from the documents (lexical bias), is small (one
  question is 2 points of recall) and has one labelled source per question, so recall is a lower
  bound. No human domain expert reviewed it.
- The refusal threshold (0.35) was tuned on the evaluation set (93% refusal accuracy there). On a
  held-out set of on-topic unanswerable questions it misses 8 of 10 (55%): the heuristic measures
  topicality, not answerability.
- No live-mode numbers exist for answer quality or faithfulness.
- The Docker image was not built locally (Docker Hub rate limit); CI builds and smoke-tests it.
- Embeddings are recomputed at every startup; reranking costs about 0.5 to 1 second per query on
  a CPU.

## Resume bullets (honest claims)

- Built a retrieval-augmented QA service (Python, FastAPI) over the MCP documentation with BM25,
  dense (MiniLM, LSA), hybrid Reciprocal Rank Fusion and cross-encoder reranking, serving grounded
  answers with inline chunk citations and a "not in the docs" refusal path.
- Designed an evaluation harness that compares 16 chunking x retriever x reranker configurations
  on a 60-question gold set with recall@k, MRR, nDCG, citation and refusal accuracy, using paired
  bootstrap confidence intervals to separate real gains (reranking: +6 to +16 points recall@5)
  from noise.
- Made retrieval quality a CI gate: GitHub Actions runs lint, mypy, 60+ tests and the full
  ablation offline with a deterministic LLM mock, and fails the build if any configuration's
  recall@5 drops more than 3 points below the committed baseline.
- Labelled the gold set with verbatim evidence quotes mapped to chunks through character offsets,
  so one gold file evaluates any chunking strategy fairly, and documented its biases and
  limitations alongside the results.
