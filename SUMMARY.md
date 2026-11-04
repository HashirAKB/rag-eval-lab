# Summary

## What was built

- A RAG question answering system over a pinned snapshot of the Model Context Protocol docs
  (37 MDX files, commit `3c4d91b`), with MDX cleaning and chunk metadata (path, heading path,
  chunk id, character offsets).
- Two chunkers (fixed-size with overlap, heading-aware markdown), four retrievers (hand-written
  BM25, dense LSA, dense MiniLM, hybrid with Reciprocal Rank Fusion) and an optional cross-encoder
  reranker: 16 configurations built through one function.
- Grounded answering with inline `[chunk_id]` citations and a "not in the docs" refusal decided
  before generation from IDF-weighted word coverage and, for reranked configs, the cross-encoder's
  top score.
- An evaluation harness: a 60-question gold set (10 unanswerable) labelled with verbatim evidence
  quotes, metrics (recall@1/5/10, MRR, nDCG@10, citation accuracy, refusal accuracy,
  faithfulness), bootstrap confidence intervals, paired comparisons, a refusal threshold sweep,
  and a written analysis in `evals/results.md`.
- A CI regression gate that fails the build if any configuration's recall@5 drops more than
  3 points below `evals/baseline.json`.
- A FastAPI service (`/ask`, `/search`) with a small HTML page that shows the answer, linked
  citations and the retrieved chunks with scores; a Dockerfile; CI running lint, mypy, tests,
  evals and a container smoke test.
- Follow-up work, each measured with the same harness: an on-disk embedding cache (startup 25 s to
  about 1 s), contextual chunk headers (helps fixed-size chunks, not promoted), light stemming
  (rejected, kept as a patch), a reranker depth sweep (20 candidates is the knee), a 20-question
  held-out set that exposed the refusal heuristic, a cross-encoder refusal gate (held-out refusal
  accuracy 55% to 75%) and a label review queue for multi-source labels. Reports live in
  `evals/experiments/`.
- Documentation: README, LEARNING.md (study guide), five ADRs, DECISIONS.md, SPEC.md, PLAN.md.

## Mocked vs real

| Part | Status |
| --- | --- |
| Corpus, cleaning, chunking | Real |
| BM25, LSA, MiniLM embeddings, RRF, cross-encoder reranking | Real (local models, CPU) |
| Retrieval metrics and the regression gate | Real, identical in mock and live mode |
| Refusal decision | Real (word coverage and cross-encoder score, computed before any LLM call) |
| Answer generation | Mock by default: extractive, quotes the best-matching sentences with citations. Live Anthropic provider when `ANTHROPIC_API_KEY` is set |
| Faithfulness | Mock: token-overlap heuristic (100% by construction for extractive answers). Live: LLM-as-judge with structured JSON output |
| Live provider | Implemented and unit-tested with a fake client; never called against the real API (no key available) |
| "Recorded fixtures" | Golden snapshots of mock outputs (`tests/fixtures/mock_answers.json`), not live recordings |

## Known gaps

- The gold set was written by the builder from the documents (lexical bias), is small (one
  question is 2 points of recall) and has one labelled source per question, so recall is a lower
  bound. No human domain expert reviewed it.
- Refusal thresholds were tuned on the evaluation set (96.7% refusal accuracy there for the
  served config). On 20 held-out questions it is 75%; word coverage alone scored 55% there,
  because it measures topicality, not answerability. The held-out set has since informed a design
  change, so it is no longer an unbiased test.
- No live-mode numbers exist for answer quality or faithfulness.
- The Docker image was not built in the development sandbox (registry rate limit, then a
  TLS-intercepting proxy); the CI `docker` job builds it and smoke-tests `/search`.
- Reranking costs about 0.5 to 1 second per query on a CPU (MiniLM corpus embeddings are cached
  on disk, so startup takes about a second once warm).

## Resume bullets (honest claims)

- Built a retrieval-augmented QA service (Python, FastAPI) over the MCP documentation with BM25,
  dense (MiniLM, LSA), hybrid Reciprocal Rank Fusion and cross-encoder reranking, serving grounded
  answers with inline chunk citations and a "not in the docs" refusal path.
- Designed an evaluation harness that compares 16 chunking x retriever x reranker configurations
  on a 60-question gold set with recall@k, MRR, nDCG, citation and refusal accuracy, using paired
  bootstrap confidence intervals to separate real gains (reranking: +6 to +16 points recall@5)
  from noise.
- Made retrieval quality a CI gate: GitHub Actions runs lint, mypy, 80 tests and the full
  ablation offline with a deterministic LLM mock, and fails the build if any configuration's
  recall@5 drops more than 3 points below the committed baseline.
- Labelled the gold set with verbatim evidence quotes mapped to chunks through character offsets,
  so one gold file evaluates any chunking strategy fairly, and documented its biases and
  limitations alongside the results.
- Showed that an in-sample metric was misleading: a held-out set of on-topic unanswerable
  questions cut refusal accuracy from 93% to 55%; replacing pure word coverage with a
  cross-encoder answerability gate (threshold chosen on the main set only) raised held-out
  refusal accuracy to 75% without changing retrieval.
