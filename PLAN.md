# Plan

Each milestone: tests first, simplest implementation, all checks green, commit, push, tick here.
Success criteria for each milestone are in SPEC.md.

- [x] M0 Scaffold: package layout, pyproject, ruff, mypy, pytest, CI workflow, Dockerfile, .env.example, LICENSE
- [x] M1 Corpus snapshot + MDX cleaning + fixed-size and heading-aware chunkers
- [x] M2 Retrievers: BM25, dense (LSA + MiniLM), hybrid RRF, cross-encoder reranker
- [x] M3 LLM provider interface (Mock + Anthropic), grounded answering, refusal
- [x] M4 Gold set (60 questions, 10 unanswerable) + metrics
- [x] M5 Eval harness, ablation table + analysis, regression gate in CI
- [x] M6 FastAPI /ask, /search, HTML page; README, LEARNING, ADRs, SUMMARY
- [x] Final senior-reviewer pass (round 1: dead code, prompt wording, judge refusal handling, citation regex, stable results.json)

## After the Definition of Done ("What I'd do next" in README)

- [x] Cache MiniLM corpus embeddings on disk (startup 25.0 s to 1.2 s); fixed Docker data paths
- [x] Experiment: contextual chunk headers (helps fixed-size chunks; not promoted, borderline)
- [x] Experiment: light stemming (rejected; kept as a patch)
- [x] Experiment: reranker depth sweep (20 stays the default)
- [x] Held-out set with on-topic unanswerable questions (exposed the refusal weakness)
- [x] Cross-encoder refusal gate for reranked configs (held-out refusal 55% to 75%)
- [~] Multi-source gold labels: review queue generated; accepting labels needs human annotators
- [ ] Live-mode results: blocked, no API key available in the build environment
- [x] Second reviewer pass over code and docs after the extra work

## Blockers and decisions log

- Docker Hub returned 429 for `python:3.11-slim` in the build sandbox: image is built and
  smoke-tested in CI instead (SPEC.md, Deviations).
- No API key: the live provider is unit-tested with a fake client only.
- Decisions are logged in DECISIONS.md (D1 to D16).
