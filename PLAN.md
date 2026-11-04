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

## Blockers and decisions log

(none yet)
