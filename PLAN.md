# Plan

Each milestone: tests first, simplest implementation, all checks green, commit, push, tick here.
Success criteria for each milestone are in SPEC.md.

- [x] M0 Scaffold: package layout, pyproject, ruff, mypy, pytest, CI workflow, Dockerfile, .env.example, LICENSE
- [ ] M1 Corpus snapshot + MDX cleaning + fixed-size and heading-aware chunkers
- [ ] M2 Retrievers: BM25, dense (LSA + MiniLM), hybrid RRF, cross-encoder reranker
- [ ] M3 LLM provider interface (Mock + Anthropic), grounded answering, refusal
- [ ] M4 Gold set (60 questions, 10 unanswerable) + metrics
- [ ] M5 Eval harness, ablation table + analysis, regression gate in CI
- [ ] M6 FastAPI /ask, /search, HTML page; README, LEARNING, ADRs, SUMMARY
- [ ] Final senior-reviewer pass

## Blockers and decisions log

(none yet)
