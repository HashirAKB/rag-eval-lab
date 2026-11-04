# Decisions log

Decisions made while building without the owner available. Format: decision, alternatives considered, why.

## D1. Push directly to `main`
- Alternatives: work on a feature branch and open a PR.
- Why: the owner explicitly asked for direct pushes to `main`.

## D2. Corpus subset: one spec version of the MCP docs
- Alternatives: the whole `docs/` tree (7 MB, six near-identical spec versions); the FastAPI docs.
- Why: duplicate pages across versions would make "which chunk is the source?" ambiguous and inflate
  recall for free. One version (2026-07-28) plus the learn and security tutorials is about 500 KB,
  large enough to be non-trivial and small enough to embed on a CPU in seconds.

## D3. Vendor the corpus subset into `data/corpus/`
- Alternatives: clone at build time in CI and Docker.
- Why: reproducible offline builds and evals; the licenses (Apache-2.0 / CC-BY-4.0) allow it with
  attribution. `scripts/fetch_corpus.sh` still documents and re-creates the exact snapshot.

## D4. Torch from the CPU wheel index in CI and Docker
- Alternatives: default PyPI torch (pulls about 4 GB of CUDA libraries).
- Why: nothing here uses a GPU; the CPU wheel is a fraction of the size and install time.

## D5. Refusal is decided before the LLM, from IDF-weighted term coverage
- Alternatives: threshold on the retriever's own top score (not comparable across BM25,
  cosine and RRF); threshold on the cross-encoder score (only exists with a reranker); let the
  LLM decide alone (impossible to test offline, and costs a call per refusal).
- Why: one retriever-independent, explainable signal that behaves the same in mock and live
  mode. Known weakness: exact word matching (no stemming), so paraphrased questions score lower.

## D6. Live model `claude-opus-5-5` with server-side refusal fallback
- Alternatives: a cheaper model for answers and judging.
- Why: current default model; override with `ANTHROPIC_MODEL`. Effort is set to `low` because
  both tasks are short reading-comprehension calls. `fallbacks: "default"` retries
  safety-classifier declines on a suitable model instead of failing the eval item.

## D7. Mock answers are extractive
- Alternatives: canned answers per question; a template that echoes the question.
- Why: an extractive answer is deterministic, grounded by construction, carries real citations,
  and exercises the citation-accuracy metric honestly. It is not a claim about answer quality.
