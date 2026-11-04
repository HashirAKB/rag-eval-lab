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
