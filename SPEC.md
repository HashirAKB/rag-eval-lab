# RAG Eval Lab: Specification

## Goal

Build a small retrieval-augmented question answering (RAG) system over a real
documentation corpus, and make its evaluation harness the real product. The
harness must answer one question with evidence: **which RAG design is actually
better, by how much, and how do you know?**

## Scope

- Ingest a pinned snapshot of the Model Context Protocol (MCP) docs (markdown/MDX)
  with clean metadata: `path`, `heading_path`, `chunk_id`, character offsets.
- Two chunking strategies: fixed-size with overlap, and heading-aware markdown.
- Four retrievers: BM25, dense (LSA embedder), dense (MiniLM embedder), hybrid
  (BM25 + MiniLM fused with Reciprocal Rank Fusion). Optional cross-encoder
  reranker on top of any retriever.
- Grounded answering with inline citations (`[chunk_id]`) and an explicit
  "not in the docs" refusal when retrieval confidence is low.
- An LLM provider interface with a deterministic `MockProvider` (default, used by
  tests, evals and demo) and an `AnthropicProvider` (used only when
  `ANTHROPIC_API_KEY` is set).
- A 60-question gold set (50 answerable, 10 unanswerable) with evidence labels.
- Metrics: recall@k, MRR, nDCG@k, citation accuracy, refusal accuracy,
  faithfulness (LLM-as-judge in live mode, token-overlap heuristic in mock mode).
- An ablation over chunking x retriever x reranker, committed to `evals/results.md`
  with a written analysis.
- A CI regression gate: fail if recall@5 of any configuration drops more than
  3 points below the committed baseline.
- FastAPI `/ask` and `/search` endpoints plus a minimal HTML page.

## Non-goals

- No vector database, no distributed indexing: a few thousand chunks fit in a
  numpy array and brute-force cosine similarity is exact and fast.
- No agent loops, query rewriting, multi-hop retrieval or conversation memory.
- No UI framework: one static HTML page with vanilla JS.
- No authentication, rate limiting or production hardening of the API.
- No fine-tuning of embedders or rerankers.
- Not a general benchmark: results are specific to this corpus and gold set.

## Architecture

```
data/corpus/*.mdx  --(corpus.py: clean MDX)-->  Documents
Documents          --(chunking.py: fixed | heading)-->  Chunks (id, path, heading_path, start, end, text)
Chunks             --(retrievers.py: bm25 | dense | hybrid [+ rerank])-->  ranked (chunk, score)
ranked chunks      --(answering.py: confidence check -> LLMProvider)-->  Answer(text, citations, refused)
gold.jsonl + runs  --(metrics.py + evals/run_eval.py)-->  evals/results.json + results.md
results.json       --(evals/check_regression.py vs evals/baseline.json)-->  CI pass / fail
api.py (FastAPI)   --  /search, /ask, /  (HTML page)
```

The pipeline is built from a `Config(chunker, retriever, reranker)` value so the
eval harness and the API use exactly the same code path.

## Tech choices

| Choice | Reason |
| --- | --- |
| Python 3.11+ | Required; type hints and dataclasses keep the code readable. |
| FastAPI + uvicorn | Required; typed request/response models with almost no code. |
| Own BM25 (about 40 lines, numpy) | Teaches the formula; avoids a dependency for a tiny algorithm. |
| scikit-learn TF-IDF + TruncatedSVD (LSA) | Offline dense embedder that needs no model download. |
| sentence-transformers `all-MiniLM-L6-v2` | Small (90 MB), CPU-friendly, standard dense baseline. |
| sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small standard reranker. |
| numpy brute-force cosine | Exact search; the corpus is small, so an ANN index adds nothing. |
| `anthropic` SDK (optional) | Live answers and LLM-as-judge only when a key is present. |
| pytest, ruff, mypy | Tests, lint + format, type checking. |
| Vendored corpus subset in `data/corpus/` | Builds, tests and CI are reproducible offline; license permits it. |

## Corpus

- Source: `https://github.com/modelcontextprotocol/modelcontextprotocol`
- Pinned commit: `3c4d91b7e832d9b74338d56eed03403ccc662123` (2026-11-02)
- Subset (one spec version only, to avoid near-duplicate pages across versions):
  - `docs/specification/2026-07-28/**/*.mdx` except `schema.mdx` (auto-generated, 726 KB)
  - `docs/docs/2026-07-28/learn/*.mdx`
  - `docs/docs/2026-07-28/tutorials/security/*.mdx`
- `scripts/fetch_corpus.sh` re-creates `data/corpus/` from the pinned commit.
- License: Apache-2.0 (specification) and CC-BY-4.0 (docs); attribution in
  `data/corpus/NOTICE.md`.

## Gold set design

Each gold item stores the question, the source document path, and one or more
short **evidence quotes** copied verbatim from that document. A chunk is
relevant to the question if its character span overlaps an evidence quote's
span. This makes labels independent of the chunking strategy: the same gold
file evaluates fixed-size and heading-aware chunks fairly. The eval prints the
derived relevant chunk ids per strategy for inspection. Unanswerable items
have no evidence and `answerable: false`.

## Metrics (definitions)

- **recall@k**: fraction of a question's evidence quotes that are covered by at
  least one relevant chunk in the top k, averaged over answerable questions.
- **MRR**: mean of 1 / rank of the first relevant chunk (0 if none in top 20).
- **nDCG@k**: binary gains, where a chunk gains 1 only the first time it covers
  a not-yet-covered evidence quote; ideal DCG assumes all quotes covered at the top.
- **citation accuracy**: of the citations in non-refused answers to answerable
  questions, the fraction that point to a relevant chunk.
- **refusal accuracy**: fraction of all 60 questions where the system refused
  exactly when the question is unanswerable.
- **faithfulness**: live mode, an LLM judge scores support of the answer by the
  cited chunks (0..1); mock mode, the fraction of answer content tokens that
  appear in the cited chunks. Results always label which one was used.

## Milestones

| ID | Milestone | Verifiable success criterion |
| --- | --- | --- |
| M0 | Scaffold: package, tooling, CI, Dockerfile | `ruff check`, `mypy src`, `pytest` pass locally; CI workflow file present. |
| M1 | Corpus + ingestion + two chunkers | `pytest tests/test_corpus.py tests/test_chunking.py` passes; every chunk has path, heading path, id, and offsets that slice back to its text. |
| M2 | Retrievers: BM25, dense (LSA, MiniLM), hybrid RRF, reranker | Unit tests pass; a known query ranks its source chunk in the top 5 for every retriever. |
| M3 | LLM providers + grounded answering + refusal | Mock answers contain only valid citations; low-confidence queries return the refusal; Anthropic provider is never constructed without a key. |
| M4 | Gold set (60 items) + metrics | Gold validator test passes (every quote found verbatim in its document, 10 unanswerable); metric unit tests on hand-computed examples pass. |
| M5 | Eval harness, ablation, regression gate | `python -m evals.run_eval` writes `results.json` + `results.md`; `check_regression` passes against baseline and fails on a synthetic 4-point drop (tested). |
| M6 | API + HTML page + docs | `/search` and `/ask` integration tests pass; README, LEARNING, ADRs, SUMMARY present; Docker image builds (if docker daemon available). |

## Risks

- **Gold set bias**: questions are written by the same agent that builds the
  system, from the documents themselves, so they favour lexical overlap and
  BM25. Mitigation: paraphrase questions away from the source wording and
  report this limitation.
- **Small gold set**: 50 answerable questions means one question is 2 points
  of recall. Mitigation: report bootstrap confidence intervals for the winner.
- **Refusal threshold tuned on the eval set**: leaks information. Mitigation:
  threshold chosen on a fixed rule and documented; report the trade-off.
- **Model downloads in CI**: MiniLM and the cross-encoder come from the
  Hugging Face Hub. Mitigation: cache in CI; LSA embedder works offline.
- **Mock faithfulness is near-trivial** for extractive mock answers. Mitigation:
  label it clearly; it exists to exercise the pipeline, not to claim quality.

## Deviations

- Gold labels are evidence quotes mapped to chunk ids per chunking strategy,
  rather than fixed chunk ids, because chunk ids differ between strategies.
- "Recorded fixtures" are golden snapshots of `MockProvider` outputs used by
  tests. No live Anthropic recordings exist because no API key was available
  while building.
- The Docker image could not be built inside the build sandbox: Docker Hub answered
  `429 Too Many Requests` for `python:3.11-slim`. A single later retry pulled the base image, but
  `pip` inside the build then failed TLS verification behind the sandbox's intercepting proxy.
  Per the project rules no workaround route was tried. Instead, CI has a `docker` job that builds
  the image and smoke-tests `/search` inside the container on GitHub's runners (passing).
- M3's live provider is covered by unit tests with a fake client; no live API call was made
  because no API key was available.
