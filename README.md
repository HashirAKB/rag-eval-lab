# RAG Eval Lab

A retrieval-augmented question answering system over the Model Context Protocol (MCP)
documentation, whose real product is its evaluation harness. It answers one question with
evidence: **which RAG design is actually better, by how much, and how do you know?** Sixteen
configurations (2 chunkers x 4 retrievers x reranker on/off) are scored on a 60-question gold set
with recall@k, MRR, nDCG, citation accuracy, refusal accuracy and faithfulness, compared with
paired bootstrap confidence intervals, and protected in CI by a recall@5 regression gate. It
builds, tests, evaluates and demos with zero API keys; a live Anthropic model is used only when
`ANTHROPIC_API_KEY` is set.

![The demo page: an answer with linked citations above the retrieved chunks and their scores](docs/demo.png)

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        A[data/corpus/*.mdx<br/>pinned MCP docs] --> B[corpus.py<br/>clean MDX]
        B --> C[chunking.py<br/>fixed or heading]
    end
    subgraph Retrieve
        C --> D1[BM25]
        C --> D2[Dense: LSA or MiniLM]
        D1 --> E[Hybrid: RRF]
        D2 --> E
        E --> F[Cross-encoder rerank]
    end
    subgraph Answer
        F --> G[answering.py<br/>confidence check]
        G -->|low| R[Refuse: not in the docs]
        G -->|ok| H[LLMProvider<br/>Mock or Anthropic]
        H --> I[Answer + citations]
    end
    subgraph Evaluate
        J[evals/gold.jsonl<br/>60 questions] --> K[run_eval.py<br/>16 configs]
        F --> K
        I --> K
        K --> L[results.md / results.json]
        L --> M[check_regression.py<br/>CI gate]
    end
    I --> N[FastAPI /ask /search + HTML]
```

## Quickstart

```bash
pip install -e ".[dev,neural]"
python -m evals.run_eval
uvicorn rag_eval_lab.api:app
```

Then open http://localhost:8000. The first command installs the package with the local models
(tip: install CPU-only torch first with `pip install torch --index-url
https://download.pytorch.org/whl/cpu` to avoid a large CUDA download). The second runs the full
ablation in mock mode (about 8 minutes on a laptop CPU) and rewrites `evals/results.md`. The third
serves the winning configuration; set `RAG_CONFIG=heading/bm25` for an instant start without
models. Docker: `docker build -t rag-eval-lab . && docker run -p 8000:8000 rag-eval-lab`.

Checks: `ruff check .`, `mypy`, `pytest`, `python -m evals.check_regression`.

## Demo output

`POST /ask {"question": "How does a client cancel an in-flight request over stdio?"}` (mock
provider, hit list shortened to ids and reranker scores):

```json
{
  "config": "fixed/hybrid+rerank",
  "provider": "mock",
  "answer": "To cancel an in-flight request, the client **MUST** send a `notifications/cancelled` notification referencing the request's ID. [specification/basic/transports/stdio#f005] Each binding defines how a client abandons an in-flight request: on stdio the client sends a `notifications/cancelled` notification; on Streamable HTTP it closes the request's response stream. [specification/basic/transports/index#f003]",
  "citations": ["specification/basic/transports/stdio#f005", "specification/basic/transports/index#f003"],
  "refused": false,
  "confidence": 0.85,
  "hits": [
    {"chunk_id": "specification/basic/transports/stdio#f005", "score": 8.38},
    {"chunk_id": "specification/basic/transports/index#f003", "score": 8.2},
    {"chunk_id": "specification/basic/patterns/cancellation#f002", "score": 5.98},
    {"chunk_id": "specification/basic/patterns/cancellation#f001", "score": 5.51},
    {"chunk_id": "specification/basic/patterns/subscriptions#f006", "score": 4.68}
  ]
}
```

An out-of-scope question is refused before any LLM call (word coverage 0.33 is below 0.35):

```json
{"question": "How do I deploy an MCP server on Kubernetes with Helm?",
 "answer": "Not in the docs: I could not find this in the MCP documentation.",
 "refused": true, "confidence": 0.33, "citations": []}
```

## Eval results

Mock mode, 50 answerable + 10 unanswerable questions. R@5 is recall@5 in percent with a 95%
bootstrap interval; latency is retrieval time per query on the build machine's CPU. Reranked
configs refuse on low word coverage *or* a low cross-encoder score; the others on coverage only. Full table,
paired comparisons, threshold sweep and analysis: [`evals/results.md`](evals/results.md).

| Config | R@5 (95% CI) | MRR | nDCG@10 | Citation acc | Refusal acc | ms/query |
| --- | --- | --- | --- | --- | --- | --- |
| fixed/hybrid+rerank | 89 (80 to 97) | 64.8 | 71.1 | 38.0 | 96.7 | 613 |
| heading/dense-lsa+rerank | 89 (80 to 96) | 65.2 | 72.2 | 37.5 | 93.3 | 991 |
| heading/bm25+rerank | 87 (77 to 96) | 64.7 | 70.9 | 35.4 | 93.3 | 987 |
| heading/hybrid+rerank | 85 (75 to 94) | 65.2 | 72.6 | 41.7 | 93.3 | 1082 |
| fixed/bm25+rerank | 83 (73 to 93) | 63.9 | 69.9 | 36.0 | 96.7 | 694 |
| fixed/dense-lsa+rerank | 83 (73 to 93) | 64.6 | 70.9 | 38.0 | 96.7 | 638 |
| fixed/dense-minilm+rerank | 81 (70 to 91) | 58.4 | 64.3 | 37.5 | 95.0 | 601 |
| heading/dense-minilm+rerank | 79 (67 to 89) | 60.7 | 66.7 | 37.2 | 93.3 | 1004 |
| heading/hybrid | 79 (67 to 90) | 59.4 | 65.4 | 34.0 | 90.0 | 17 |
| fixed/bm25 | 77 (65 to 88) | 60.0 | 65.6 | 35.0 | 90.0 | 1 |
| heading/bm25 | 77 (65 to 88) | 57.3 | 63.3 | 34.7 | 88.3 | 1 |
| fixed/hybrid | 75 (63 to 86) | 57.0 | 62.7 | 34.4 | 90.0 | 15 |
| heading/dense-lsa | 73 (61 to 85) | 52.4 | 61.3 | 33.3 | 86.7 | 4 |
| heading/dense-minilm | 73 (61 to 85) | 57.6 | 63.3 | 44.4 | 88.3 | 15 |
| fixed/dense-lsa | 71 (58 to 83) | 52.8 | 59.4 | 37.2 | 85.0 | 3 |
| fixed/dense-minilm | 69 (56 to 81) | 45.8 | 51.2 | 33.3 | 83.3 | 31 |

What the numbers say:

- **Reranking is the change that matters**: +6 to +16 points of recall@5 for every first stage.
- **The top four are a tie.** Their differences (0 to 4 points) sit inside the paired bootstrap
  intervals. `fixed/hybrid+rerank` is served because it is tied for best and is the cheapest of
  the four to rerank.
- **BM25 beats a small general-purpose embedder** on protocol documentation full of identifiers,
  and hybrid retrieval pays off mainly as a better candidate pool for the reranker (candidate
  recall@20 of 96 to 98%).
- **Answer metrics are mock-mode numbers.** Citation accuracy reflects an extractive mock that
  quotes the two most word-overlapping sentences; faithfulness is 100% by construction for the
  same reason and is omitted here.

## Design decisions and trade-offs

- **Gold labels are evidence quotes, not chunk ids** ([ADR 0001](docs/adr/0001-evidence-quotes-as-gold-labels.md)).
  One gold file evaluates any chunking strategy; the cost is that each question credits only one
  labelled passage, so recall is a lower bound.
- **Exact numpy search, no vector database** ([ADR 0002](docs/adr/0002-brute-force-vector-search.md)).
  About 800 chunks need no approximate index; evaluation is not blurred by ANN recall loss.
- **A provider interface with a deterministic mock as the default** ([ADR 0003](docs/adr/0003-provider-interface-with-deterministic-mock.md)).
  Offline, reproducible CI; the price is that answer-quality numbers are only meaningful in live mode.
- **Refuse before generation** ([ADR 0004](docs/adr/0004-refuse-before-generation.md)), from
  IDF-weighted word coverage plus, for reranked configs, the cross-encoder's top score. Cheap and
  testable offline; coverage alone is blind to on-topic unanswerable questions, which is why the
  reranker signal was added.
- **Serve hybrid + rerank, gate every config in CI** ([ADR 0005](docs/adr/0005-rerank-and-gate-every-config.md)).
  The reranker costs about half a second per query on a CPU; a 3-point tolerance absorbs model
  jitter but still catches a 2-question regression.
- **BM25 written by hand** (about 40 lines in `retrievers.py`) to keep the formula readable, and
  a vendored corpus snapshot (pinned commit, licence notice) so builds never depend on the network.
- Smaller decisions made along the way are logged in [`DECISIONS.md`](DECISIONS.md).

## Limitations

- **The gold set is small and author-written.** 60 questions written by the system's builder from
  the documents; they favour lexical matching. One question is 2 points of recall. See
  [`evals/GOLD.md`](evals/GOLD.md).
- **Single-source labels.** When another page also answers a question, a correct retrieval counts
  as a miss (2 of the winner's 6 misses are this).
- **Refusals are in-sample optimistic.** The served config reaches 96.7% refusal accuracy on the
  set its thresholds were tuned on, but 75% on 20 held-out questions with on-topic unanswerable
  items (it misses 3 of 10 and falsely refuses 2 of 10). Word coverage alone scored 55% there;
  adding the cross-encoder score fixed part of that. The held-out set has now informed one design
  decision, so a fresh set is needed for the next unbiased estimate. See
  [`evals/experiments/refusal_signals`](evals/experiments/refusal_signals/README.md).
- **No live-mode results.** No API key was available while building, so answer quality and LLM-judge
  faithfulness were never measured live; the live path is covered by unit tests with a fake client.
- **MiniLM truncates** inputs at 256 word pieces, so the longest heading-aware chunks (up to 1500
  characters) are partly ignored by the dense retriever.
- **CPU latency.** Reranking takes about 0.5 to 1 second per query. BM25 and LSA indexes are
  rebuilt at startup; MiniLM corpus embeddings are cached on disk (`.cache/embeddings`).
- The Docker image is built in CI; it could not be built locally because Docker Hub rate-limited
  the build machine.

## What I'd do next

1. ~~Cache embeddings on disk~~ **Done**: MiniLM corpus embeddings are saved under
   `.cache/embeddings`, keyed by a hash of the model name and every chunk text. Startup of the
   default config dropped from 25.0 s to 1.2 s on the build machine.
2. ~~Contextual chunk headers~~ **Measured**: indexing each chunk with its heading path prepended
   lifts every fixed-size config (+2 to +10 points recall@5; `fixed+ctx/hybrid+rerank` reaches 93%)
   and leaves heading-aware chunks unchanged. Not yet promoted because the gain for the served
   config is borderline. Report: [`evals/experiments/contextual_headers`](evals/experiments/contextual_headers/README.md).
3. ~~Stemming in the tokenizer~~ **Measured, not adopted**: small first-stage gains (+2 to +8,
   all within noise), no change for the served config, and worse refusals (4 to 6 missed). Kept
   as a patch with results: [`evals/experiments/stemming`](evals/experiments/stemming/README.md).
4. ~~Reranker depth sweep~~ **Measured**: recall@5 is 75 / 83 / 89 / 87 when reranking 5 / 10 /
   20 / 40 candidates, at about 150 / 260 / 535 / 1071 ms per query. 20 stays the default; 10 is
   the latency-friendly option. Report: [`evals/experiments/rerank_depth`](evals/experiments/rerank_depth/README.md).
5. ~~Harder unanswerable questions~~ **Done as a held-out set** (`evals/holdout.jsonl`, 10
   answerable + 10 on-topic unanswerable, never used for tuning). Result: with word coverage
   alone, refusal accuracy drops from 93% to 55% for the served config, which exposed the main
   weakness of the refusal heuristic and motivated item 6.
   Report: [`evals/experiments/holdout`](evals/experiments/holdout/README.md).
6. ~~An answerability signal that reads the passage~~ **Done**: reranked configs also refuse when
   the cross-encoder's top score is below 0.25 (chosen on the main set). Held-out refusal accuracy
   for the served config rose from 55% to 75%; main-set accuracy from 93.3% to 96.7%. Report:
   [`evals/experiments/refusal_signals`](evals/experiments/refusal_signals/README.md).
7. **Multi-source gold labels**: *started*. `python -m evals.label_review_queue` lists, per
   question, the chunks from other documents that the served config ranks above the labelled
   source (41 candidates across 16 questions, in [`evals/label_review_queue.md`](evals/label_review_queue.md)).
   Accepting them needs a human annotator plus a second reviewer; the script deliberately never
   edits `gold.jsonl`.
8. **Live-mode results**: run the eval with `ANTHROPIC_API_KEY` set to measure real answer quality
   and LLM-judge faithfulness, and record a few live answers as replay fixtures.
