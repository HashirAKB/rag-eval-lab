# Learning guide

This guide is for studying the code: the concepts behind it, the order to read it in, one query
traced end to end, interview questions with answers that point at the code, and exercises.

## 1. Concepts from first principles

### Retrieval-augmented generation (RAG)

A language model only knows what was in its training data. RAG answers a question in two steps:
**retrieve** a few passages from your own documents that look relevant, then **generate** an answer
from those passages only. Two separate things can go wrong, and good RAG work measures them
separately:

1. Retrieval failure: the passage with the answer was not in what the model saw.
2. Generation failure: the passage was there, but the answer ignored it, misread it, or added
   claims it does not support.

This project measures (1) with recall@k, MRR and nDCG, and (2) with citation accuracy,
faithfulness and refusal accuracy.

### Chunking

Documents are too long to pass whole, and retrieval works better on focused passages, so documents
are cut into chunks. The trade-off: small chunks are precise but may cut an answer in half; large
chunks keep context but dilute the match and cost more tokens.

- **Fixed-size with overlap** (`chunking.chunk_fixed`): windows of about 800 characters that
  overlap by 160, so a sentence cut at one boundary appears whole in the next window.
- **Heading-aware** (`chunking.chunk_headings`): one chunk per markdown section, so chunks follow
  the author's structure. Tiny sections merge forward; long ones split at paragraph breaks.

Every chunk keeps `start`/`end` character offsets into its document. That one invariant
(`doc.text[start:end] == chunk.text`) is what makes evidence-based evaluation possible.

### Lexical retrieval: BM25

BM25 scores a chunk by the query words it contains. Two ideas:

- **IDF (inverse document frequency)**: a word that appears in few chunks ("ttlMs") tells you more
  than one that appears everywhere ("server"). `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`.
- **Saturating term frequency with length normalisation**: the 10th occurrence of a word adds
  less than the 1st (controlled by `k1`), and long chunks are penalised so they do not win just by
  containing more words (controlled by `b`).

It is fast, needs no model, and is excellent at exact identifiers. It fails on paraphrase: "stop a
running request" does not match "cancellation".

### Dense retrieval: embeddings

An embedder maps text to a vector so that texts with similar meaning are close. With unit-length
vectors, cosine similarity is a dot product, so scoring every chunk is one matrix multiplication
(`retrievers.DenseRetriever`). Two embedders here:

- **LSA** (`embedders.LsaEmbedder`): TF-IDF vectors compressed with truncated SVD. Learns word
  co-occurrence from this corpus only. Weak but fully offline.
- **MiniLM** (`embedders.MiniLmEmbedder`): a small pretrained transformer. Understands paraphrase
  in general English, but knows nothing about MCP identifiers.

### Hybrid retrieval and Reciprocal Rank Fusion (RRF)

BM25 and dense retrieval fail on different questions, so combine them. Their scores live on
different scales, so fuse **ranks**, not scores: each list gives an item `1 / (60 + rank)` and the
sums are re-sorted (`retrievers.reciprocal_rank_fusion`). No score normalisation, no tuning.

### Reranking with a cross-encoder

A bi-encoder (MiniLM) embeds query and chunk separately, so it never sees them together. A
cross-encoder reads the pair as one input and outputs a relevance score. It is much more accurate
and much slower, so it is used as a second stage: retrieve 20 candidates cheaply, then rerank them
(`retrievers.RerankingRetriever`). The reranker can only reorder what the first stage found, so
first-stage recall@20 is its ceiling.

### Grounding, citations and refusal

The answerer passes the top 5 chunks to the LLM with their ids and asks for an answer that cites
`[chunk_id]` after each claim. Citations make answers checkable. When the evidence is weak, the
right answer is "not in the docs"; here that decision is made **before** generation by a
confidence score (`answering.ConfidenceScorer`), and again after it if the LLM says
`NOT_IN_DOCS` or produces no valid citation.

### Evaluation

- **Gold set**: questions with known answers. Here labels are verbatim evidence quotes, so they
  work for any chunking (see `docs/adr/0001-evidence-quotes-as-gold-labels.md`).
- **recall@k**: did the evidence appear in the top k? The most important retrieval number,
  because the LLM only reads the top k.
- **MRR**: 1 / rank of the first relevant chunk, averaged. Rewards putting it first.
- **nDCG@k**: like recall, but a hit at rank 1 counts more than a hit at rank 10 (log discount).
- **Citation accuracy**: of the chunks the answer cites, how many actually hold the evidence.
- **Faithfulness**: is every claim in the answer supported by its cited chunks? Needs a judge: an
  LLM in live mode, a crude token-overlap heuristic in mock mode.
- **Refusal accuracy**: refused exactly when the question is unanswerable.
- **Bootstrap confidence intervals**: resample the questions with replacement thousands of times;
  the spread of the resampled means shows how much a score depends on which questions were asked.
  With 50 questions, a 4-point gap is often noise. The *paired* bootstrap compares two systems on
  the same resampled questions, which is the right test for "is A better than B?".
- **Regression gate**: CI recomputes recall@5 and fails if it drops more than 3 points below a
  committed baseline, so retrieval quality is protected like any other test.

## 2. Reading order

| # | File | What it teaches |
| --- | --- | --- |
| 1 | `SPEC.md`, `PLAN.md` | What is built and why; how milestones are verified. |
| 2 | `src/rag_eval_lab/corpus.py` | Cleaning MDX deterministically; why offsets need a stable text. |
| 3 | `src/rag_eval_lab/chunking.py` | Two chunkers, heading paths, the offset invariant. |
| 4 | `src/rag_eval_lab/text.py` | Tokenisation and stopwords; small choices with large effects. |
| 5 | `src/rag_eval_lab/retrievers.py` | BM25 by hand, dense search, RRF, reranking. Read the docstrings. |
| 6 | `src/rag_eval_lab/embedders.py`, `models.py` | LSA vs pretrained embeddings; lazy model loading. |
| 7 | `src/rag_eval_lab/llm.py` | Provider interface, the extractive mock, the live provider and judge. |
| 8 | `src/rag_eval_lab/answering.py` | Confidence scoring, refusal, citation extraction. |
| 9 | `src/rag_eval_lab/pipeline.py` | Configs, and how indexes are shared between them. |
| 10 | `evals/gold.jsonl`, `evals/GOLD.md`, `src/rag_eval_lab/gold.py` | Labels as quotes, mapping quotes to chunks, the gold set's limits. |
| 11 | `src/rag_eval_lab/metrics.py` | recall@k, MRR, nDCG with multiple evidence quotes, bootstrap. |
| 12 | `evals/run_eval.py`, `evals/results.md` | The ablation loop, paired comparisons, the written analysis. |
| 13 | `evals/check_regression.py`, `.github/workflows/ci.yml` | Turning a metric into a CI gate. |
| 14 | `src/rag_eval_lab/api.py`, `static/index.html` | Serving the same pipeline over HTTP. |
| 15 | `tests/` | How each piece is pinned down, including the golden snapshot. |
| 16 | `docs/adr/`, `DECISIONS.md` | The decisions and their trade-offs, in writing. |

## 3. One query traced end to end

Request: `POST /ask {"question": "How does a client cancel an in-flight request over stdio?"}`
with the default config `fixed/hybrid+rerank`.

1. **Startup** (`api.create_app` lifespan, once): `pipeline.build_pipeline` loads the 37 documents
   (`corpus.load_corpus`), cuts 801 fixed-size chunks (`chunking.chunk_corpus`), and
   `pipeline.build_retrievers` builds a BM25 index, a MiniLM embedding matrix (801 x 384), the
   hybrid retriever over both, and wraps it in `RerankingRetriever`. `ConfidenceScorer` counts
   document frequencies. `llm.get_provider` returns `MockProvider` because no key is set.
2. **Route** (`api.ask`): pydantic validates the body (1 to 500 characters) and calls
   `Answerer.ask`.
3. **First stage** (`HybridRetriever.search`): BM25 tokenises the question to
   `[client, cancel, flight, request, over, stdio]` (stopwords removed) and scores chunks through
   the inverted index; MiniLM embeds the question and takes a dot product with the matrix. Each
   returns its top 50; `reciprocal_rank_fusion` merges the two id lists.
4. **Rerank** (`RerankingRetriever.search`): the top 20 fused chunks are scored by the
   cross-encoder as (question, chunk) pairs and re-sorted. The top 5 are kept.
5. **Confidence** (`ConfidenceScorer.confidence`): rare words like "stdio" and "cancel" carry most
   of the IDF weight; the best chunk contains all of them, so confidence is about 0.85, above the
   0.35 threshold, and the answerer proceeds.
6. **Generate** (`MockProvider.answer`): the mock splits each chunk into prose sentences (skipping
   code, headings, tables and link definitions), ranks them by word overlap with the question, and
   returns the best two, each followed by its `[chunk_id]`. With a key, `AnthropicProvider.answer`
   sends the chunks in `<chunk id="...">` tags with instructions to cite and to reply
   `NOT_IN_DOCS` when unsupported.
7. **Check citations** (`answering.extract_citations`): keeps only ids that were actually
   retrieved; an answer with no valid citation becomes a refusal.
8. **Respond** (`api.AskResponse`): answer text, citations, refusal flag, confidence and the five
   chunks with scores. The HTML page links each `[chunk_id]` to its chunk card.

In the eval harness (`evals/run_eval.evaluate_config`) the same retriever runs at depth 20 for the
ranking metrics, and `Answerer.answer_from_hits` reuses those hits, so the answer evaluated is the
answer the API would give.

## 4. Interview questions with model answers

1. **Why store gold labels as quotes instead of chunk ids?**
   Chunk ids depend on the chunking strategy, so they cannot compare two chunkers. Quotes are
   mapped to chunks per strategy in `gold.relevant_chunk_sets` using character offsets, which the
   chunkers guarantee. A test checks every quote still exists verbatim after a corpus refresh.

2. **How do you know the winning configuration is actually better?**
   I don't claim it is better than the other top three. `evals/results.md` reports a paired
   bootstrap of per-question recall@5 differences; against the other top reranked configs the 95%
   intervals include zero. What the data does support is that reranking helps: the winner beats
   every non-reranked config with intervals whose lower bound is at or above zero.

3. **Why does BM25 beat MiniLM here?**
   The questions contain protocol identifiers (`server/discover`, `ttlMs`) and spec vocabulary that
   exact matching handles and a small general-domain embedder does not. Part of it is bias: the
   questions were written from the documents. They also fail on different questions (11 vs 7 unique
   hits), which motivates hybrid retrieval.

4. **Why fuse with RRF instead of adding normalised scores?**
   BM25 scores are unbounded and query-dependent, cosine is in [-1, 1]; any normalisation is a
   tuning knob that shifts with the query. RRF only uses ranks, has one constant (k = 60) that is
   rarely tuned, and is robust. The cost: it ignores how confident each retriever was.

5. **Why did hybrid without reranking barely beat BM25?**
   RRF rewards agreement; when MiniLM ranks a wrong chunk highly, fusion promotes it. Hybrid's real
   value shows as candidate recall@20 (96-98%), which the reranker then exploits.

6. **What limits the reranker?**
   It reorders only the first stage's top 20, so first-stage recall@20 caps it. It is also the
   latency bottleneck (about 0.5 to 1 s per query on CPU), so depth is a recall vs latency knob.

7. **How is the refusal decision made, and what are its failure modes?**
   `ConfidenceScorer`: the IDF-weighted share of question words present in the best of the top 5
   chunks, thresholded at 0.35. It fails on paraphrases (no stemming, so false refusals) and,
   badly, on on-topic unanswerable questions that use only corpus vocabulary: on the held-out set
   it misses 8 of 10 (`evals/experiments/holdout`). It measures topicality, not answerability.

8. **Why decide refusal before calling the LLM?**
   It is retriever-independent, testable offline, identical in mock and live mode, and saves a
   call. The LLM can still refuse afterwards, and answers without valid citations are refused.

9. **How would you measure faithfulness, and why is the mock number 100%?**
   Live: an LLM judge with a JSON schema output (`AnthropicProvider.faithfulness`) scores support of
   each claim by the cited chunks. Mock: token overlap with the cited chunks. The mock answer is
   extractive, so overlap is 100% by construction; the results label it as a heuristic.

10. **What does nDCG add over recall@k, and how do you handle a quote split across chunks?**
    nDCG rewards rank position with a log discount. In `metrics._dcg` a chunk gains a point only for
    evidence quotes not already covered, so two halves of the same quote are not double counted;
    the ideal DCG uses a greedy best ordering.

11. **Why is the regression gate on every configuration with a 3-point tolerance?**
    Shared code (tokeniser, chunker) can silently break a config that is not served today. Three
    points is 1.5 questions: catches real regressions, absorbs CPU floating-point jitter in the
    neural models. Intentional changes update `evals/baseline.json` in the same commit.

12. **How does the project run with no API key, and what is real vs mocked?**
    Everything goes through `llm.LLMProvider`. Retrieval, reranking, chunking and metrics are real
    in both modes. Only answer generation and the faithfulness judge are mocked, and every result
    records which provider produced it.

13. **What are the biggest threats to the validity of these numbers?**
    The questions were written by the system's builder from the source text (lexical bias), there
    are only 50 answerable and 10 unanswerable items, each question has one labelled source (recall
    is a lower bound; q19 and q22 are label misses), and the refusal threshold was tuned on the same
    set.

14. **Why no vector database?**
    About 800 chunks: an exact dot product over a numpy matrix takes well under a millisecond and
    has no recall loss, so evaluation is not blurred by approximate search. I would add an ANN index
    and persisted embeddings at hundreds of thousands of chunks (ADR 0002).

15. **What would you change first to improve the system?**
    Measured candidates: contextual chunk headers (prepend the heading path before indexing),
    stemming in the tokeniser (helps BM25 and the refusal score on paraphrases), multi-source
    labels for the gold set, harder on-topic negatives, and a reranker depth sweep for latency.
    Each is a config or a gold change, so the harness answers whether it helped.

## 5. Hands-on exercises

1. **Break the chunker invariant.** In `chunking._split_long`, make pieces strip trailing
   whitespace. Run `pytest tests/test_chunking.py tests/test_gold.py`. Which tests fail, and why
   does the gold mapping depend on it?
   *Hint:* look for `doc.text[c.start : c.end] == c.text`.

2. **Tune BM25.** Change `k1` and `b` in `Bm25Retriever` (try `b=0` and `b=1`) and run
   `python -m evals.run_eval --configs fixed/bm25,heading/bm25 --output-dir /tmp/bm25`. Does length
   normalisation matter more for fixed or heading chunks?
   *Hint:* heading chunks vary much more in length.

3. **Add stemming.** Add a tiny suffix stripper (`ing`, `ed`, `s`) to `text.tokenize`. Re-run the
   eval and the refusal sweep. Did false refusals drop? Did BM25 recall change? Does the regression
   gate pass, and should you update the baseline?
   *Hint:* both BM25 and `ConfidenceScorer` use `tokenize`.

4. **Make refusals harder.** Add five on-topic unanswerable questions to `evals/gold.jsonl` (for
   example about a method the spec does not define). Update the count in `tests/test_gold.py` and
   re-run. How does refusal accuracy change, and what threshold would you pick now?
   *Hint:* look at `refusal_sweep` in `evals/results.json`.

5. **Add a retriever.** Implement a `TitleBoostRetriever` that adds a bonus when a query word
   appears in the chunk's `heading_path`. Register it in `pipeline.RETRIEVERS` and
   `build_retrievers`, add a unit test in `tests/test_retrievers.py`, run the ablation, and report
   the paired bootstrap against BM25.
   *Hint:* the heading path is already on every `Chunk`.
