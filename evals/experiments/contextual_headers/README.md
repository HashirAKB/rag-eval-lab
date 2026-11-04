# Experiment: contextual chunk headers

Paired bootstrap over the answerable questions: change in recall@5 (new minus base), in points, with a 95% confidence interval.

| Config | Base recall@5 | New recall@5 | Change | 95% CI |
| --- | --- | --- | --- | --- |
| fixed+ctx/bm25 | 77.0 | 79.0 | +2.0 | -4.0 to +8.0 |
| fixed+ctx/bm25+rerank | 83.0 | 91.0 | +8.0 | +2.0 to +16.0 |
| fixed+ctx/dense-minilm | 69.0 | 73.0 | +4.0 | -4.0 to +12.0 |
| fixed+ctx/dense-minilm+rerank | 81.0 | 85.0 | +4.0 | -2.0 to +12.0 |
| fixed+ctx/hybrid | 75.0 | 85.0 | +10.0 | +2.0 to +18.0 |
| fixed+ctx/hybrid+rerank | 89.0 | 93.0 | +4.0 | +0.0 to +10.0 |
| heading+ctx/bm25 | 77.0 | 77.0 | +0.0 | -6.0 to +6.0 |
| heading+ctx/bm25+rerank | 87.0 | 85.0 | -2.0 | -8.0 to +4.0 |
| heading+ctx/dense-minilm | 73.0 | 74.0 | +1.0 | -6.0 to +8.0 |
| heading+ctx/dense-minilm+rerank | 79.0 | 81.0 | +2.0 | -6.0 to +10.0 |
| heading+ctx/hybrid | 79.0 | 79.0 | +0.0 | +0.0 to +0.0 |
| heading+ctx/hybrid+rerank | 85.0 | 87.0 | +2.0 | -4.0 to +8.0 |

## What was tested

Each chunk is indexed (BM25, MiniLM embeddings and the reranker input) as
`"Title > Section > Subsection\n\n" + chunk text` instead of the chunk text alone
(`chunking.contextual_text`). Chunk boundaries, answers and citations are unchanged.
Base numbers come from `evals/results.json`; new numbers from `results.json` in this folder.

## Findings

- **Fixed-size chunks benefit; heading-aware chunks do not.** Every fixed-size config improves
  (+2 to +10 points), with three intervals at or above zero. Heading-aware configs move by -2 to +2,
  all within noise. This matches the mechanism: a heading-aware chunk already begins with its own
  heading line, while a fixed window cut from the middle of a section has lost the section title,
  and titles carry exactly the words questions use ("Pagination", "Cancellation").
- **The largest gain is first-stage hybrid on fixed chunks (+10, CI +2 to +18).** Both BM25 and
  MiniLM see the title words, so their fused candidate list improves.
- **`fixed+ctx/hybrid+rerank` reaches 93% recall@5**, the best number measured so far, but its gain
  over the served `fixed/hybrid+rerank` (+4, CI +0 to +10) is borderline on 50 questions.

## Decision

Not promoted to the served default yet: the served-config gain is within noise, and gating a 17th
configuration would lengthen CI. It is the first candidate to re-test on a larger, multi-source
gold set. See DECISIONS.md (D13).

## Reproduce

```bash
python -m evals.run_eval --output-dir evals/experiments/contextual_headers --configs "$(python -c "from rag_eval_lab.pipeline import EXPERIMENT_CONFIGS as E; print(','.join(c.name for c in E))")"
```

```bash
python -m evals.compare evals/results.json evals/experiments/contextual_headers/results.json --strip +ctx
```
