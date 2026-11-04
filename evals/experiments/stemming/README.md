# Experiment: light stemming in the tokenizer

Paired bootstrap over the answerable questions: change in recall@5 (new minus base), in points, with a 95% confidence interval.

| Config | Base recall@5 | New recall@5 | Change | 95% CI |
| --- | --- | --- | --- | --- |
| fixed/bm25 | 77.0 | 82.0 | +5.0 | -7.0 to +17.0 |
| fixed/bm25+rerank | 83.0 | 87.0 | +4.0 | +0.0 to +10.0 |
| fixed/dense-lsa | 71.0 | 73.0 | +2.0 | -4.0 to +10.0 |
| fixed/dense-lsa+rerank | 83.0 | 89.0 | +6.0 | +0.0 to +14.0 |
| fixed/dense-minilm | 69.0 | 69.0 | +0.0 | +0.0 to +0.0 |
| fixed/dense-minilm+rerank | 81.0 | 81.0 | +0.0 | +0.0 to +0.0 |
| fixed/hybrid | 75.0 | 81.0 | +6.0 | -2.0 to +14.0 |
| fixed/hybrid+rerank | 89.0 | 89.0 | +0.0 | +0.0 to +0.0 |
| heading/bm25 | 77.0 | 79.0 | +2.0 | -6.0 to +10.0 |
| heading/bm25+rerank | 87.0 | 83.0 | -4.0 | -10.0 to +0.0 |
| heading/dense-lsa | 73.0 | 81.0 | +8.0 | -2.0 to +18.0 |
| heading/dense-lsa+rerank | 89.0 | 87.0 | -2.0 | -8.0 to +4.0 |
| heading/dense-minilm | 73.0 | 73.0 | +0.0 | +0.0 to +0.0 |
| heading/dense-minilm+rerank | 79.0 | 79.0 | +0.0 | +0.0 to +0.0 |
| heading/hybrid | 79.0 | 79.0 | +0.0 | -6.0 to +6.0 |
| heading/hybrid+rerank | 85.0 | 79.0 | -6.0 | -12.0 to +0.0 |

Refusals (all 60 questions, threshold 0.35):

| Config | Refusal acc (base) | Refusal acc (stemmed) | Missed refusals of 10 (base / stemmed) |
| --- | --- | --- | --- |
| fixed/hybrid+rerank (served) | 93.3 | 90.0 | 4 / 6 |
| fixed/bm25 | 90.0 | 88.3 | 6 / 7 |
| heading/bm25 | 88.3 | 90.0 | 6 / 6 |

## What was tested

`stemming.patch` adds a tiny suffix stripper to `text.tokenize` ("requests" -> "request",
"cancelled" -> "cancel", "running" -> "run"). Because `tokenize` is shared, it changes BM25, the
LSA vectorizer, the refusal confidence score and the mock answerer at once. MiniLM and the
cross-encoder use raw text and are unaffected, so pure-MiniLM rows are unchanged by construction.

## Findings

- **First-stage lexical retrieval improves a little**: +2 to +8 points for BM25, LSA and hybrid
  without reranking, but every interval includes zero.
- **Reranked configs get both better and worse** (+6 to -6). Stemming changes which 20
  candidates reach the cross-encoder; a plausible but unverified explanation for the heading-aware
  drops is that stem matches favour long sections that share vocabulary without the answer.
- **Refusals get worse for the served config** (4 to 6 missed refusals): stemming raises the
  word coverage of unanswerable questions too, which is the opposite of what the refusal
  heuristic needs.
- **The served config's recall@5 does not change** (89.0 either way).

## Decision

Not adopted. No gain where it matters (the served config), a measurable loss in refusal
behaviour, and a wider blast radius than its benefit. The code is kept only as `stemming.patch`
so the experiment is reproducible without leaving unused code in the package.

## Reproduce

```bash
git apply evals/experiments/stemming/stemming.patch
```

```bash
python -m evals.run_eval --output-dir /tmp/stemmed && python -m evals.compare evals/results.json /tmp/stemmed/results.json
```

```bash
git apply -R evals/experiments/stemming/stemming.patch
```
