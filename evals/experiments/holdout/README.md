# Held-out check of the refusal threshold

20 questions not used to tune any threshold (`evals/holdout.jsonl`): word coverage threshold 0.35; reranked configs also refuse below a cross-encoder score of 0.25 (chosen on the main set). Mock mode.

| Config | Refusal acc | False refusals (of 10) | Missed refusals (of 10) | Recall@5 |
| --- | --- | --- | --- | --- |
| fixed/hybrid+rerank | 75 | 2 | 3 | 90 |
| fixed/bm25 | 60 | 0 | 8 | 60 |
| heading/bm25 | 55 | 0 | 9 | 80 |

Threshold sweep for fixed/hybrid+rerank on the held-out questions:

| Threshold | Refusal acc | False refusals | Missed refusals |
| --- | --- | --- | --- |
| 0.25 | 75 | 1 | 4 |
| 0.3 | 70 | 2 | 4 |
| 0.35 | 75 | 2 | 3 |
| 0.4 | 75 | 2 | 3 |
| 0.45 | 70 | 4 | 2 |
| 0.5 | 75 | 4 | 1 |

Per-question confidence for the served config:

| Id | Answerable | Confidence | Refused |
| --- | --- | --- | --- |
| h01 | True | 0.51 | False |
| h02 | True | 0.58 | False |
| h03 | True | 0.59 | False |
| h04 | True | 0.51 | False |
| h05 | True | 0.40 | False |
| h06 | True | 1.00 | False |
| h07 | True | 0.27 | True |
| h08 | True | 1.00 | False |
| h09 | True | 0.41 | False |
| h10 | True | 0.47 | True |
| h11 | False | 0.50 | False |
| h12 | False | 0.34 | True |
| h13 | False | 0.59 | False |
| h14 | False | 0.68 | True |
| h15 | False | 0.24 | True |
| h16 | False | 0.38 | True |
| h17 | False | 0.41 | True |
| h18 | False | 0.44 | True |
| h19 | False | 0.42 | False |
| h20 | False | 0.41 | True |

## Findings

- **Word coverage alone does not generalise to hard negatives.** In the first run of this set
  (before the reranker gate existed) the served config scored 55%, missing 8 of 10 unanswerable
  questions. These questions ("Which TCP port does the specification assign to Streamable HTTP
  servers?") use only words the corpus contains, so their coverage (0.38 to 0.68) overlaps with
  answerable questions (0.27 to 1.00). No coverage threshold separates them; the BM25 rows above
  still show this (55 to 60%).
- **The signal measured topicality, not answerability.** Coverage says "the docs talk about ports";
  it cannot say "the docs never state a port number". That needs a model that reads the passage
  against the question, which led to the reranker gate (`evals/experiments/refusal_signals`).
- **With the reranker gate the served config reaches 75%** (2 false refusals, 3 missed). That is
  better, but well below the 96.7% measured on the tuning set: in-sample refusal numbers are
  optimistic.
- **This set is no longer pristine.** Its first result motivated the reranker gate, so it has
  influenced a design decision; a new held-out set is needed for the next unbiased estimate.
- **Retrieval generalises reasonably.** Recall@5 on the 10 new answerable questions is 90% for the
  served config (89% on the main set), with a very wide interval on 10 questions.

## Reproduce

```bash
python -m evals.holdout_eval
```
