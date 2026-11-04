# Held-out check of the refusal threshold

20 questions never used for tuning (`evals/holdout.jsonl`), threshold frozen at 0.35. Mock mode.

| Config | Refusal acc | False refusals (of 10) | Missed refusals (of 10) | Recall@5 |
| --- | --- | --- | --- | --- |
| fixed/hybrid+rerank | 55 | 1 | 8 | 90 |
| fixed/bm25 | 60 | 0 | 8 | 60 |
| heading/bm25 | 55 | 0 | 9 | 80 |

Threshold sweep for fixed/hybrid+rerank on the held-out questions:

| Threshold | Refusal acc | False refusals | Missed refusals |
| --- | --- | --- | --- |
| 0.25 | 55 | 0 | 9 |
| 0.3 | 50 | 1 | 9 |
| 0.35 | 55 | 1 | 8 |
| 0.4 | 60 | 1 | 7 |
| 0.45 | 70 | 3 | 3 |
| 0.5 | 70 | 4 | 2 |

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
| h10 | True | 0.47 | False |
| h11 | False | 0.50 | False |
| h12 | False | 0.34 | True |
| h13 | False | 0.59 | False |
| h14 | False | 0.68 | False |
| h15 | False | 0.24 | True |
| h16 | False | 0.38 | False |
| h17 | False | 0.41 | False |
| h18 | False | 0.44 | False |
| h19 | False | 0.42 | False |
| h20 | False | 0.41 | False |

## Findings

- **The refusal threshold does not generalise to hard negatives.** On the tuning set the served
  config reached 93% refusal accuracy; here it is 55%, missing 8 of 10 unanswerable questions.
  These questions ("Which TCP port does the specification assign to Streamable HTTP servers?")
  use only words the corpus contains, so IDF-weighted word coverage is as high as for answerable
  questions (0.38 to 0.68 vs 0.27 to 1.00). No threshold separates them: raising it to 0.45 catches
  7 of 10 negatives but falsely refuses 3 of 10 answerable questions.
- **The signal measures topicality, not answerability.** Coverage says "the docs talk about
  ports, tools and subscriptions"; it cannot say "the docs never state a port number". Detecting
  that needs a model that reads the passage against the question: the live LLM's `NOT_IN_DOCS`
  path, or a cross-encoder or NLI answerability score.
- **Retrieval generalises reasonably.** Recall@5 on the 10 new answerable questions is 90% for the
  served config (in line with 89% on the main set, though 10 questions give a very wide interval).
- The main results' 93% refusal accuracy should be read as an optimistic, in-sample number; the
  README and SUMMARY say so.

## Reproduce

```bash
python -m evals.holdout_eval
```
