# Experiment: refusal signals

Served config `fixed/hybrid+rerank`. Reranker threshold chosen on the main set (0.25) and frozen for the held-out set. Cells: refusal accuracy (false refusals, missed refusals).

| Rule | Main set (60 questions) | Held-out (20 questions) |
| --- | --- | --- |
| coverage < 0.35 | 93% (0, 4) | 55% (1, 8) |
| reranker < 0.25 | 93% (0, 4) | 75% (1, 4) |
| either | 97% (0, 2) | 75% (2, 3) |

## Findings

- **The cross-encoder score generalises better than word coverage.** Both rules score 93% on the
  main set, but on held-out questions the reranker rule keeps 75% while coverage falls to 55%. The
  cross-encoder reads the question and passage together, so it can tell "this passage is about
  ports" from "this passage states the port".
- **Combining them ("either") is best in-sample** (97%) and ties on held-out (75%), trading one
  more false refusal for one fewer missed refusal. It is the adopted rule for reranked configs
  (`pipeline.build_answerer`, `RERANK_MIN_SCORE = 0.25`), because it also keeps a refusal signal
  when the reranker is confident about an off-topic passage.
- **Limits.** 20 held-out questions is very few; the threshold is a raw logit of one specific
  cross-encoder and must be re-chosen if the model changes; non-reranked configs still rely on
  coverage alone.

## Reproduce

```bash
python -m evals.run_eval && python -m evals.holdout_eval && python -m evals.refusal_signals
```
