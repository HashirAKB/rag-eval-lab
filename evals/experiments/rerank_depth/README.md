# Experiment: reranker depth

First stage `fixed/hybrid`, cross-encoder over the top N candidates, 50 answerable questions. Latency is the mean end-to-end search time per query on the build machine's CPU (single process, warm models).

| Candidates reranked | Recall@5 | MRR | ms/query |
| --- | --- | --- | --- |
| 5 | 75.0 | 56.9 | 150 |
| 10 | 83.0 | 61.4 | 260 |
| 20 | 89.0 | 64.8 | 535 |
| 40 | 87.0 | 64.3 | 1071 |

## Findings

- **Depth 5 cannot change recall@5**: reranking five candidates only reorders the same five chunks,
  so recall@5 equals the first stage's (75.0); only MRR moves. It is a sanity check on the harness.
- **Latency is linear in depth** (one cross-encoder pass per candidate): about 27 ms per candidate
  on this CPU, on top of about 15 ms for the hybrid first stage.
- **20 is the knee.** Going from 10 to 20 adds 6 points of recall@5 for about 275 ms; going from 20
  to 40 costs another 535 ms and loses 2 points, because more candidates also means more
  plausible distractors for the cross-encoder to promote. On 50 questions a 2-point loss is noise,
  so the safe reading is "no gain past 20".
- **Recommendation**: keep the default depth of 20. If latency matters more than the last few
  points, depth 10 keeps most of the reranking gain (83.0 vs 75.0 without reranking) at half the cost.

## Reproduce

```bash
python -m evals.rerank_depth
```
