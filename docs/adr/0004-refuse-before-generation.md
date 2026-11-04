# ADR 0004: Decide refusals before generation, from IDF-weighted term coverage

- Status: accepted
- Date: 2026-11-04

## Context

The system must say "not in the docs" when retrieval is weak instead of letting the LLM improvise.
Retriever scores cannot be compared across configurations (BM25 is unbounded, cosine is in
[-1, 1], RRF scores are tiny, cross-encoder scores are logits), so one threshold on "the top score"
would need re-tuning per configuration.

## Decision

`answering.ConfidenceScorer` computes, for the best of the top 5 chunks, the IDF-weighted share of
the question's words that appear in it. Words absent from the corpus get the highest IDF, so a
question about "Kubernetes" scores low everywhere. The answerer refuses below a threshold (0.35,
chosen from the sweep in `evals/results.md`). A live LLM can additionally refuse by returning
`NOT_IN_DOCS`, and an answer without a valid citation is also treated as a refusal.

## Consequences

- Same behaviour in mock and live mode, for every retriever, and cheap: no LLM call is spent on a
  question that will be refused.
- Exact word matching penalises paraphrases (no stemming or synonyms), and on-topic questions
  whose specific answer is missing still pass. The eval shows this: 4 of 10 unanswerable questions
  are answered by the winning config.
- The threshold was tuned on the eval set, which the results state explicitly.

## Amendment (2026-11-04): add the cross-encoder score for reranked configs

A held-out set of on-topic unanswerable questions showed that word coverage measures topicality,
not answerability: refusal accuracy for the served config fell from 93% (tuning set) to 55%.
Reranked configs now also refuse when the cross-encoder's top score is below 0.25 (a logit chosen
on the main set). Held-out refusal accuracy rose to 75%. Non-reranked configs keep the coverage
rule. Evidence: `evals/experiments/holdout`, `evals/experiments/refusal_signals`.
