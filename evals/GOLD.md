# How the gold set was built

`gold.jsonl` holds 60 questions about the MCP documentation snapshot in `data/corpus/`:
50 answerable and 10 unanswerable.

Each answerable item stores the source document path and one or more **evidence quotes**
copied verbatim from the cleaned document text. For a given chunking strategy, the relevant
chunks are the ones that contain a quote (or, if a quote straddles a chunk boundary, every
chunk that overlaps it). This keeps one gold file valid for every chunking strategy.

## Process

1. **Heuristic candidates.** `scripts/draft_gold_candidates.py` lists prose sentences that look
   like checkable facts (normative MUST/SHOULD statements, definitions, numbers), a few per
   document, so candidates cover the whole corpus.
2. **Manual curation** (by the agent that built this project, not by a domain expert). For each
   kept candidate a question was written that a developer might plausibly ask, deliberately
   paraphrased away from the source wording where possible ("How long may a tool name be?"
   instead of "Tool names SHOULD be between 1 and 128 characters"). Some candidates were dropped
   for being trivial or ambiguous, and extra items were added from sections the heuristic missed
   (error handling, caching, security tutorials). The set was then rebalanced so that answerable
   items come from 31 of the 37 documents.
3. **Unanswerable items** are plausible-sounding questions about things the snapshot does not
   cover (Helm charts, gRPC, WebSockets, pricing) plus one absurd control question.
4. **Validation.** `tests/test_gold.py` checks that every quote occurs verbatim in its document,
   that the counts are right, and that every quote maps to at least one chunk in both strategies.

## Limitations (read before trusting any number)

- **Author bias.** The questions were written by the same agent that built the retrievers, after
  reading the documents. Even when paraphrased, they share vocabulary with the source text, which
  favours lexical retrieval (BM25). Real user questions are vaguer.
- **Small.** 50 answerable questions means one question is worth 2 points of recall@k. Results
  report 95% bootstrap confidence intervals for this reason.
- **Single-source labels.** Each question has one labelled source passage. If the corpus answers
  the same question elsewhere too (the MCP docs repeat themselves, for example between the
  `learn/` pages and the specification), a retriever that finds the other passage is scored as a
  miss. Recall numbers are therefore a lower bound.
- **Few, mixed-difficulty negatives.** Some unanswerable items contain a word that never appears
  in the corpus ("Kubernetes", "Helm"), which is exactly what the refusal heuristic keys on; others
  use only corpus vocabulary ("Which Python web framework does the specification require?") and
  are routinely missed. Ten negatives are too few to estimate refusal accuracy precisely.
- **No human review.** No MCP maintainer checked the questions or the chosen evidence.
