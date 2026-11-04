# ADR 0003: Every LLM call goes through a provider interface with a deterministic mock

- Status: accepted
- Date: 2026-11-04

## Context

The project must build, test, run evals and demo with zero API keys, and CI must be
reproducible. LLM calls are slow, cost money and are not deterministic.

## Decision

`llm.LLMProvider` has two methods, `answer(question, contexts)` and
`faithfulness(answer, contexts)`. `MockProvider` (the default) answers extractively by quoting the
sentences with the most word overlap and citing their chunks, and scores faithfulness by token
overlap. `AnthropicProvider` is used only when `ANTHROPIC_API_KEY` is set. Every result file
records which provider and which faithfulness method produced it.

## Consequences

- Tests, evals and the demo are offline and deterministic; the CI gate can compare numbers exactly.
- Retrieval metrics are identical in both modes because retrieval does not use the LLM.
- Mock answer metrics (citation accuracy, faithfulness) exercise the pipeline but say nothing
  about live answer quality, and are labelled as such.
- The live code path is covered by unit tests with a fake client, not by real API calls.
