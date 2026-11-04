# ADR 0001: Gold labels are evidence quotes, not chunk ids

- Status: accepted
- Date: 2026-11-04

## Context

The project compares two chunking strategies. A chunk id like `specification/server/tools#h007`
only exists under one strategy, so a gold set labelled with chunk ids would have to be labelled
twice, and every change to a chunker would silently invalidate the labels.

## Decision

Each gold item stores its source document path and one or more short quotes copied verbatim from
the cleaned document text. At evaluation time, `gold.relevant_chunk_sets` maps every quote to the
chunks that contain it (or, if the quote straddles a chunk boundary, to every chunk that overlaps
it). Metrics count evidence quotes covered, not chunks retrieved.

## Consequences

- One gold file evaluates any chunking strategy fairly, including ones added later.
- The cleaner (`corpus.clean_mdx`) must be deterministic: offsets are into its output.
- A test (`tests/test_gold.py`) catches quotes that stop matching after a corpus refresh.
- A question answered by several passages is only credited for the labelled one, so recall is a
  lower bound. Multi-source labels are listed as future work.
