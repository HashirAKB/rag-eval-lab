"""Retrieval and answer metrics.

`evidence` is a list with one set per evidence quote: the chunk ids that hold that
quote (see gold.relevant_chunk_sets). Most questions have exactly one quote.
"""

import math
import random


def recall_at_k(ranked: list[str], evidence: list[set[str]], k: int) -> float:
    """Fraction of evidence quotes held by at least one of the top-k chunks."""
    top = set(ranked[:k])
    return sum(bool(top & chunk_ids) for chunk_ids in evidence) / len(evidence)


def reciprocal_rank(ranked: list[str], evidence: list[set[str]]) -> float:
    """1 / rank of the first chunk that holds any evidence quote (0 if none does)."""
    relevant = set().union(*evidence)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def _dcg(ranked: list[str], evidence: list[set[str]]) -> float:
    """A chunk gains one point per evidence quote it is the *first* to cover."""
    covered: set[int] = set()
    score = 0.0
    for rank, chunk_id in enumerate(ranked, start=1):
        new = {i for i, ids in enumerate(evidence) if chunk_id in ids and i not in covered}
        covered |= new
        score += len(new) / math.log2(rank + 1)
    return score


def _ideal_ranking(evidence: list[set[str]]) -> list[str]:
    """Best possible order: greedily pick the chunk covering the most uncovered quotes."""
    uncovered = set(range(len(evidence)))
    ranking: list[str] = []
    candidates = sorted(set().union(*evidence))
    while uncovered:
        best = max(candidates, key=lambda c: sum(c in evidence[i] for i in uncovered))
        ranking.append(best)
        uncovered -= {i for i in uncovered if best in evidence[i]}
    return ranking


def ndcg_at_k(ranked: list[str], evidence: list[set[str]], k: int) -> float:
    ideal = _dcg(_ideal_ranking(evidence)[:k], evidence)
    return _dcg(ranked[:k], evidence) / ideal


def citation_accuracy(citations: list[str], relevant: set[str]) -> float:
    """Fraction of cited chunks that hold gold evidence (0 when nothing is cited)."""
    if not citations:
        return 0.0
    return sum(c in relevant for c in citations) / len(citations)


def bootstrap_ci(
    values: list[float], samples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`.

    Resample the questions with replacement many times; the spread of the resampled
    means shows how much the score depends on which questions happened to be asked.
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    tail = (1 - confidence) / 2
    return means[int(tail * samples)], means[int((1 - tail) * samples) - 1]
