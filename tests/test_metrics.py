import math

import pytest

from rag_eval_lab.metrics import (
    bootstrap_ci,
    citation_accuracy,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

ONE_QUOTE = [{"b"}]  # one evidence quote, covered by chunk "b"
TWO_QUOTES = [{"b"}, {"d", "e"}]  # quote 1 in "b"; quote 2 straddles "d" and "e"


def test_recall_counts_covered_evidence_quotes() -> None:
    assert recall_at_k(["a", "b", "c"], ONE_QUOTE, k=1) == 0.0
    assert recall_at_k(["a", "b", "c"], ONE_QUOTE, k=2) == 1.0
    assert recall_at_k(["a", "b", "e"], TWO_QUOTES, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], TWO_QUOTES, k=3) == 0.5


def test_reciprocal_rank_uses_first_relevant_chunk() -> None:
    assert reciprocal_rank(["a", "b"], ONE_QUOTE) == 0.5
    assert reciprocal_rank(["e", "b"], TWO_QUOTES) == 1.0
    assert reciprocal_rank(["a", "c"], ONE_QUOTE) == 0.0


def test_ndcg_single_quote_matches_textbook_binary_ndcg() -> None:
    assert ndcg_at_k(["b", "a"], ONE_QUOTE, k=5) == 1.0
    assert ndcg_at_k(["a", "b"], ONE_QUOTE, k=5) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["a", "c"], ONE_QUOTE, k=5) == 0.0


def test_ndcg_does_not_reward_a_second_chunk_for_the_same_quote() -> None:
    # "d" and "e" both cover quote 2; only the first one gains.
    ranking = ["d", "e", "b"]
    dcg = 1 / math.log2(2) + 0 + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranking, TWO_QUOTES, k=3) == pytest.approx(dcg / ideal)


def test_citation_accuracy() -> None:
    assert citation_accuracy(["b", "x"], {"b"}) == 0.5
    assert citation_accuracy([], {"b"}) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = [0.0, 1.0] * 25
    low, high = bootstrap_ci(values)
    assert low < 0.5 < high
    assert bootstrap_ci(values) == (low, high)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
