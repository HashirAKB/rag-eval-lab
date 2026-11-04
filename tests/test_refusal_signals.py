from evals.refusal_signals import choose_threshold, outcomes, rules

ROWS = [
    {"answerable": True, "confidence": 0.9, "top_score": 2.0},
    {"answerable": True, "confidence": 0.3, "top_score": 1.5},  # low coverage, strong reranker
    {"answerable": False, "confidence": 0.6, "top_score": -1.0},  # on topic, reranker says no
    {"answerable": False, "confidence": 0.1, "top_score": 0.5},
]


def test_outcomes_counts_false_and_missed_refusals() -> None:
    result = outcomes(ROWS, lambda r: r["confidence"] < 0.35)
    assert result == {"accuracy": 0.5, "false_refusals": 1, "missed_refusals": 1}


def test_choose_threshold_picks_the_lowest_best_threshold() -> None:
    assert choose_threshold(ROWS) == 0.75  # first candidate above 0.5 and below 1.5


def test_either_rule_refuses_when_any_signal_is_low() -> None:
    either = rules(0.75)["either"]
    assert [either(r) for r in ROWS] == [False, True, True, True]
