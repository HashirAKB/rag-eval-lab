"""Experiment: which signal should decide "not in the docs"?

Compares three refusal rules on the served (reranked) config:
  coverage  refuse if IDF-weighted word coverage < 0.35 (the current rule)
  reranker  refuse if the cross-encoder's top score < t
  either    refuse if either signal is low
The reranker threshold t is chosen on the main gold set and then *frozen* for the
held-out set, so the held-out column is an honest estimate.

Usage (after python -m evals.run_eval and python -m evals.holdout_eval):
    python -m evals.refusal_signals   (writes evals/experiments/refusal_signals/README.md)
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_eval_lab.answering import DEFAULT_THRESHOLD
from rag_eval_lab.pipeline import DEFAULT_CONFIG

EVALS_DIR = Path(__file__).resolve().parent
MAIN_RESULTS = EVALS_DIR / "results.json"
HOLDOUT_RESULTS = EVALS_DIR / "experiments" / "holdout" / "results.json"
OUTPUT = EVALS_DIR / "experiments" / "refusal_signals" / "README.md"
CANDIDATES = [x / 4 for x in range(-8, 13)]  # -2.0 .. 3.0 in steps of 0.25

Rule = Callable[[dict[str, Any]], bool]  # row -> refuse?


def outcomes(rows: list[dict[str, Any]], refuse: Rule) -> dict[str, float]:
    decisions = [(refuse(r), r["answerable"]) for r in rows]
    return {
        "accuracy": sum(ref != ans for ref, ans in decisions) / len(decisions),
        "false_refusals": sum(ref and ans for ref, ans in decisions),
        "missed_refusals": sum(not ref and not ans for ref, ans in decisions),
    }


def choose_threshold(rows: list[dict[str, Any]]) -> float:
    """The lowest candidate threshold with the best refusal accuracy on these rows."""
    return max(
        CANDIDATES, key=lambda t: (outcomes(rows, lambda r: r["top_score"] < t)["accuracy"], -t)
    )


def rules(reranker_threshold: float) -> dict[str, Rule]:
    def low_coverage(r: dict[str, Any]) -> bool:
        return bool(r["confidence"] < DEFAULT_THRESHOLD)

    def low_reranker(r: dict[str, Any]) -> bool:
        return bool(r["top_score"] < reranker_threshold)

    return {
        f"coverage < {DEFAULT_THRESHOLD}": low_coverage,
        f"reranker < {reranker_threshold}": low_reranker,
        "either": lambda r: low_coverage(r) or low_reranker(r),
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(
        json.loads(path.read_text())["configs"][DEFAULT_CONFIG.name]["per_question"].values()
    )


def main() -> None:
    main_rows, holdout_rows = _rows(MAIN_RESULTS), _rows(HOLDOUT_RESULTS)
    threshold = choose_threshold(main_rows)
    lines = [
        "# Experiment: refusal signals",
        "",
        f"Served config `{DEFAULT_CONFIG.name}`. Reranker threshold chosen on the main set"
        f" ({threshold}) and frozen for the held-out set. Cells: refusal accuracy (false refusals,"
        " missed refusals).",
        "",
        "| Rule | Main set (60 questions) | Held-out (20 questions) |",
        "| --- | --- | --- |",
    ]
    for name, rule in rules(threshold).items():
        cells = []
        for rows in (main_rows, holdout_rows):
            o = outcomes(rows, rule)
            cells.append(
                f"{100 * o['accuracy']:.0f}% ({o['false_refusals']}, {o['missed_refusals']})"
            )
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    findings = OUTPUT.read_text().partition("\n## Findings")[2] if OUTPUT.exists() else ""
    OUTPUT.write_text("\n".join(lines) + "\n" + ("\n## Findings" + findings if findings else ""))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
