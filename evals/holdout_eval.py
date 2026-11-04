"""Held-out check: does the refusal threshold tuned on the main gold set generalise?

`evals/holdout.jsonl` has 20 questions written after the threshold was frozen and never
used for tuning: 10 answerable paraphrases and 10 *on-topic* unanswerable questions that use
only corpus vocabulary (the hard case for the IDF-coverage heuristic).

Usage: python -m evals.holdout_eval   (writes evals/experiments/holdout/README.md)
"""

import json
from pathlib import Path
from typing import Any

from evals.run_eval import run
from rag_eval_lab.answering import DEFAULT_THRESHOLD
from rag_eval_lab.gold import load_gold
from rag_eval_lab.llm import MockProvider
from rag_eval_lab.pipeline import DEFAULT_CONFIG, Config

EVALS_DIR = Path(__file__).resolve().parent
HOLDOUT_PATH = EVALS_DIR / "holdout.jsonl"
OUTPUT = EVALS_DIR / "experiments" / "holdout" / "README.md"
RESULTS = OUTPUT.parent / "results.json"
CONFIGS = [DEFAULT_CONFIG, Config("fixed", "bm25"), Config("heading", "bm25")]


def render(results: dict[str, Any]) -> str:
    lines = [
        "# Held-out check of the refusal threshold",
        "",
        f"20 questions never used for tuning (`evals/holdout.jsonl`), threshold frozen at"
        f" {DEFAULT_THRESHOLD}. Mock mode.",
        "",
        "| Config | Refusal acc | False refusals (of 10) | Missed refusals (of 10) | Recall@5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, data in results["configs"].items():
        s = data["summary"]
        lines.append(
            f"| {name} | {100 * s['refusal_accuracy']:.0f} | {s['false_refusals']}"
            f" | {s['missed_refusals']} | {100 * s['recall@5']:.0f} |"
        )
    served = results["configs"][DEFAULT_CONFIG.name]
    lines += [
        "",
        f"Threshold sweep for {DEFAULT_CONFIG.name} on the held-out questions:",
        "",
        "| Threshold | Refusal acc | False refusals | Missed refusals |",
        "| --- | --- | --- | --- |",
    ]
    for row in served["refusal_sweep"]:
        lines.append(
            f"| {row['threshold']} | {100 * row['refusal_accuracy']:.0f}"
            f" | {row['false_refusals']} | {row['missed_refusals']} |"
        )
    lines += ["", "Per-question confidence for the served config:", ""]
    lines += ["| Id | Answerable | Confidence | Refused |", "| --- | --- | --- | --- |"]
    for qid, row in served["per_question"].items():
        lines.append(
            f"| {qid} | {row['answerable']} | {row['confidence']:.2f} | {row['refused']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = run(CONFIGS, MockProvider(), gold=load_gold(HOLDOUT_PATH))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    # Keep the hand-written findings below the generated tables.
    findings = OUTPUT.read_text().partition("\n## Findings")[2] if OUTPUT.exists() else ""
    OUTPUT.write_text(render(results) + ("\n## Findings" + findings if findings else ""))
    print(render(results))


if __name__ == "__main__":
    main()
