"""Compare two eval result files config by config, with a paired bootstrap on recall@5.

Usage:
    python -m evals.compare evals/results.json NEW/results.json [--strip +ctx] [--title T]

Each config in the new file is paired with the config of the same name in the base file
(after removing --strip from the name, e.g. "fixed+ctx/bm25" -> "fixed/bm25"). The
difference is computed per question, so both runs must use the same gold set.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from rag_eval_lab.metrics import bootstrap_ci

METRIC = "recall@5"


def compare(base_path: Path, new_path: Path, strip: str = "") -> list[dict[str, Any]]:
    base = json.loads(base_path.read_text())["configs"]
    new = json.loads(new_path.read_text())["configs"]
    rows = []
    for name, data in new.items():
        base_name = name.replace(strip, "") if strip else name
        if base_name not in base:
            continue
        base_rows, new_rows = base[base_name]["per_question"], data["per_question"]
        diffs = [
            new_rows[q][METRIC] - base_rows[q][METRIC]
            for q in new_rows
            if new_rows[q]["answerable"]
        ]
        low, high = bootstrap_ci(diffs)
        rows.append(
            {
                "config": name,
                "base": base[base_name]["summary"][METRIC],
                "new": data["summary"][METRIC],
                "diff": mean(diffs),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def render(rows: list[dict[str, Any]], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Paired bootstrap over the answerable questions: change in {METRIC} (new minus base),"
        " in points, with a 95% confidence interval.",
        "",
        f"| Config | Base {METRIC} | New {METRIC} | Change | 95% CI |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {100 * r['base']:.1f} | {100 * r['new']:.1f}"
            f" | {100 * r['diff']:+.1f} | {100 * r['ci_low']:+.1f} to {100 * r['ci_high']:+.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--strip", default="", help="text to remove from new config names")
    parser.add_argument("--title", default="Comparison")
    args = parser.parse_args()
    print(render(compare(args.base, args.new, args.strip), args.title), end="")


if __name__ == "__main__":
    main()
