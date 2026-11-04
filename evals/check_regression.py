"""Regression gate: fail if any configuration's recall@5 drops below the committed baseline.

Usage:
    python -m evals.check_regression                     # compare evals/results.json to baseline
    python -m evals.check_regression --results out/results.json
    python -m evals.check_regression --update-baseline   # accept current results as the baseline
"""

import argparse
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
BASELINE_PATH = EVALS_DIR / "baseline.json"
TOLERANCE = 0.03  # 3 points of recall@5
METRIC = "recall@5"


def find_regressions(
    current: dict[str, float], baseline: dict[str, float], tolerance: float = TOLERANCE
) -> list[str]:
    """Human-readable problems; an empty list means the gate passes."""
    problems = []
    for config, expected in sorted(baseline.items()):
        if config not in current:
            problems.append(f"{config}: missing from results")
        elif current[config] < expected - tolerance:
            problems.append(
                f"{config}: {METRIC} {current[config]:.3f} < baseline {expected:.3f}"
                f" - {tolerance:.2f}"
            )
    return problems


def scores_from_results(results: dict[str, object]) -> dict[str, float]:
    configs = results["configs"]
    assert isinstance(configs, dict)
    return {name: data["summary"][METRIC] for name, data in configs.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=EVALS_DIR / "results.json")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    current = scores_from_results(json.loads(args.results.read_text()))
    if args.update_baseline:
        rounded = {name: round(score, 4) for name, score in sorted(current.items())}
        args.baseline.write_text(json.dumps(rounded, indent=2) + "\n")
        print(f"baseline updated: {args.baseline}")
        return 0

    problems = find_regressions(current, json.loads(args.baseline.read_text()))
    for problem in problems:
        print(f"REGRESSION {problem}")
    if problems:
        return 1
    print(f"OK: {METRIC} within {TOLERANCE:.2f} of baseline for {len(current)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
