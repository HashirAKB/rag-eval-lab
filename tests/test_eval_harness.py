import json
from pathlib import Path

import pytest

from evals.check_regression import find_regressions, main
from evals.run_eval import render_markdown, run
from rag_eval_lab.llm import MockProvider
from rag_eval_lab.pipeline import Config


def test_gate_passes_within_tolerance_and_fails_on_a_four_point_drop() -> None:
    baseline = {"heading/bm25": 0.80, "fixed/bm25": 0.70}
    assert find_regressions({"heading/bm25": 0.78, "fixed/bm25": 0.70}, baseline) == []
    problems = find_regressions({"heading/bm25": 0.76, "fixed/bm25": 0.70}, baseline)
    assert len(problems) == 1 and problems[0].startswith("heading/bm25")


def test_gate_fails_when_a_baseline_config_is_missing() -> None:
    assert find_regressions({}, {"heading/bm25": 0.8}) == ["heading/bm25: missing from results"]


def test_gate_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = {"configs": {"heading/bm25": {"summary": {"recall@5": 0.75}}}}
    results_path, baseline_path = tmp_path / "results.json", tmp_path / "baseline.json"
    results_path.write_text(json.dumps(results))
    baseline_path.write_text(json.dumps({"heading/bm25": 0.80}))
    argv = ["check_regression", "--results", str(results_path), "--baseline", str(baseline_path)]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 1
    baseline_path.write_text(json.dumps({"heading/bm25": 0.77}))
    assert main() == 0


def test_run_produces_bounded_metrics_and_markdown() -> None:
    configs = [Config("heading", "bm25"), Config("fixed", "bm25")]
    results = run(configs, MockProvider())
    assert results["mode"] == "mock"
    assert "token-overlap" in results["faithfulness_method"]
    for name in ("heading/bm25", "fixed/bm25"):
        summary = results["configs"][name]["summary"]
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "refusal_accuracy"):
            assert 0.0 <= summary[metric] <= 1.0, (name, metric)
        assert summary["recall@1"] <= summary["recall@5"] <= summary["recall@10"]
        assert len(results["configs"][name]["per_question"]) == 60
    markdown = render_markdown(results, "Hand-written analysis.")
    assert "| heading/bm25 |" in markdown and "Hand-written analysis." in markdown
