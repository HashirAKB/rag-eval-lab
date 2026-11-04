"""Run the ablation: every chunking x retriever x reranker configuration on the gold set.

Usage:
    python -m evals.run_eval                       # all configs, writes evals/results.{json,md}
    python -m evals.run_eval --configs heading/bm25,heading/hybrid --output-dir /tmp/out

Mock mode (no ANTHROPIC_API_KEY) is fully offline and deterministic apart from
latency numbers. With a key, answers and the faithfulness judge use the live API.
"""

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from rag_eval_lab.answering import DEFAULT_THRESHOLD, Answerer, ConfidenceScorer
from rag_eval_lab.chunking import STRATEGIES, chunk_corpus
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.gold import GoldItem, load_gold, relevant_chunk_sets
from rag_eval_lab.llm import LLMProvider, get_provider
from rag_eval_lab.metrics import (
    bootstrap_ci,
    citation_accuracy,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval_lab.pipeline import ALL_CONFIGS, Config, build_retrievers
from rag_eval_lab.retrievers import Retriever

EVALS_DIR = Path(__file__).resolve().parent
DEPTH = 20  # how many chunks we retrieve per question for ranking metrics
ANSWER_K = 5  # how many of those the answerer reads
THRESHOLDS = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
ANALYSIS_START, ANALYSIS_END = "<!-- analysis:start -->", "<!-- analysis:end -->"


def evaluate_config(
    retriever: Retriever,
    evidence_by_id: dict[str, list[set[str]]],
    gold: list[GoldItem],
    answerer: Answerer,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Per-question results for one configuration."""
    per_question: dict[str, dict[str, Any]] = {}
    for item in gold:
        started = time.perf_counter()
        hits = retriever.search(item.question, DEPTH)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [h.chunk.chunk_id for h in hits]
        answer = answerer.answer_from_hits(item.question, hits)
        row: dict[str, Any] = {
            "answerable": item.answerable,
            "refused": answer.refused,
            "confidence": round(answer.confidence, 4),
            "latency_ms": latency_ms,
        }
        if item.answerable:
            evidence = evidence_by_id[item.id]
            relevant = set().union(*evidence)
            row |= {
                "first_relevant_rank": next(
                    (rank for rank, cid in enumerate(ranked, 1) if cid in relevant), None
                ),
                "recall@1": recall_at_k(ranked, evidence, 1),
                "recall@5": recall_at_k(ranked, evidence, 5),
                "recall@10": recall_at_k(ranked, evidence, 10),
                "mrr": reciprocal_rank(ranked, evidence),
                "ndcg@10": ndcg_at_k(ranked, evidence, 10),
            }
            if not answer.refused:
                row["citation_accuracy"] = citation_accuracy(answer.citations, relevant)
                row["faithfulness"] = provider.faithfulness(
                    answer.text, [h.chunk for h in answer.hits]
                )
        per_question[item.id] = row
    return per_question


def summarize(per_question: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(per_question.values())
    answerable = [r for r in rows if r["answerable"]]
    answered = [r for r in answerable if "citation_accuracy" in r]
    summary: dict[str, Any] = {
        metric: mean(r[metric] for r in answerable)
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")
    }
    summary["recall@5_ci95"] = bootstrap_ci([r["recall@5"] for r in answerable])
    summary["citation_accuracy"] = (
        mean(r["citation_accuracy"] for r in answered) if answered else 0.0
    )
    summary["faithfulness"] = mean(r["faithfulness"] for r in answered) if answered else 0.0
    summary["refusal_accuracy"] = mean(r["refused"] != r["answerable"] for r in rows)
    summary["false_refusals"] = sum(r["refused"] for r in answerable)
    summary["missed_refusals"] = sum(not r["refused"] for r in rows if not r["answerable"])
    summary["latency_ms"] = mean(r["latency_ms"] for r in rows)
    return summary


def threshold_sweep(per_question: dict[str, dict[str, Any]]) -> list[dict[str, float]]:
    """Refusal behaviour at other thresholds, from the confidences already recorded."""
    rows = list(per_question.values())
    sweep = []
    for threshold in THRESHOLDS:
        refused = [r["confidence"] < threshold for r in rows]
        sweep.append(
            {
                "threshold": threshold,
                "refusal_accuracy": mean(
                    ref != r["answerable"] for ref, r in zip(refused, rows, strict=True)
                ),
                "false_refusals": sum(
                    ref and r["answerable"] for ref, r in zip(refused, rows, strict=True)
                ),
                "missed_refusals": sum(
                    not ref and not r["answerable"] for ref, r in zip(refused, rows, strict=True)
                ),
            }
        )
    return sweep


def run(configs: list[Config], provider: LLMProvider) -> dict[str, Any]:
    docs = load_corpus()
    docs_by_path = {d.path: d for d in docs}
    gold = load_gold()
    results: dict[str, Any] = {
        "mode": provider.name,
        "faithfulness_method": provider.faithfulness_method,
        "refusal_threshold": DEFAULT_THRESHOLD,
        "configs": {},
    }
    for strategy in STRATEGIES:
        strategy_configs = [c for c in configs if c.chunking == strategy]
        if not strategy_configs:
            continue
        chunks = chunk_corpus(docs, strategy)
        evidence_by_id = {g.id: relevant_chunk_sets(g, chunks, docs_by_path) for g in gold}
        scorer = ConfidenceScorer(chunks)
        retrievers = build_retrievers(chunks, strategy_configs)
        for config in strategy_configs:
            print(f"evaluating {config.name} ...", flush=True)
            answerer = Answerer(retrievers[config], provider, scorer, k=ANSWER_K)
            per_question = evaluate_config(
                retrievers[config], evidence_by_id, gold, answerer, provider
            )
            results["configs"][config.name] = {
                "summary": summarize(per_question),
                "refusal_sweep": threshold_sweep(per_question),
                "avg_chunk_chars": round(mean(len(c.text) for c in chunks)),
                "num_chunks": len(chunks),
                "per_question": per_question,
            }
    return results


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def render_markdown(results: dict[str, Any], analysis: str) -> str:
    configs = results["configs"]
    ranked = sorted(configs, key=lambda n: -configs[n]["summary"]["recall@5"])
    winner = ranked[0]
    lines = [
        "# Evaluation results",
        "",
        f"- Mode: **{results['mode']}** (faithfulness: {results['faithfulness_method']})",
        "- Gold set: 60 questions (50 answerable, 10 unanswerable), see `evals/GOLD.md`",
        f"- Retrieval depth {DEPTH}; the answerer reads the top {ANSWER_K};"
        f" refusal threshold {results['refusal_threshold']}",
        "- Retrieval metrics are averaged over the 50 answerable questions, in percent.",
        "- Generated by `python -m evals.run_eval`; latency varies by machine.",
        "",
        "## Ablation (sorted by recall@5)",
        "",
        "| Config | R@1 | R@5 (95% CI) | R@10 | MRR | nDCG@10 | Cite acc | Refusal acc"
        " | Faithful | ms/query |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ranked:
        s = configs[name]["summary"]
        low, high = s["recall@5_ci95"]
        lines.append(
            f"| {name} | {_pct(s['recall@1'])} | {_pct(s['recall@5'])} ({_pct(low)}-{_pct(high)})"
            f" | {_pct(s['recall@10'])} | {_pct(s['mrr'])} | {_pct(s['ndcg@10'])}"
            f" | {_pct(s['citation_accuracy'])} | {_pct(s['refusal_accuracy'])}"
            f" | {_pct(s['faithfulness'])} | {s['latency_ms']:.1f} |"
        )

    lines += [
        "",
        "## Chunking strategies",
        "",
        "| Chunking | Chunks | Avg chars |",
        "| --- | --- | --- |",
    ]
    for strategy in STRATEGIES:
        match = next((configs[n] for n in configs if n.startswith(f"{strategy}/")), None)
        if match:
            lines.append(f"| {strategy} | {match['num_chunks']} | {match['avg_chunk_chars']} |")

    # Paired comparison: per-question recall@5 differences between the winner and others.
    lines += [
        "",
        f"## Is the winner ({winner}) really better?",
        "",
        "Paired bootstrap over questions: mean difference in recall@5 (winner minus other),"
        " with a 95% confidence interval. An interval that includes 0 means the gap could be"
        " noise from this small gold set.",
        "",
        "| Other config | Difference (points) | 95% CI |",
        "| --- | --- | --- |",
    ]
    winner_rows = configs[winner]["per_question"]
    for name in ranked[1:]:
        other_rows = configs[name]["per_question"]
        diffs = [
            winner_rows[q]["recall@5"] - other_rows[q]["recall@5"]
            for q in winner_rows
            if winner_rows[q]["answerable"]
        ]
        low, high = bootstrap_ci(diffs)
        lines.append(f"| {name} | {_pct(mean(diffs))} | {_pct(low)} to {_pct(high)} |")

    lines += [
        "",
        f"## Refusal threshold sweep ({winner})",
        "",
        "The threshold was picked by looking at this table, so refusal accuracy at the chosen"
        " threshold is optimistic (tuned on the evaluation set).",
        "",
        "| Threshold | Refusal acc | False refusals (of 50) | Missed refusals (of 10) |",
        "| --- | --- | --- | --- |",
    ]
    for row in configs[winner]["refusal_sweep"]:
        lines.append(
            f"| {row['threshold']} | {_pct(row['refusal_accuracy'])} | {row['false_refusals']}"
            f" | {row['missed_refusals']} |"
        )

    lines += ["", "## Analysis", "", ANALYSIS_START, analysis.strip(), ANALYSIS_END, ""]
    return "\n".join(lines)


def _existing_analysis(path: Path) -> str:
    """Keep the hand-written analysis when the tables are regenerated."""
    if not path.exists():
        return "_Not written yet._"
    match = re.search(
        re.escape(ANALYSIS_START) + r"(.*)" + re.escape(ANALYSIS_END), path.read_text(), re.DOTALL
    )
    return match.group(1) if match else "_Not written yet._"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", help="comma-separated config names (default: all)")
    parser.add_argument("--output-dir", type=Path, default=EVALS_DIR)
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.configs:
        wanted = set(args.configs.split(","))
        configs = [c for c in ALL_CONFIGS if c.name in wanted]
        unknown = wanted - {c.name for c in configs}
        if unknown:
            parser.error(f"unknown configs: {sorted(unknown)}")

    results = run(configs, get_provider())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = args.output_dir / "results.json", args.output_dir / "results.md"
    json_path.write_text(json.dumps(results, indent=1) + "\n")
    md_path.write_text(render_markdown(results, _existing_analysis(md_path)))
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
