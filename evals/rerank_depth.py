"""Experiment: how many first-stage candidates should the cross-encoder rerank?

More candidates can only raise the ceiling (the reranker cannot promote a chunk it never
sees) but every candidate costs one cross-encoder forward pass.

Usage: python -m evals.rerank_depth   (writes evals/experiments/rerank_depth/README.md)
"""

import time
from pathlib import Path
from statistics import mean
from typing import Any

from rag_eval_lab.chunking import chunk_corpus
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.gold import GoldItem, load_gold, relevant_chunk_sets
from rag_eval_lab.metrics import recall_at_k, reciprocal_rank
from rag_eval_lab.pipeline import Config, build_retrievers
from rag_eval_lab.retrievers import RerankingRetriever, Retriever

DEPTHS = (5, 10, 20, 40)
OUTPUT = Path(__file__).resolve().parent / "experiments" / "rerank_depth" / "README.md"


def sweep(
    base: Retriever,
    gold: list[GoldItem],
    evidence: dict[str, list[set[str]]],
    depths: tuple[int, ...],
) -> list[dict[str, Any]]:
    """recall@5, MRR and mean latency of `base` reranked at each depth (answerable items only)."""
    items = [g for g in gold if g.answerable]
    rows = []
    for depth in depths:
        reranker = RerankingRetriever(base, depth=depth)
        recalls, rrs, latencies = [], [], []
        for item in items:
            started = time.perf_counter()
            ranked = [h.chunk.chunk_id for h in reranker.search(item.question, 20)]
            latencies.append((time.perf_counter() - started) * 1000)
            recalls.append(recall_at_k(ranked, evidence[item.id], 5))
            rrs.append(reciprocal_rank(ranked, evidence[item.id]))
        rows.append(
            {"depth": depth, "recall@5": mean(recalls), "mrr": mean(rrs), "ms": mean(latencies)}
        )
    return rows


def main() -> None:
    config = Config("fixed", "hybrid")  # first stage of the served fixed/hybrid+rerank
    docs = load_corpus()
    chunks = chunk_corpus(docs, config.chunking)
    gold = load_gold()
    by_path = {d.path: d for d in docs}
    evidence = {g.id: relevant_chunk_sets(g, chunks, by_path) for g in gold}
    base = build_retrievers(chunks, [config])[config]
    base.search("warm up the models", 5)
    RerankingRetriever(base, depth=2).search("warm up the reranker", 1)
    rows = sweep(base, gold, evidence, DEPTHS)

    lines = [
        "# Experiment: reranker depth",
        "",
        f"First stage `{config.name}`, cross-encoder over the top N candidates, 50 answerable"
        " questions. Latency is the mean end-to-end search time per query on the build"
        " machine's CPU (single process, warm models).",
        "",
        "| Candidates reranked | Recall@5 | MRR | ms/query |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['depth']} | {100 * r['recall@5']:.1f} | {100 * r['mrr']:.1f} | {r['ms']:.0f} |"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
