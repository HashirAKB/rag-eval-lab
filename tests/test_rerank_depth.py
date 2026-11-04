from conftest import requires_neural

from evals.rerank_depth import sweep
from rag_eval_lab.chunking import chunk_corpus
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.gold import load_gold, relevant_chunk_sets
from rag_eval_lab.retrievers import Bm25Retriever


@requires_neural
def test_sweep_reports_one_row_per_depth_with_bounded_metrics() -> None:
    docs = load_corpus()
    chunks = chunk_corpus(docs, "heading")
    gold = [g for g in load_gold() if g.answerable][:3]
    by_path = {d.path: d for d in docs}
    evidence = {g.id: relevant_chunk_sets(g, chunks, by_path) for g in gold}
    rows = sweep(Bm25Retriever(chunks), gold, evidence, (2, 5))
    assert [r["depth"] for r in rows] == [2, 5]
    for row in rows:
        assert 0.0 <= row["recall@5"] <= 1.0 and 0.0 <= row["mrr"] <= 1.0 and row["ms"] > 0
