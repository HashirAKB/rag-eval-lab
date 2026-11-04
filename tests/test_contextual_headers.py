import json
from pathlib import Path

from conftest import make_chunk

from evals.compare import compare
from rag_eval_lab.chunking import Chunk, contextual_text
from rag_eval_lab.pipeline import EXPERIMENT_CONFIGS, Config
from rag_eval_lab.retrievers import Bm25Retriever


def test_contextual_text_prefixes_the_heading_path() -> None:
    chunk = Chunk("d#h001", "d.mdx", ("Tools", "Error Handling"), 0, 12, "Two kinds...")
    assert contextual_text(chunk) == "Tools > Error Handling\n\nTwo kinds..."


def test_bm25_with_headers_matches_words_that_only_appear_in_headings() -> None:
    chunks = [
        Chunk("d#h000", "d.mdx", ("Pagination",), 0, 20, "Use the opaque value."),
        make_chunk(1, "Something unrelated about servers."),
    ]
    assert Bm25Retriever(chunks).search("pagination", k=2) == []
    hits = Bm25Retriever(chunks, text_of=contextual_text).search("pagination", k=2)
    assert [h.chunk.chunk_id for h in hits] == ["d#h000"]


def test_experiment_config_names_round_trip() -> None:
    assert all(c.context_headers for c in EXPERIMENT_CONFIGS)
    for config in EXPERIMENT_CONFIGS:
        assert Config.from_name(config.name) == config
    assert Config.from_name("heading+ctx/hybrid+rerank").name == "heading+ctx/hybrid+rerank"


def _results(name: str, recalls: list[float]) -> dict[str, object]:
    rows = {f"q{i}": {"answerable": True, "recall@5": r} for i, r in enumerate(recalls)}
    mean = sum(recalls) / len(recalls)
    return {"configs": {name: {"summary": {"recall@5": mean}, "per_question": rows}}}


def test_compare_pairs_configs_and_reports_the_difference(tmp_path: Path) -> None:
    base, new = tmp_path / "base.json", tmp_path / "new.json"
    base.write_text(json.dumps(_results("fixed/bm25", [0.0, 1.0, 0.0, 1.0])))
    new.write_text(json.dumps(_results("fixed+ctx/bm25", [1.0, 1.0, 0.0, 1.0])))
    [row] = compare(base, new, strip="+ctx")
    assert row["config"] == "fixed+ctx/bm25"
    assert row["base"] == 0.5 and row["new"] == 0.75 and row["diff"] == 0.25
    assert row["ci_low"] <= 0.25 <= row["ci_high"]
