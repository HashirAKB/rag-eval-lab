"""The gold set: questions with verbatim evidence quotes from the source document.

Labels are *quotes*, not chunk ids, because chunk ids depend on the chunking
strategy. For any strategy we map each quote to the chunk(s) that hold it.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag_eval_lab.chunking import Chunk
from rag_eval_lab.corpus import Document

GOLD_PATH = Path(__file__).resolve().parents[2] / "evals" / "gold.jsonl"


@dataclass(frozen=True)
class GoldItem:
    id: str
    question: str
    answerable: bool
    doc_path: str | None
    evidence: tuple[str, ...]


def load_gold(path: Path = GOLD_PATH) -> list[GoldItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        raw["evidence"] = tuple(raw["evidence"])
        items.append(GoldItem(**raw))
    return items


def relevant_chunk_sets(
    item: GoldItem, chunks: list[Chunk], docs: dict[str, Document]
) -> list[set[str]]:
    """For each evidence quote, the ids of the chunks that hold it.

    A chunk that contains the whole quote is relevant. If no single chunk does
    (the quote straddles a chunk boundary), every chunk overlapping it is relevant.
    Unanswerable items have no evidence, so they get an empty list.
    """
    if item.doc_path is None:
        return []
    text = docs[item.doc_path].text
    doc_chunks = [c for c in chunks if c.doc_path == item.doc_path]
    result = []
    for quote in item.evidence:
        start = text.index(quote)  # first occurrence; the gold test checks it exists
        end = start + len(quote)
        containing = {c.chunk_id for c in doc_chunks if c.start <= start and end <= c.end}
        overlapping = {c.chunk_id for c in doc_chunks if c.start < end and start < c.end}
        result.append(containing or overlapping)
    return result
