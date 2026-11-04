from evals.label_review_queue import review_candidates
from rag_eval_lab.chunking import Chunk
from rag_eval_lab.gold import GoldItem
from rag_eval_lab.retrievers import Hit


def hit(chunk_id: str, doc_path: str) -> Hit:
    return Hit(Chunk(chunk_id, doc_path, ("T",), 0, 1, "x"), 1.0)


ITEM = GoldItem("q1", "?", True, "a.mdx", ("quote",))


def test_lists_other_documents_ranked_above_the_labelled_chunk() -> None:
    hits = [hit("b#f000", "b.mdx"), hit("a#f001", "a.mdx"), hit("a#f002", "a.mdx")]
    hits.append(hit("c#f000", "c.mdx"))
    candidates = review_candidates(ITEM, hits, relevant={"a#f002"})
    # same-document chunks are not alternative sources; anything after the label is ignored
    assert [h.chunk.chunk_id for h in candidates] == ["b#f000"]


def test_lists_all_other_documents_when_the_label_is_missed() -> None:
    hits = [hit("b#f000", "b.mdx"), hit("c#f000", "c.mdx")]
    assert len(review_candidates(ITEM, hits, relevant={"a#f009"})) == 2
