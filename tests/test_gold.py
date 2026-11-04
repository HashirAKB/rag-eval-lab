from rag_eval_lab.chunking import Chunk, chunk_corpus
from rag_eval_lab.corpus import Document, load_corpus
from rag_eval_lab.gold import GoldItem, load_gold, relevant_chunk_sets


def test_gold_set_shape() -> None:
    gold = load_gold()
    assert len(gold) == 60
    assert sum(not g.answerable for g in gold) == 10
    assert len({g.id for g in gold}) == 60
    assert len({g.question for g in gold}) == 60


def test_every_evidence_quote_is_verbatim_in_its_document() -> None:
    docs = {d.path: d.text for d in load_corpus()}
    for item in load_gold():
        if not item.answerable:
            assert item.doc_path is None and item.evidence == ()
            continue
        assert item.evidence, item.id
        for quote in item.evidence:
            assert quote in docs[item.doc_path], (item.id, quote)


def test_gold_covers_many_documents() -> None:
    paths = {g.doc_path for g in load_gold() if g.answerable}
    assert len(paths) >= 25


def test_relevant_chunks_prefer_chunks_that_contain_the_whole_quote() -> None:
    doc = Document("d.mdx", "D", "aaaa QUOTE HERE bbbb")
    chunks = [
        Chunk("d#f000", "d.mdx", ("D",), 0, 10, doc.text[0:10]),  # "aaaa QUOTE" (partial)
        Chunk("d#f001", "d.mdx", ("D",), 5, 20, doc.text[5:20]),  # whole quote
    ]
    item = GoldItem("q", "?", True, "d.mdx", ("QUOTE HERE",))
    assert relevant_chunk_sets(item, chunks, {doc.path: doc}) == [{"d#f001"}]


def test_relevant_chunks_fall_back_to_overlap_when_quote_straddles_a_boundary() -> None:
    doc = Document("d.mdx", "D", "aaaa QUOTE HERE bbbb")
    chunks = [
        Chunk("d#f000", "d.mdx", ("D",), 0, 11, doc.text[0:11]),
        Chunk("d#f001", "d.mdx", ("D",), 11, 20, doc.text[11:20]),
    ]
    item = GoldItem("q", "?", True, "d.mdx", ("QUOTE HERE",))
    assert relevant_chunk_sets(item, chunks, {doc.path: doc}) == [{"d#f000", "d#f001"}]


def test_every_answerable_item_maps_to_chunks_in_both_strategies() -> None:
    docs = load_corpus()
    by_path = {d.path: d for d in docs}
    for strategy in ("fixed", "heading"):
        chunks = chunk_corpus(docs, strategy)
        for item in load_gold():
            sets = relevant_chunk_sets(item, chunks, by_path)
            assert all(sets) and len(sets) == len(item.evidence), (strategy, item.id)
