import pytest

from rag_eval_lab.chunking import Chunk, chunk_corpus, chunk_fixed, chunk_headings
from rag_eval_lab.corpus import Document, load_corpus

DOC = Document(
    path="spec/tools.mdx",
    title="Tools",
    text=(
        "# Tools\n\nIntro paragraph about tools.\n\n"
        "## Capabilities\n\nServers declare the tools capability.\n\n"
        "### List changed\n\nlistChanged means notifications are sent.\n\n"
        "```python\n# not a heading, just a comment\nprint('hi')\n```\n\n"
        "## Errors\n\nTwo error mechanisms exist.\n"
    ),
)


def assert_offsets_slice_back(chunks: list[Chunk], doc: Document) -> None:
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text


def test_fixed_chunks_have_overlap_and_cover_the_whole_document() -> None:
    chunks = chunk_fixed(DOC, size=60, overlap=15)
    assert_offsets_slice_back(chunks, DOC)
    assert chunks[0].start == 0 and chunks[-1].end == len(DOC.text)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start < prev.end, "consecutive windows must overlap"
        assert nxt.start > prev.start, "windows must always advance"
    assert all(len(c.text) <= 60 for c in chunks)


def test_fixed_chunks_do_not_split_words() -> None:
    for c in chunk_fixed(DOC, size=60, overlap=15):
        end = c.end
        assert end == len(DOC.text) or DOC.text[end - 1].isspace() or DOC.text[end].isspace()


def test_fixed_chunk_ids_and_heading_path_at_chunk_start() -> None:
    chunks = chunk_fixed(DOC, size=60, overlap=15)
    assert [c.chunk_id for c in chunks[:2]] == ["spec/tools#f000", "spec/tools#f001"]
    assert chunks[0].heading_path == ("Tools",)
    last = chunks[-1]
    assert last.heading_path == ("Tools", "Errors")


def test_heading_chunks_follow_sections_and_ignore_hashes_in_code() -> None:
    chunks = chunk_headings(DOC, max_chars=1000, min_chars=0)
    assert_offsets_slice_back(chunks, DOC)
    assert [c.heading_path for c in chunks] == [
        ("Tools",),
        ("Tools", "Capabilities"),
        ("Tools", "Capabilities", "List changed"),
        ("Tools", "Errors"),
    ]
    assert chunks[1].text.startswith("## Capabilities")
    assert "print('hi')" in chunks[2].text
    assert [c.chunk_id for c in chunks] == [f"spec/tools#h{i:03d}" for i in range(4)]


def test_heading_chunks_merge_tiny_sections_forward() -> None:
    text = "# D\n\n## A\n\n### B\n\nBody of B is long enough.\n"
    doc = Document(path="d.mdx", title="D", text=text)
    chunks = chunk_headings(doc, max_chars=1000, min_chars=20)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("D", "A", "B")
    assert chunks[0].text == doc.text


def test_heading_chunks_split_long_sections_on_paragraphs() -> None:
    body = "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(10))
    doc = Document(path="d.mdx", title="D", text=f"# D\n\n{body}\n")
    chunks = chunk_headings(doc, max_chars=300, min_chars=0)
    assert len(chunks) > 1
    assert_offsets_slice_back(chunks, doc)
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.heading_path == ("D",) for c in chunks)
    assert "".join(c.text for c in chunks) == doc.text


@pytest.mark.parametrize("strategy", ["fixed", "heading"])
def test_real_corpus_chunks_are_valid(strategy: str) -> None:
    docs = load_corpus()
    by_path = {d.path: d for d in docs}
    chunks = chunk_corpus(docs, strategy)
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk ids must be unique"
    for c in chunks:
        assert by_path[c.doc_path].text[c.start : c.end] == c.text
        assert c.heading_path and c.heading_path[0] == by_path[c.doc_path].title
        assert c.text.strip()


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_corpus([DOC], "sentence")
