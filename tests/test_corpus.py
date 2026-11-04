from pathlib import Path

from rag_eval_lab.corpus import CORPUS_DIR, clean_mdx, load_corpus


def test_clean_mdx_strips_frontmatter_and_jsx_but_keeps_code() -> None:
    raw = (
        "---\ntitle: Tools\n---\n\n"
        '<div id="enable-section-numbers" />\n\n'
        "Intro text&mdash;with an entity.\n\n"
        "<Note>\n  Remember this.\n</Note>\n\n"
        '```tsx\n<Card title="kept">\n```\n'
    )
    title, text = clean_mdx(raw)
    assert title == "Tools"
    assert text.startswith("# Tools\n")
    assert "<div" not in text
    assert "<Note>" not in text and "</Note>" not in text
    assert "Remember this." in text
    assert "Intro text—with an entity." in text
    # JSX-looking lines inside fenced code blocks are content and must survive.
    assert '<Card title="kept">' in text
    assert "\n\n\n" not in text


def test_load_corpus_reads_every_mdx_file_with_relative_paths() -> None:
    docs = load_corpus()
    files = sorted(p.relative_to(CORPUS_DIR).as_posix() for p in CORPUS_DIR.rglob("*.mdx"))
    assert [d.path for d in docs] == files
    assert len(docs) == 37
    tools = next(d for d in docs if d.path == "specification/server/tools.mdx")
    assert tools.title == "Tools"
    assert "model-controlled" in tools.text


def test_load_corpus_accepts_a_custom_directory(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.mdx").write_text("---\ntitle: X\n---\nBody\n")
    [doc] = load_corpus(tmp_path)
    assert doc.path == "a/x.mdx"
    assert doc.text == "# X\n\nBody\n"
