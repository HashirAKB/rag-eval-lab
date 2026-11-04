"""Load the MDX documentation corpus and clean it into plain markdown.

Every later offset (chunk start/end, gold evidence spans) refers to the *cleaned*
text produced here, so cleaning must be deterministic.
"""

import html
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_TITLE = re.compile(r"^title:\s*['\"]?(.*?)['\"]?\s*$", re.MULTILINE)
# A line that is only a JSX/HTML tag, e.g. `<Note>`, `</Warning>`, `<div id="x" />`,
# `<Card title="..." href="...">`. Markdown content between tags is kept.
_TAG_LINE = re.compile(r"^\s*</?[A-Za-z][^>]*>\s*$")


@dataclass(frozen=True)
class Document:
    path: str  # relative to the corpus root, e.g. "specification/server/tools.mdx"
    title: str
    text: str  # cleaned markdown, starting with "# {title}"


def clean_mdx(raw: str) -> tuple[str, str]:
    """Return (title, cleaned markdown) for one MDX file."""
    title = ""
    match = _FRONTMATTER.match(raw)
    if match:
        title_match = _TITLE.search(match.group(1))
        title = title_match.group(1) if title_match else ""
        raw = raw[match.end() :]

    lines: list[str] = []
    in_code = False
    for line in raw.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            lines.append(line)
        elif in_code:
            lines.append(line)  # code is content: keep it byte-for-byte
        elif not _TAG_LINE.match(line):
            lines.append(html.unescape(line.rstrip()))

    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return title, f"# {title}\n\n{body}\n"


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    """Load every .mdx file under corpus_dir, sorted by path for determinism."""
    docs = []
    for file in sorted(corpus_dir.rglob("*.mdx")):
        title, text = clean_mdx(file.read_text(encoding="utf-8"))
        docs.append(Document(path=file.relative_to(corpus_dir).as_posix(), title=title, text=text))
    return docs
