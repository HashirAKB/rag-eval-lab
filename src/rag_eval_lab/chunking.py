"""Two chunking strategies over cleaned documents.

* fixed:   sliding character windows with overlap, snapped to whitespace.
* heading: one chunk per markdown section; tiny sections are merged forward and
           long sections are split on paragraph boundaries.

A chunk always satisfies `doc.text[chunk.start:chunk.end] == chunk.text`. That
invariant is what lets gold evidence (stored as document character spans) be
mapped onto the chunks of *any* strategy.
"""

import re
from dataclasses import dataclass

from rag_eval_lab.corpus import Document

FIXED_SIZE = 800
FIXED_OVERLAP = 160
HEADING_MAX_CHARS = 1500
HEADING_MIN_CHARS = 200

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # e.g. "specification/server/tools#h007" (f = fixed, h = heading)
    doc_path: str
    heading_path: tuple[str, ...]  # ("Tools", "Protocol Messages", "Calling Tools")
    start: int  # character offsets into the cleaned document text
    end: int
    text: str


def _headings(doc: Document) -> list[tuple[int, tuple[str, ...]]]:
    """Return (offset, heading_path) for every heading line outside code fences.

    The first line is always the "# {title}" line added by the cleaner; it becomes
    the root of every path, so body headings of any level sit below it.
    """
    result: list[tuple[int, tuple[str, ...]]] = [(0, (doc.title,))]
    stack: list[tuple[int, str]] = []  # (level, heading text) below the title
    in_code = False
    offset = 0
    for line in doc.text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code and offset > 0 and (match := _HEADING.match(line)):
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
            result.append((offset, (doc.title, *(text for _, text in stack))))
        offset += len(line)
    return result


def _path_at(headings: list[tuple[int, tuple[str, ...]]], position: int) -> tuple[str, ...]:
    path = headings[0][1]
    for offset, heading_path in headings:
        if offset > position:
            break
        path = heading_path
    return path


def _stem(doc: Document) -> str:
    return doc.path.removesuffix(".mdx")


def _window_end(text: str, start: int, size: int) -> int:
    """End of a window of at most `size` chars, moved back so no word is cut."""
    end = min(start + size, len(text))
    if end == len(text):
        return end
    cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
    return cut + 1 if cut > start else end  # a single huge "word": cut it anyway


def chunk_fixed(doc: Document, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[Chunk]:
    headings = _headings(doc)
    text = doc.text
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = _window_end(text, start, size)
        spans.append((start, end))
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Move forward to the beginning of a word so chunks do not start mid-word.
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start
    return [
        Chunk(f"{_stem(doc)}#f{i:03d}", doc.path, _path_at(headings, s), s, e, text[s:e])
        for i, (s, e) in enumerate(spans)
    ]


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split text[start:end] into pieces of at most max_chars at paragraph breaks."""
    pieces: list[tuple[int, int]] = []
    piece_start = start
    while end - piece_start > max_chars:
        limit = piece_start + max_chars
        cut = text.rfind("\n\n", piece_start, limit - 1)
        # Cut after the blank line; if there is no paragraph break, cut at a word.
        cut = cut + 2 if cut > piece_start else _window_end(text, piece_start, max_chars)
        pieces.append((piece_start, cut))
        piece_start = cut
    pieces.append((piece_start, end))
    return pieces


def chunk_headings(
    doc: Document, max_chars: int = HEADING_MAX_CHARS, min_chars: int = HEADING_MIN_CHARS
) -> list[Chunk]:
    headings = _headings(doc)
    bounds = [offset for offset, _ in headings] + [len(doc.text)]
    sections = [(bounds[i], bounds[i + 1], headings[i][1]) for i in range(len(headings))]

    # Merge sections shorter than min_chars (often a bare heading) into the next one.
    # The merged chunk takes the next section's (deeper, more specific) heading path.
    merged: list[tuple[int, int, tuple[str, ...]]] = []
    pending_start: int | None = None
    for start, end, path in sections:
        start = start if pending_start is None else pending_start
        if end - start < min_chars and end < len(doc.text):
            pending_start = start
            continue
        pending_start = None
        merged.append((start, end, path))

    chunks: list[Chunk] = []
    for start, end, path in merged:
        for s, e in _split_long(doc.text, start, end, max_chars):
            chunk_id = f"{_stem(doc)}#h{len(chunks):03d}"
            chunks.append(Chunk(chunk_id, doc.path, path, s, e, doc.text[s:e]))
    return chunks


STRATEGIES = ("fixed", "heading")


def chunk_corpus(docs: list[Document], strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        return [c for doc in docs for c in chunk_fixed(doc)]
    if strategy == "heading":
        return [c for doc in docs for c in chunk_headings(doc)]
    raise ValueError(f"unknown chunking strategy {strategy!r}; expected one of {STRATEGIES}")
