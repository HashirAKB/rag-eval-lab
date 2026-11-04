"""Tokenization shared by BM25, the refusal confidence check and the mock LLM."""

import re

# A small English stopword list. Kept short on purpose: words like "not" or "can"
# carry meaning in a specification ("MUST NOT", "can be cancelled").
STOPWORDS = frozenset(
    """
    a an and are as at be by do does for from how i if in into is it its of on or
    that the their then there these this to use used using was what when where
    which who why will with you your
    """.split()  # noqa: SIM905 (a word list is easier to read than a list literal)
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
