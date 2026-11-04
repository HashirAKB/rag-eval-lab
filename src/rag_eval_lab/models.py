"""Lazy, cached loading of the local sentence-transformers models.

These are imported only when used, so the package works without the optional
`neural` extra (BM25 and the LSA embedder need no model download).
"""

from functools import cache
from typing import Any

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@cache
def load_embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@cache
def load_reranker_model() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL, device="cpu")


def preload() -> None:
    """Download both models into the local cache (used by the Dockerfile)."""
    load_embedding_model()
    load_reranker_model()
