"""Embedders turn texts into L2-normalised vectors, so a dot product is cosine similarity.

* LsaEmbedder:   TF-IDF + truncated SVD ("latent semantic analysis"). Fitted on the
                 corpus itself; fully offline; a weak but honest dense baseline.
* MiniLmEmbedder: a pretrained sentence-transformers model (all-MiniLM-L6-v2).
"""

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_eval_lab.models import load_embedding_model
from rag_eval_lab.text import tokenize


class Embedder(Protocol):
    name: str

    def fit(self, corpus: list[str]) -> None:
        """Learn from the corpus if the embedder needs it (LSA does, MiniLM does not)."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with unit-length rows."""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalised: np.ndarray = vectors / np.maximum(norms, 1e-12)
    return normalised


class LsaEmbedder:
    name = "lsa"

    def __init__(self, dimensions: int = 256) -> None:
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dimensions, random_state=0)

    def fit(self, corpus: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(corpus)
        # SVD cannot produce more components than there are features.
        self.svd.n_components = min(self.svd.n_components, tfidf.shape[1] - 1)
        self.svd.fit(tfidf)

    def embed(self, texts: list[str]) -> np.ndarray:
        reduced = self.svd.transform(self.vectorizer.transform(texts))
        return _normalise(np.asarray(reduced, dtype=np.float32))


class MiniLmEmbedder:
    name = "minilm"

    def fit(self, corpus: list[str]) -> None:
        pass  # pretrained: nothing to learn from our corpus

    def embed(self, texts: list[str]) -> np.ndarray:
        model = load_embedding_model()
        vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
