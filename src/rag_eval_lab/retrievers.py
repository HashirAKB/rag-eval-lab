"""Retrievers: given a query, return the top-k chunks with a score.

* Bm25Retriever:      lexical; rewards rare query words that appear often in a chunk.
* DenseRetriever:     semantic; cosine similarity between query and chunk embeddings.
* HybridRetriever:    fuses several rankings with Reciprocal Rank Fusion (RRF).
* RerankingRetriever: re-scores a base retriever's top candidates with a cross-encoder.

Scores are only comparable *within* one retriever. BM25 scores are unbounded,
cosine is in [-1, 1], RRF scores are tiny. That is why fusion uses ranks, not scores.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_eval_lab.chunking import Chunk
from rag_eval_lab.embedders import Embedder, cache_dir_from_env, embed_corpus
from rag_eval_lab.models import load_reranker_model
from rag_eval_lab.text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def plain_text(chunk: Chunk) -> str:
    return chunk.text


# What a retriever indexes for each chunk; see chunking.contextual_text for the alternative.
TextOf = Callable[[Chunk], str]


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    # A stable sort on -score keeps ties in corpus order, so results are deterministic.
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(chunks[i], float(scores[i])) for i in order]


class Bm25Retriever:
    """Okapi BM25.

    score(q, d) = sum over unique query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    k1 controls how quickly repeated terms stop adding score; b controls how much
    long chunks are penalised.
    """

    name = "bm25"

    def __init__(
        self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, text_of: TextOf = plain_text
    ) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        token_lists = [tokenize(text_of(c)) for c in chunks]
        self.doc_lengths = np.array([len(tokens) for tokens in token_lists], dtype=np.float64)
        self.avg_length = float(self.doc_lengths.mean())
        # Inverted index: term -> list of (chunk index, term frequency in that chunk).
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(token_lists):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((index, tf))

    def idf(self, term: str) -> float:
        n, df = len(self.chunks), len(self.postings.get(term, []))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[Hit]:
        scores = np.zeros(len(self.chunks))
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for index, tf in self.postings.get(term, []):
                length_norm = 1 - self.b + self.b * self.doc_lengths[index] / self.avg_length
                scores[index] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        hits = _top_k(self.chunks, scores, k)
        return [h for h in hits if h.score > 0]  # no shared words means no evidence


class DenseRetriever:
    def __init__(
        self, chunks: list[Chunk], embedder: Embedder, text_of: TextOf = plain_text
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.name = f"dense-{embedder.name}"
        texts = [text_of(c) for c in chunks]
        embedder.fit(texts)
        # shape (n_chunks, dim), unit-length rows; loaded from disk when already computed
        self.matrix = embed_corpus(embedder, texts, cache_dir_from_env())

    def search(self, query: str, k: int) -> list[Hit]:
        query_vector = self.embedder.embed([query])[0]
        return _top_k(self.chunks, self.matrix @ query_vector, k)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """RRF: each list contributes 1 / (k + rank) for every id it contains (rank from 1).

    k = 60 comes from the original paper (Cormack et al., 2009); it damps the
    advantage of the very top ranks so one list cannot dominate.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], depth: int = 50, rrf_k: int = 60) -> None:
        self.retrievers = retrievers
        self.depth, self.rrf_k = depth, rrf_k
        self.name = "hybrid"

    def search(self, query: str, k: int) -> list[Hit]:
        by_id: dict[str, Chunk] = {}
        rankings = []
        for retriever in self.retrievers:
            hits = retriever.search(query, self.depth)
            by_id.update((h.chunk.chunk_id, h.chunk) for h in hits)
            rankings.append([h.chunk.chunk_id for h in hits])
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
        return [Hit(by_id[chunk_id], score) for chunk_id, score in fused]


class RerankingRetriever:
    """Retrieve `depth` candidates cheaply, then re-score each (query, chunk) pair jointly.

    A cross-encoder reads query and chunk together, so it is more accurate than
    comparing two independent embeddings, but too slow to run over every chunk.
    """

    def __init__(self, base: Retriever, depth: int = 20, text_of: TextOf = plain_text) -> None:
        self.base = base
        self.depth = depth
        self.text_of = text_of
        self.name = f"{base.name}+rerank"

    def search(self, query: str, k: int) -> list[Hit]:
        candidates = self.base.search(query, self.depth)
        if not candidates:
            return []
        model = load_reranker_model()
        scores = np.asarray(model.predict([(query, self.text_of(h.chunk)) for h in candidates]))
        return _top_k([h.chunk for h in candidates], scores, k)
