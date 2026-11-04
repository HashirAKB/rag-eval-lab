import numpy as np
import pytest
from conftest import requires_neural

from rag_eval_lab.chunking import Chunk, chunk_corpus
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.embedders import LsaEmbedder
from rag_eval_lab.retrievers import (
    Bm25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    Retriever,
    reciprocal_rank_fusion,
)


def ids(hits: list[Hit]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_bm25_ranks_the_chunk_with_rare_query_terms_first(toy_chunks: list[Chunk]) -> None:
    hits = Bm25Retriever(toy_chunks).search("server cursor pagination", k=3)
    assert ids(hits)[0] == "doc#h005"
    assert hits[0].score > hits[1].score


def test_bm25_scores_match_the_formula_by_hand() -> None:
    from conftest import make_chunk

    chunks = [make_chunk(0, "apple apple banana"), make_chunk(1, "banana cherry")]
    retriever = Bm25Retriever(chunks, k1=1.5, b=0.75)
    [top, *_] = retriever.search("apple", k=2)
    # N=2, df(apple)=1 -> idf = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln 2
    # tf=2, |d|=3, avgdl=2.5 -> 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2.5))
    idf = np.log(2.0)
    tf_part = 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 2.5))
    assert top.chunk.chunk_id == "doc#h000"
    assert top.score == pytest.approx(idf * tf_part)


def test_bm25_returns_nothing_for_unknown_words(toy_chunks: list[Chunk]) -> None:
    assert Bm25Retriever(toy_chunks).search("kubernetes helm", k=3) == []


def test_dense_lsa_scores_are_cosine_similarities(toy_chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(toy_chunks, LsaEmbedder(dimensions=5))
    hits = retriever.search("subprocess stdin stdout", k=6)
    assert "doc#h003" in ids(hits)[:2]
    assert all(-1.0 - 1e-6 <= h.score <= 1.0 + 1e-6 for h in hits)


def test_rrf_rewards_documents_ranked_well_by_both_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert [doc for doc, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_combines_lexical_and_dense(toy_chunks: list[Chunk]) -> None:
    hybrid = HybridRetriever(
        [Bm25Retriever(toy_chunks), DenseRetriever(toy_chunks, LsaEmbedder(dimensions=5))]
    )
    hits = hybrid.search("cursor for pages", k=3)
    assert ids(hits)[0] == "doc#h005"
    assert len(hits) == 3


@requires_neural
def test_minilm_dense_handles_paraphrase(toy_chunks: list[Chunk]) -> None:
    from rag_eval_lab.embedders import MiniLmEmbedder

    retriever = DenseRetriever(toy_chunks, MiniLmEmbedder())
    hits = retriever.search("how do I stop a request that is still running?", k=2)
    assert ids(hits)[0] == "doc#h004"


@requires_neural
def test_cross_encoder_reranker_reorders_candidates(toy_chunks: list[Chunk]) -> None:
    from rag_eval_lab.retrievers import RerankingRetriever

    base = Bm25Retriever(toy_chunks)
    reranked = RerankingRetriever(base, depth=6)
    hits = reranked.search("which transport runs the server as a child process?", k=3)
    assert ids(hits)[0] == "doc#h003"


@pytest.fixture(scope="module")
def heading_chunks() -> list[Chunk]:
    return chunk_corpus(load_corpus(), "heading")


# (query, a document path whose chunks should rank in the top 5). Paraphrased on purpose.
KNOWN_QUERIES = [
    ("launch the server as a subprocess and exchange messages over standard input", "stdio"),
    ("opaque cursor for fetching results page by page", "pagination"),
]


def _check_known_queries(retriever: Retriever) -> None:
    for query, expected in KNOWN_QUERIES:
        paths = [h.chunk.doc_path for h in retriever.search(query, k=5)]
        assert any(expected in p for p in paths), (retriever.name, query, paths)


def test_bm25_and_lsa_find_known_sources_in_real_corpus(heading_chunks: list[Chunk]) -> None:
    _check_known_queries(Bm25Retriever(heading_chunks))
    _check_known_queries(DenseRetriever(heading_chunks, LsaEmbedder()))


@requires_neural
def test_neural_retrievers_find_known_sources_in_real_corpus(heading_chunks: list[Chunk]) -> None:
    from rag_eval_lab.embedders import MiniLmEmbedder
    from rag_eval_lab.retrievers import RerankingRetriever

    bm25 = Bm25Retriever(heading_chunks)
    dense = DenseRetriever(heading_chunks, MiniLmEmbedder())
    for retriever in [dense, HybridRetriever([bm25, dense]), RerankingRetriever(bm25)]:
        _check_known_queries(retriever)
