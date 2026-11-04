"""Build a complete RAG pipeline from a small configuration value.

The eval harness and the API both go through `build_retrievers`, so what is
evaluated is exactly what is served.
"""

from dataclasses import dataclass

from rag_eval_lab.chunking import STRATEGIES, Chunk, chunk_corpus
from rag_eval_lab.corpus import Document, load_corpus
from rag_eval_lab.embedders import LsaEmbedder, MiniLmEmbedder
from rag_eval_lab.retrievers import (
    Bm25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    Retriever,
)

RETRIEVERS = ("bm25", "dense-lsa", "dense-minilm", "hybrid")
NEURAL_RETRIEVERS = {"dense-minilm", "hybrid"}  # need the MiniLM model


@dataclass(frozen=True)
class Config:
    chunking: str = "heading"
    retriever: str = "hybrid"
    rerank: bool = False

    @property
    def name(self) -> str:
        return f"{self.chunking}/{self.retriever}{'+rerank' if self.rerank else ''}"

    @property
    def needs_neural(self) -> bool:
        return self.rerank or self.retriever in NEURAL_RETRIEVERS

    @classmethod
    def from_name(cls, name: str) -> "Config":
        """Parse "heading/hybrid+rerank" back into a Config."""
        chunking, _, retriever = name.partition("/")
        rerank = retriever.endswith("+rerank")
        config = cls(chunking, retriever.removesuffix("+rerank"), rerank)
        if config not in ALL_CONFIGS:
            raise ValueError(f"unknown config {name!r}")
        return config


ALL_CONFIGS = [
    Config(chunking, retriever, rerank)
    for chunking in STRATEGIES
    for retriever in RETRIEVERS
    for rerank in (False, True)
]


# Served by the API unless RAG_CONFIG says otherwise; chosen from the ablation results.
DEFAULT_CONFIG = Config("fixed", "hybrid", rerank=True)


def build_retrievers(chunks: list[Chunk], configs: list[Config]) -> dict[Config, Retriever]:
    """Build each requested retriever once, sharing indexes between configs.

    All configs must use the chunking strategy the chunks were made with. The
    MiniLM index is built at most once and reused by the hybrid retriever.
    """
    cache: dict[str, Retriever] = {}

    def base(name: str) -> Retriever:
        if name not in cache:
            if name == "bm25":
                cache[name] = Bm25Retriever(chunks)
            elif name == "dense-lsa":
                cache[name] = DenseRetriever(chunks, LsaEmbedder())
            elif name == "dense-minilm":
                cache[name] = DenseRetriever(chunks, MiniLmEmbedder())
            elif name == "hybrid":
                cache[name] = HybridRetriever([base("bm25"), base("dense-minilm")])
            else:
                raise ValueError(f"unknown retriever {name!r}; expected one of {RETRIEVERS}")
        return cache[name]

    return {
        config: RerankingRetriever(base(config.retriever))
        if config.rerank
        else base(config.retriever)
        for config in configs
    }


def build_pipeline(
    config: Config, docs: list[Document] | None = None
) -> tuple[list[Chunk], Retriever]:
    """Chunks and retriever for one configuration (used by the API)."""
    chunks = chunk_corpus(docs if docs is not None else load_corpus(), config.chunking)
    return chunks, build_retrievers(chunks, [config])[config]
