"""Build a complete RAG pipeline from a small configuration value.

The eval harness and the API both go through `build_retrievers`, so what is
evaluated is exactly what is served.
"""

from dataclasses import dataclass

from rag_eval_lab.chunking import STRATEGIES, Chunk, chunk_corpus, contextual_text
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.embedders import LsaEmbedder, MiniLmEmbedder
from rag_eval_lab.retrievers import (
    Bm25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    Retriever,
    plain_text,
)

RETRIEVERS = ("bm25", "dense-lsa", "dense-minilm", "hybrid")


@dataclass(frozen=True)
class Config:
    chunking: str  # one of chunking.STRATEGIES
    retriever: str  # one of RETRIEVERS
    rerank: bool = False
    context_headers: bool = False  # index chunks with their heading path prepended

    @property
    def name(self) -> str:
        ctx = "+ctx" if self.context_headers else ""
        return f"{self.chunking}{ctx}/{self.retriever}{'+rerank' if self.rerank else ''}"

    @classmethod
    def from_name(cls, name: str) -> "Config":
        """Parse "heading/hybrid+rerank" or "fixed+ctx/bm25" back into a Config."""
        chunking, _, retriever = name.partition("/")
        config = cls(
            chunking.removesuffix("+ctx"),
            retriever.removesuffix("+rerank"),
            rerank=retriever.endswith("+rerank"),
            context_headers=chunking.endswith("+ctx"),
        )
        if config not in ALL_CONFIGS + EXPERIMENT_CONFIGS:
            raise ValueError(f"unknown config {name!r}")
        return config


ALL_CONFIGS = [
    Config(chunking, retriever, rerank)
    for chunking in STRATEGIES
    for retriever in RETRIEVERS
    for rerank in (False, True)
]


# Not part of the gated ablation (kept out of CI to bound its run time); evaluated with
# `python -m evals.run_eval --configs ...` and compared with `python -m evals.compare`.
EXPERIMENT_CONFIGS = [
    Config(chunking, retriever, rerank, context_headers=True)
    for chunking in STRATEGIES
    for retriever in ("bm25", "dense-minilm", "hybrid")
    for rerank in (False, True)
]

# Served by the API unless RAG_CONFIG says otherwise; chosen from the ablation results.
DEFAULT_CONFIG = Config("fixed", "hybrid", rerank=True)


def build_retrievers(chunks: list[Chunk], configs: list[Config]) -> dict[Config, Retriever]:
    """Build each requested retriever once, sharing indexes between configs.

    All configs must use the chunking strategy the chunks were made with. Each index
    (per retriever and per indexed text) is built at most once and shared.
    """
    cache: dict[tuple[str, bool], Retriever] = {}

    def base(name: str, ctx: bool) -> Retriever:
        text_of = contextual_text if ctx else plain_text
        if (name, ctx) not in cache:
            if name == "bm25":
                cache[name, ctx] = Bm25Retriever(chunks, text_of=text_of)
            elif name == "dense-lsa":
                cache[name, ctx] = DenseRetriever(chunks, LsaEmbedder(), text_of=text_of)
            elif name == "dense-minilm":
                cache[name, ctx] = DenseRetriever(chunks, MiniLmEmbedder(), text_of=text_of)
            elif name == "hybrid":
                parts = [base("bm25", ctx), base("dense-minilm", ctx)]
                cache[name, ctx] = HybridRetriever(parts)
            else:
                raise ValueError(f"unknown retriever {name!r}; expected one of {RETRIEVERS}")
        return cache[name, ctx]

    retrievers: dict[Config, Retriever] = {}
    for config in configs:
        ctx = config.context_headers
        retriever = base(config.retriever, ctx)
        if config.rerank:
            text_of = contextual_text if ctx else plain_text
            retriever = RerankingRetriever(retriever, text_of=text_of)
        retrievers[config] = retriever
    return retrievers


def build_pipeline(config: Config) -> tuple[list[Chunk], Retriever]:
    """Chunks and retriever for one configuration (used by the API)."""
    chunks = chunk_corpus(load_corpus(), config.chunking)
    return chunks, build_retrievers(chunks, [config])[config]
