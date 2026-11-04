"""Grounded answering: retrieve, decide whether to answer at all, then generate with citations.

The refusal decision is made *before* calling the LLM, so it behaves identically in mock
and live mode. Two signals can trigger it:

* word coverage (ConfidenceScorer), available for every retriever, which measures whether
  the docs talk about the question's topic;
* the cross-encoder's top score, available only for reranked retrievers, which reads the
  question and passage together and so says more about whether the passage answers it.

A live LLM can additionally refuse by returning NOT_IN_DOCS.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

from rag_eval_lab.chunking import Chunk
from rag_eval_lab.llm import CITATION, NOT_IN_DOCS, LLMProvider
from rag_eval_lab.retrievers import Hit, Retriever
from rag_eval_lab.text import tokenize

REFUSAL_TEXT = "Not in the docs: I could not find this in the MCP documentation."
DEFAULT_THRESHOLD = 0.35  # chosen from the sweep in evals/results.md


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]
    refused: bool
    confidence: float
    hits: list[Hit] = field(default_factory=list)


def extract_citations(text: str, allowed_ids: set[str]) -> list[str]:
    """Cited chunk ids in first-mention order, ignoring ids that were not retrieved."""
    seen: list[str] = []
    for chunk_id in CITATION.findall(text):
        if chunk_id in allowed_ids and chunk_id not in seen:
            seen.append(chunk_id)
    return seen


class ConfidenceScorer:
    """How much of the question's *information* does the best retrieved chunk contain?

    confidence = max over chunks of  sum(idf(t) for query terms t in the chunk)
                                     / sum(idf(t) for all query terms)

    Rare words carry most of a question's meaning, so they dominate. A word that
    never occurs in the corpus gets the highest idf: if it is missing from every
    chunk, the docs probably do not cover the question.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.n = len(chunks)
        self.df = Counter(term for c in chunks for term in set(tokenize(c.text)))

    def idf(self, term: str) -> float:
        return math.log(1 + (self.n + 0.5) / (self.df.get(term, 0) + 0.5))

    def confidence(self, question: str, chunks: list[Chunk]) -> float:
        terms = set(tokenize(question))
        total = sum(self.idf(t) for t in terms)
        if not chunks or total == 0:
            return 0.0
        best = 0.0
        for chunk in chunks:
            present = terms & set(tokenize(chunk.text))
            best = max(best, sum(self.idf(t) for t in present) / total)
        return best


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        scorer: ConfidenceScorer,
        k: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
        min_top_score: float | None = None,
    ) -> None:
        self.retriever, self.provider, self.scorer = retriever, provider, scorer
        self.k, self.threshold = k, threshold
        # Only meaningful when hit scores are cross-encoder logits (reranked retrievers).
        self.min_top_score = min_top_score

    def ask(self, question: str) -> Answer:
        return self.answer_from_hits(question, self.retriever.search(question, self.k))

    def answer_from_hits(self, question: str, hits: list[Hit]) -> Answer:
        """Answer from already-retrieved hits (the eval harness retrieves once, deeper)."""
        hits = hits[: self.k]
        chunks = [h.chunk for h in hits]
        confidence = self.scorer.confidence(question, chunks)
        weak_passage = self.min_top_score is not None and (
            not hits or hits[0].score < self.min_top_score
        )
        if confidence < self.threshold or weak_passage:
            return Answer(question, REFUSAL_TEXT, [], True, confidence, hits)

        text = self.provider.answer(question, chunks)
        citations = extract_citations(text, {c.chunk_id for c in chunks})
        if text.strip() == NOT_IN_DOCS or not citations:
            # An answer without a valid citation is not grounded: treat it as a refusal.
            return Answer(question, REFUSAL_TEXT, [], True, confidence, hits)
        return Answer(question, text, citations, False, confidence, hits)
