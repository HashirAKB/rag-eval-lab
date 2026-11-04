"""LLM providers: everything that would need a language model goes through this interface.

* MockProvider (default): deterministic and offline. Its "answer" is extractive: the
  sentences from the retrieved chunks that share the most words with the question,
  each followed by a citation. Its faithfulness score is a token-overlap heuristic.
* AnthropicProvider: used only when ANTHROPIC_API_KEY is set. Writes a grounded answer
  with citations and acts as an LLM judge for faithfulness.

Both return the literal NOT_IN_DOCS when the context does not contain an answer.
"""

import json
import os
import re
from typing import TYPE_CHECKING, Protocol

from rag_eval_lab.chunking import Chunk
from rag_eval_lab.text import tokenize

if TYPE_CHECKING:
    from anthropic.types.beta import BetaOutputConfigParam

NOT_IN_DOCS = "NOT_IN_DOCS"
CITATION = re.compile(r"\[([^\[\]\s]+#[fh]\d{3})\]")
DEFAULT_MODEL = "claude-opus-5-5"


class LLMProvider(Protocol):
    name: str
    faithfulness_method: str  # reported next to every faithfulness number

    def answer(self, question: str, contexts: list[Chunk]) -> str: ...

    def faithfulness(self, answer: str, contexts: list[Chunk]) -> float: ...


_LINK_DEFINITION = re.compile(r"^\s*\[[^\]]+\]:\s")  # "[label]: /some/url" lines


def _sentences(text: str) -> list[str]:
    """Prose sentences of a chunk: code, headings, tables and link definitions are skipped."""
    prose = re.sub(r"```.*?(```|$)", " ", text, flags=re.DOTALL)
    lines = [
        ln
        for ln in prose.splitlines()
        if not ln.lstrip().startswith(("#", "|")) and not _LINK_DEFINITION.match(ln)
    ]
    joined = re.sub(r"\s+", " ", " ".join(lines))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if len(s.split()) >= 4]


class MockProvider:
    name = "mock"
    faithfulness_method = "token-overlap heuristic (mock)"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        query_terms = set(tokenize(question))
        scored = []  # (overlap, -context rank, -sentence position, sentence, chunk id)
        for rank, chunk in enumerate(contexts):
            for position, sentence in enumerate(_sentences(chunk.text)):
                overlap = len(query_terms & set(tokenize(sentence)))
                if overlap:
                    scored.append((overlap, -rank, -position, sentence, chunk.chunk_id))
        if not scored:
            return NOT_IN_DOCS
        best = sorted(scored, reverse=True)[: self.max_sentences]
        return " ".join(f"{sentence} [{chunk_id}]" for *_, sentence, chunk_id in best)

    def faithfulness(self, answer: str, contexts: list[Chunk]) -> float:
        """Fraction of answer tokens that appear in the cited chunks."""
        cited = set(CITATION.findall(answer))
        source_terms = {t for c in contexts if c.chunk_id in cited for t in tokenize(c.text)}
        answer_terms = tokenize(CITATION.sub(" ", answer))
        if not answer_terms:
            return 1.0
        return sum(t in source_terms for t in answer_terms) / len(answer_terms)


ANSWER_SYSTEM = f"""You answer questions about the Model Context Protocol documentation.
Use only the numbered context chunks you are given. After every sentence, cite the chunk
ids that support it in square brackets, for example [specification/server/tools#h003].
If the chunks do not contain the answer, reply with exactly {NOT_IN_DOCS} and nothing else.
Answer in at most four sentences."""

JUDGE_SYSTEM = """You grade whether an answer is supported by its cited source chunks.
Score 1.0 if every claim is directly supported, 0.0 if none is, and in between for the
fraction of claims that are supported. Judge support only, not style or completeness."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
    "required": ["unsupported_claims", "score"],
    "additionalProperties": False,
}


def _format_contexts(contexts: list[Chunk]) -> str:
    return "\n\n".join(f'<chunk id="{c.chunk_id}">\n{c.text}\n</chunk>' for c in contexts)


class AnthropicProvider:
    name = "anthropic"
    faithfulness_method = "LLM-as-judge (live)"

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # optional dependency: `pip install .[live]`

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL

    def _complete(self, system: str, prompt: str, output_config: "BetaOutputConfigParam") -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
            # If a safety classifier declines, retry server-side on a suitable model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return NOT_IN_DOCS
        return "".join(block.text for block in response.content if block.type == "text")

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        prompt = f"{_format_contexts(contexts)}\n\nQuestion: {question}"
        return self._complete(ANSWER_SYSTEM, prompt, {"effort": "low"}).strip()

    def faithfulness(self, answer: str, contexts: list[Chunk]) -> float:
        cited = set(CITATION.findall(answer))
        sources = [c for c in contexts if c.chunk_id in cited]
        prompt = f"{_format_contexts(sources)}\n\nAnswer to grade:\n{answer}"
        output_config: BetaOutputConfigParam = {
            "effort": "low",
            "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
        }
        result = json.loads(self._complete(JUDGE_SYSTEM, prompt, output_config))
        return min(max(float(result["score"]), 0.0), 1.0)


def get_provider() -> LLMProvider:
    """Live provider only when an API key is present; otherwise the offline mock."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    return MockProvider()
