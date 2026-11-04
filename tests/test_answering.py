import json
import os
from pathlib import Path
from typing import Any

import pytest
from conftest import make_chunk

from rag_eval_lab.answering import (
    REFUSAL_TEXT,
    Answerer,
    ConfidenceScorer,
    extract_citations,
)
from rag_eval_lab.chunking import Chunk, chunk_corpus
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.llm import NOT_IN_DOCS, MockProvider, get_provider
from rag_eval_lab.retrievers import Bm25Retriever

FIXTURE = Path(__file__).parent / "fixtures" / "mock_answers.json"


def test_extract_citations_keeps_only_known_ids_in_order() -> None:
    text = "A [doc#h001]. B [doc#h002][doc#h001]. C [made-up#h999]. D [doc#f1234]."
    allowed = {"doc#h001", "doc#h002", "doc#f1234"}
    assert extract_citations(text, allowed) == ["doc#h001", "doc#h002", "doc#f1234"]


def test_confidence_is_idf_weighted_coverage_of_best_chunk(toy_chunks: list[Chunk]) -> None:
    scorer = ConfidenceScorer(toy_chunks)
    stdio = [toy_chunks[3]]
    assert scorer.confidence("stdio subprocess", stdio) == pytest.approx(1.0)
    # "kubernetes" is not in the corpus, so it gets the highest possible idf and
    # drags confidence well below one half.
    assert scorer.confidence("stdio kubernetes", stdio) < 0.5
    assert scorer.confidence("anything", []) == 0.0


def test_mock_provider_is_extractive_and_cites_its_sources(toy_chunks: list[Chunk]) -> None:
    provider = MockProvider()
    text = provider.answer("How does pagination use a cursor?", toy_chunks)
    assert "opaque cursor" in text
    assert extract_citations(text, {c.chunk_id for c in toy_chunks}) == ["doc#h005"]
    assert provider.answer("How does pagination use a cursor?", toy_chunks) == text


def test_mock_provider_ignores_code_headings_and_link_definitions() -> None:
    chunk = make_chunk(
        0,
        "## Cancel heading\n\n[cancel-ref]: /spec/cancel\n\n```\ncancel(request)\n```\n"
        "Clients cancel a request by sending a notification.",
    )
    text = MockProvider().answer("how to cancel a request", [chunk])
    assert text == "Clients cancel a request by sending a notification. [doc#h000]"


def test_mock_provider_says_not_in_docs_without_overlap(toy_chunks: list[Chunk]) -> None:
    assert MockProvider().answer("kubernetes helm chart", toy_chunks) == NOT_IN_DOCS


def test_mock_faithfulness_is_token_overlap_with_cited_chunks() -> None:
    chunk = make_chunk(0, "Cursors are opaque strings.")
    provider = MockProvider()
    assert provider.faithfulness("Cursors are opaque [doc#h000].", [chunk]) == 1.0
    assert provider.faithfulness(
        "Cursors are blue and opaque [doc#h000].", [chunk]
    ) == pytest.approx(2 / 3)


def test_answerer_refuses_when_confidence_is_low(toy_chunks: list[Chunk]) -> None:
    answerer = Answerer(Bm25Retriever(toy_chunks), MockProvider(), ConfidenceScorer(toy_chunks))
    answer = answerer.ask("How do I deploy with kubernetes helm charts?")
    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


def test_answerer_answers_with_valid_citations(toy_chunks: list[Chunk]) -> None:
    answerer = Answerer(Bm25Retriever(toy_chunks), MockProvider(), ConfidenceScorer(toy_chunks))
    answer = answerer.ask("What does the stdio transport launch as a subprocess?")
    assert not answer.refused
    assert answer.citations == ["doc#h003"]
    assert set(answer.citations) <= {h.chunk.chunk_id for h in answer.hits}


def test_get_provider_uses_mock_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(get_provider(), MockProvider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert isinstance(get_provider(), MockProvider)


def test_get_provider_uses_anthropic_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    from rag_eval_lab.llm import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    assert isinstance(get_provider(), AnthropicProvider)  # constructed, never called


GOLDEN_QUESTIONS = [
    "What does a client send to cancel an in-progress request?",
    "How should a server report that a tool call failed?",
    "What is the airspeed velocity of an unladen swallow?",
]


def test_mock_answers_match_recorded_fixture() -> None:
    """Golden snapshot: any change to chunking, retrieval or the mock shows up here.

    Re-record intentionally with: UPDATE_FIXTURES=1 pytest tests/test_answering.py
    """
    chunks = chunk_corpus(load_corpus(), "heading")
    answerer = Answerer(Bm25Retriever(chunks), MockProvider(), ConfidenceScorer(chunks))
    actual = {
        q: {"text": a.text, "citations": a.citations, "refused": a.refused}
        for q in GOLDEN_QUESTIONS
        for a in [answerer.ask(q)]
    }
    if os.environ.get("UPDATE_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(actual, indent=2) + "\n")
    assert actual == json.loads(FIXTURE.read_text())


class _FakeMessages:
    """Stands in for client.beta.messages so the live code path runs without a network."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.text, self.stop_reason = text, stop_reason
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        from types import SimpleNamespace

        self.calls.append(kwargs)
        block = SimpleNamespace(type="text", text=self.text)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[block])


def _fake_anthropic(monkeypatch: pytest.MonkeyPatch, messages: _FakeMessages) -> Any:
    pytest.importorskip("anthropic")
    from types import SimpleNamespace

    from rag_eval_lab.llm import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    provider = AnthropicProvider(model="claude-opus-5-5")
    provider.client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return provider


def test_anthropic_provider_sends_contexts_and_returns_text(
    monkeypatch: pytest.MonkeyPatch, toy_chunks: list[Chunk]
) -> None:
    messages = _FakeMessages("Cursors are opaque [doc#h005].")
    provider = _fake_anthropic(monkeypatch, messages)
    assert provider.answer("How do cursors work?", toy_chunks) == "Cursors are opaque [doc#h005]."
    prompt = messages.calls[0]["messages"][0]["content"]
    assert '<chunk id="doc#h005">' in prompt and "How do cursors work?" in prompt


def test_anthropic_judge_parses_structured_score(
    monkeypatch: pytest.MonkeyPatch, toy_chunks: list[Chunk]
) -> None:
    messages = _FakeMessages('{"unsupported_claims": [], "score": 0.75}')
    provider = _fake_anthropic(monkeypatch, messages)
    assert provider.faithfulness("Cursors are opaque [doc#h005].", toy_chunks) == 0.75
    assert messages.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_judge_refusal_scores_zero(
    monkeypatch: pytest.MonkeyPatch, toy_chunks: list[Chunk]
) -> None:
    provider = _fake_anthropic(monkeypatch, _FakeMessages("", stop_reason="refusal"))
    assert provider.faithfulness("Cursors are opaque [doc#h005].", toy_chunks) == 0.0


def test_anthropic_refusal_stop_reason_becomes_not_in_docs(
    monkeypatch: pytest.MonkeyPatch, toy_chunks: list[Chunk]
) -> None:
    provider = _fake_anthropic(monkeypatch, _FakeMessages("", stop_reason="refusal"))
    assert provider.answer("anything", toy_chunks) == NOT_IN_DOCS
