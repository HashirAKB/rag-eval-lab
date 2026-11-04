from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_eval_lab.api import create_app
from rag_eval_lab.pipeline import Config


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    # BM25 keeps this test fast and offline; the code path is the same for every config.
    app = create_app(Config("heading", "bm25"))
    with TestClient(app) as test_client:  # the context manager runs the startup lifespan
        yield test_client


def test_search_returns_ranked_chunks_with_metadata(client: TestClient) -> None:
    response = client.get("/search", params={"q": "stdio newline delimited messages", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["config"] == "heading/bm25"
    assert len(body["hits"]) == 3
    top = body["hits"][0]
    assert set(top) == {"chunk_id", "doc_path", "heading_path", "score", "text"}
    assert "transports" in top["doc_path"]
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)


def test_ask_returns_grounded_answer_with_citations(client: TestClient) -> None:
    response = client.post(
        "/ask", json={"question": "How must a client cancel a request on stdio?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "mock"
    assert not body["refused"]
    assert body["citations"]
    hit_ids = {h["chunk_id"] for h in body["hits"]}
    assert set(body["citations"]) <= hit_ids
    for chunk_id in body["citations"]:
        assert f"[{chunk_id}]" in body["answer"]


def test_ask_refuses_out_of_scope_questions(client: TestClient) -> None:
    body = client.post(
        "/ask", json={"question": "How do I deploy MCP on Kubernetes with Helm?"}
    ).json()
    assert body["refused"]
    assert body["citations"] == []
    assert body["answer"].startswith("Not in the docs")


def test_input_validation(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>RAG Eval Lab</title>" in response.text
    assert 'fetch("/ask"' in response.text


def test_config_from_name_round_trips() -> None:
    for name in ["heading/bm25", "fixed/hybrid+rerank", "heading/dense-minilm"]:
        assert Config.from_name(name).name == name
    with pytest.raises(ValueError):
        Config.from_name("heading/magic")
