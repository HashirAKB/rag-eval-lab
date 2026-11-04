import importlib.util

import pytest

from rag_eval_lab.chunking import Chunk


def make_chunk(i: int, text: str) -> Chunk:
    return Chunk(f"doc#h{i:03d}", "doc.mdx", ("Doc",), 0, len(text), text)


TOY_TEXTS = [
    "Servers expose tools that language models can call to query databases.",
    "Resources are files or data that a server shares with the client for context.",
    "Prompts are reusable message templates that users pick from a menu.",
    "The stdio transport launches the server as a subprocess and talks over stdin and stdout.",
    "Cancellation lets either side abort an in-flight request with a notification.",
    "Pagination uses an opaque cursor to fetch large result lists page by page.",
]


@pytest.fixture
def toy_chunks() -> list[Chunk]:
    return [make_chunk(i, t) for i, t in enumerate(TOY_TEXTS)]


def neural_available() -> bool:
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    try:
        from rag_eval_lab.models import load_embedding_model

        load_embedding_model()
    except Exception:  # no network and no cached model
        return False
    return True


requires_neural = pytest.mark.skipif(not neural_available(), reason="local models unavailable")
