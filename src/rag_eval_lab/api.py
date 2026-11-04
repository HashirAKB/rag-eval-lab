"""HTTP API: /search (retrieval only), /ask (grounded answer), / (a minimal HTML page).

The pipeline is built once at startup. Pick the configuration with the RAG_CONFIG
environment variable, e.g. RAG_CONFIG=heading/bm25 (default: the eval winner).
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag_eval_lab.answering import Answerer, ConfidenceScorer
from rag_eval_lab.llm import get_provider
from rag_eval_lab.pipeline import DEFAULT_CONFIG, Config, build_pipeline
from rag_eval_lab.retrievers import Hit


class HitOut(BaseModel):
    chunk_id: str
    doc_path: str
    heading_path: list[str]
    score: float
    text: str


class SearchResponse(BaseModel):
    config: str
    hits: list[HitOut]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    config: str
    provider: str
    answer: str
    citations: list[str]
    refused: bool
    confidence: float
    hits: list[HitOut]


def _hit_out(hit: Hit) -> HitOut:
    c = hit.chunk
    return HitOut(
        chunk_id=c.chunk_id,
        doc_path=c.doc_path,
        heading_path=list(c.heading_path),
        score=hit.score,
        text=c.text,
    )


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config.from_name(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG.name))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        chunks, retriever = build_pipeline(config)
        provider = get_provider()
        app.state.retriever = retriever
        app.state.provider = provider
        app.state.answerer = Answerer(retriever, provider, ConfidenceScorer(chunks))
        yield

    app = FastAPI(title="RAG Eval Lab", lifespan=lifespan)

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request, q: str = Query(min_length=1), k: int = Query(5, ge=1, le=20)
    ) -> SearchResponse:
        hits = request.app.state.retriever.search(q, k)
        return SearchResponse(config=config.name, hits=[_hit_out(h) for h in hits])

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest) -> AskResponse:
        answer = request.app.state.answerer.ask(body.question)
        return AskResponse(
            config=config.name,
            provider=request.app.state.provider.name,
            answer=answer.text,
            citations=answer.citations,
            refused=answer.refused,
            confidence=answer.confidence,
            hits=[_hit_out(h) for h in answer.hits],
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("rag_eval_lab").joinpath("static/index.html").read_text(encoding="utf-8")

    return app


app = create_app()
