"""The two HTTP endpoints: GET /health and POST /query."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import get_settings
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, SourceDetail
from app.services.generation import Generator, used_markers
from app.services.retrieval import Retriever

logger = logging.getLogger(__name__)
router = APIRouter()


# The retriever and generator are built once during startup and parked on
# app.state. Reading them from the request keeps the route functions easy to
# test - a test can just put fake objects on app.state.
def _retriever(request: Request) -> Retriever:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None or not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not loaded. Check the server logs and /health.",
        )
    return retriever


def _generator(request: Request) -> Generator:
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model client is not initialised.",
        )
    return generator


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Liveness + readiness. The frontend calls this before enabling the input box."""
    settings = get_settings()
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)

    ready = retriever is not None and retriever.is_ready
    llm_ok = generator.is_reachable() if generator is not None else False

    return HealthResponse(
        status="ok" if (ready and llm_ok) else "degraded",
        version=settings.app_version,
        vector_store_loaded=ready,
        chunk_count=retriever.chunk_count if ready else 0,
        collection=settings.collection_name,
        embedding_model=settings.embedding_model,
        llm_model=settings.ollama_model,
        llm_reachable=llm_ok,
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """
    The RAG pipeline, end to end:

        question -> retrieve chunks -> build prompt -> LLM -> grounded answer
    """
    settings = get_settings()
    retriever = _retriever(request)
    generator = _generator(request)

    started = time.perf_counter()
    top_k = payload.top_k or settings.top_k

    # 1. RETRIEVE
    chunks = retriever.search(payload.question, top_k=top_k)

    # 2. GENERATE (returns the refusal message if chunks is empty)
    try:
        answer, grounded = generator.generate(payload.question, chunks)
    except Exception as exc:
        logger.exception("Generation failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not reach the language model at {settings.ollama_host}. "
                f"Is `ollama serve` running and is the model '{settings.ollama_model}' pulled? "
                f"({exc.__class__.__name__})"
            ),
        ) from exc

    # 3. ATTACH CITATIONS
    # Only report the sources the model actually cited. Listing all 5 when it
    # used 2 makes the citation display misleading.
    details: list[SourceDetail] = []
    if grounded:
        cited = used_markers(answer) or set(range(1, len(chunks) + 1))
        for index, chunk in enumerate(chunks, start=1):
            if index not in cited:
                continue
            details.append(
                SourceDetail(
                    marker=f"[{index}]",
                    document=chunk.document,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_id=chunk.chunk_id,
                    similarity=chunk.similarity,
                    snippet=chunk.text[:400].strip(),
                )
            )

    return QueryResponse(
        answer=answer,
        sources=[
            f"{d.marker} {d.document}" + (f", p. {d.page}" if d.page else "")
            for d in details
        ],
        source_details=details,
        grounded=grounded,
        model=settings.ollama_model,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
