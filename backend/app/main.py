"""
FastAPI application: CORS, startup loading, and the /health + /query routes.

Run it with:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.generation import Generator
from app.services.retrieval import Retriever
from app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE when the server starts, before it accepts any request.

    This is where the embedding model and the vector store get loaded. Doing it
    here (instead of inside the /query handler) is the difference between a
    ~200 ms query and a ~6 s query.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Adopt whatever the notebook recorded, so query-time embeddings always
    # match index-time embeddings.
    settings.apply_vector_store_manifest()

    app.state.retriever = Retriever(
        vector_store_dir=settings.vector_store_dir,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        min_similarity=settings.min_similarity,
    )
    try:
        app.state.retriever.load()
    except Exception:
        # Start anyway so /health can explain what is wrong instead of the
        # server simply refusing to boot with a stack trace.
        logger.exception("Could not load the vector store. /query will return 503.")

    app.state.generator = Generator(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
        temperature=settings.ollama_temperature,
        num_ctx=settings.ollama_num_ctx,
    )
    if not app.state.generator.is_reachable():
        logger.warning(
            "Ollama is not reachable at %s. Start it with `ollama serve`.",
            settings.ollama_host,
        )

    yield

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A Retrieval-Augmented Generation API over a corpus of FAA aviation "
            "handbooks. Answers are generated only from retrieved passages and "
            "carry [n] citations back to the source document and page."
        ),
        lifespan=lifespan,
    )

    # The Streamlit frontend runs on a different port, so the browser treats it
    # as a different origin and blocks the request without this.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(query_router)
    return app


app = create_app()
