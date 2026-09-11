"""
Shared test fixtures.

The tests must NOT need a real vector store or a running Ollama - otherwise
they would be slow and would fail on a fresh clone. So we swap in tiny fake
objects that behave like the real Retriever and Generator.

This is why the routes read `request.app.state.retriever` instead of building
their own: it makes the real thing replaceable in one line.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.generation import NO_CONTEXT_ANSWER
from app.services.retrieval import RetrievedChunk

SAMPLE_CHUNKS = [
    RetrievedChunk(
        chunk_id="phak::p142::c0",
        text=(
            "Indicated airspeed (IAS) is the speed shown on the airspeed indicator. "
            "True airspeed (TAS) is the speed of the aircraft relative to the air "
            "mass, and increases with altitude for a given indicated airspeed."
        ),
        document="Pilot's Handbook of Aeronautical Knowledge",
        page=142,
        section="Airspeed Indicator",
        similarity=0.81,
    ),
    RetrievedChunk(
        chunk_id="phak::p143::c1",
        text="Calibrated airspeed is indicated airspeed corrected for installation error.",
        document="Pilot's Handbook of Aeronautical Knowledge",
        page=143,
        section="Airspeed Indicator",
        similarity=0.64,
    ),
]


class FakeRetriever:
    """Stands in for Retriever. `chunks` controls what a search returns."""

    def __init__(self, chunks=None):
        self.chunks = SAMPLE_CHUNKS if chunks is None else chunks
        self.is_ready = True

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def search(self, question: str, top_k: int):
        return self.chunks[:top_k]


class FakeGenerator:
    """Stands in for Generator. Mirrors the real refusal behaviour."""

    def __init__(self, answer: str = "TAS increases with altitude for a given IAS [1]."):
        self.answer = answer

    def is_reachable(self) -> bool:
        return True

    def generate(self, question: str, chunks):
        if not chunks:
            return NO_CONTEXT_ANSWER, False
        return self.answer, True


@pytest.fixture
def client():
    """
    A TestClient with fake services attached.

    Note: we deliberately do NOT use `with TestClient(app)`. The `with` form
    runs the real startup lifespan, which would load the actual embedding
    model and vector store.
    """
    app.state.retriever = FakeRetriever()
    app.state.generator = FakeGenerator()
    yield TestClient(app)
    app.state.retriever = None
    app.state.generator = None
