"""Tests for /health and /query."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services.generation import (
    NO_CONTEXT_ANSWER,
    build_prompt,
    used_markers,
)
from tests.conftest import SAMPLE_CHUNKS, FakeGenerator, FakeRetriever


# --- /health ---------------------------------------------------------------


def test_health_reports_ready(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_store_loaded"] is True
    assert body["chunk_count"] == len(SAMPLE_CHUNKS)


# --- /query happy path -----------------------------------------------------


def test_query_returns_grounded_answer_with_sources(client):
    response = client.post(
        "/query",
        json={"question": "What is the difference between indicated and true airspeed?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["grounded"] is True
    assert body["answer"]
    # The fake answer cites [1], so exactly one source should come back.
    assert len(body["sources"]) == 1
    assert "Pilot's Handbook" in body["sources"][0]
    assert body["source_details"][0]["page"] == 142
    assert body["source_details"][0]["marker"] == "[1]"
    assert body["latency_ms"] >= 0


# --- /query invalid input --------------------------------------------------


def test_query_rejects_too_short_question(client):
    """min_length=3 on the schema means FastAPI validates before our code runs."""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 422


def test_query_rejects_missing_question_field(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_rejects_out_of_range_top_k(client):
    response = client.post("/query", json={"question": "valid question", "top_k": 99})
    assert response.status_code == 422


# --- /query grounding behaviour -------------------------------------------


def test_query_declines_when_nothing_relevant_is_retrieved():
    """
    The core anti-hallucination guarantee: if retrieval returns nothing, the
    API must refuse rather than let the LLM answer from its own memory.
    """
    app.state.retriever = FakeRetriever(chunks=[])
    app.state.generator = FakeGenerator()
    local_client = TestClient(app)

    response = local_client.post(
        "/query", json={"question": "What is the best recipe for koshari?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["sources"] == []


def test_query_returns_503_when_vector_store_missing():
    app.state.retriever = None
    app.state.generator = FakeGenerator()
    local_client = TestClient(app)

    response = local_client.post("/query", json={"question": "a valid question"})
    assert response.status_code == 503


# --- prompt construction ---------------------------------------------------


def test_prompt_numbers_every_passage_and_includes_the_question():
    prompt = build_prompt("What is TAS?", SAMPLE_CHUNKS)

    assert "[1]" in prompt and "[2]" in prompt
    assert "page 142" in prompt
    assert "What is TAS?" in prompt


def test_used_markers_extracts_citations():
    assert used_markers("Facts here [1] and there [3].") == {1, 3}
    assert used_markers("No citations at all.") == set()
