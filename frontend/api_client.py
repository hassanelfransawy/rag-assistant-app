"""
A thin wrapper around the backend HTTP API.

Keeping every network call in this one file means app.py contains only UI code,
and it is the only place the backend URL appears. The URL comes from an
environment variable - never hard-code http://localhost:8000 in the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))


class BackendError(Exception):
    """Raised when the backend is unreachable or returns an error."""


@dataclass
class Answer:
    text: str
    sources: list[dict]
    grounded: bool
    model: str
    latency_ms: int


def get_health() -> dict:
    """Ask the backend whether it is ready. Used to show a status badge."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise BackendError(
            f"Cannot reach the backend at {API_BASE_URL}. "
            "Is it running? Start it with: uvicorn app.main:app --reload"
        ) from exc


def ask(question: str, top_k: int | None = None) -> Answer:
    """Send a question to POST /query and return the parsed answer."""
    payload: dict = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k

    try:
        response = requests.post(
            f"{API_BASE_URL}/query", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.Timeout as exc:
        raise BackendError(
            "The backend took too long to answer. A local LLM can be slow on "
            "the first request while the model loads into memory - try again."
        ) from exc
    except requests.RequestException as exc:
        raise BackendError(f"Cannot reach the backend at {API_BASE_URL}.") from exc

    if response.status_code == 422:
        raise BackendError("That question was rejected: it must be 3-1000 characters.")

    if not response.ok:
        # FastAPI puts the useful message in the "detail" field.
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise BackendError(f"Backend error {response.status_code}: {detail}")

    body = response.json()
    return Answer(
        text=body["answer"],
        sources=body.get("source_details", []),
        grounded=body.get("grounded", False),
        model=body.get("model", "unknown"),
        latency_ms=body.get("latency_ms", 0),
    )
