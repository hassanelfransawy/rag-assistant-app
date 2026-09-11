"""Request / response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="A natural-language question about the document corpus.",
        examples=["What is the difference between indicated and true airspeed?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override how many chunks to retrieve. Defaults to server config.",
    )


class SourceDetail(BaseModel):
    """Full provenance for one retrieved chunk, used by the frontend."""

    marker: str = Field(..., description="Citation marker as it appears in the answer, e.g. '[1]'")
    document: str = Field(..., description="Human-readable document title")
    page: int | None = Field(default=None, description="1-indexed page number in the source PDF")
    section: str | None = Field(default=None, description="Nearest section heading, if detected")
    chunk_id: str
    similarity: float = Field(..., description="Cosine similarity to the question, 0..1")
    snippet: str = Field(..., description="The retrieved text the answer was grounded in")


class QueryResponse(BaseModel):
    answer: str
    # Required by the project spec: a plain list of readable citation strings.
    sources: list[str] = Field(default_factory=list)
    # Richer provenance so the UI can show snippets and scores.
    source_details: list[SourceDetail] = Field(default_factory=list)
    grounded: bool = Field(
        ...,
        description="False when no chunk cleared the similarity floor; the assistant declines.",
    )
    model: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_loaded: bool
    chunk_count: int
    collection: str
    embedding_model: str
    llm_model: str
    llm_reachable: bool
