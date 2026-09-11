"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# backend/app/core/config.py -> backend/
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Every tunable value the backend needs. Nothing is hard-coded elsewhere."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Document Assistant"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Comma-separated list, e.g. "http://localhost:8501,http://127.0.0.1:8501"
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    # --- Vector store (produced by notebooks/rag_pipeline.ipynb) -----------
    vector_store_dir: Path = BACKEND_ROOT / "data" / "vector_store"
    collection_name: str = "faa_handbooks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval --------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=20)
    # Cosine similarity below this means "nothing in the corpus is relevant".
    # This is what stops the assistant answering from the LLM's own memory.
    min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Generation (Ollama) ---------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 120.0
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    ollama_num_ctx: int = 4096

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def apply_vector_store_manifest(self) -> dict | None:
        """
        Read config.json written by the notebook alongside the vector store.

        The embedding model used for the query MUST match the one used to build
        the index, otherwise retrieval silently returns nonsense. The notebook
        records what it used; we adopt it and warn on any mismatch.
        """
        manifest_path = self.vector_store_dir / "config.json"
        if not manifest_path.is_file():
            logger.warning(
                "No config.json next to the vector store (%s). "
                "Falling back to configured defaults.",
                manifest_path,
            )
            return None

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for field in ("embedding_model", "collection_name"):
            recorded = manifest.get(field)
            if recorded and recorded != getattr(self, field):
                logger.warning(
                    "Overriding %s=%r with %r from the vector store manifest.",
                    field,
                    getattr(self, field),
                    recorded,
                )
                setattr(self, field, recorded)

        return manifest


@lru_cache
def get_settings() -> Settings:
    return Settings()
