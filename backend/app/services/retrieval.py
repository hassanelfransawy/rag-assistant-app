"""
Retrieval: turn a question into the most relevant chunks of our documents.

THE IDEA
--------
The notebook already did the expensive work: it split the PDFs into chunks,
turned each chunk into a vector (a list of 384 numbers that captures meaning),
and saved them into a Chroma database on disk.

At request time all we do is:
    1. Turn the *question* into a vector using the SAME model.
    2. Ask Chroma for the chunks whose vectors point in the most similar
       direction (cosine similarity).
    3. Throw away anything that isn't similar enough.

Step 3 is the important one. Without it, a question about cooking would still
return the 5 "least bad" aviation chunks, and the LLM would try to answer from
them. Returning nothing is what lets the assistant honestly say "I don't know".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# chromadb and sentence_transformers are imported inside load() rather than
# here on purpose. They pull in torch and take ~10 s to import, and the tests
# never touch them (they use a fake retriever). Importing lazily keeps
# `pytest` fast and lets it run without the ML stack installed.
if TYPE_CHECKING:  # pragma: no cover - type hints only
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One piece of a document that we think answers the question."""

    chunk_id: str
    text: str
    document: str
    page: int | None
    section: str | None
    similarity: float  # 1.0 = identical meaning, 0.0 = unrelated


class Retriever:
    """Loads the vector store once at startup and answers searches from memory."""

    def __init__(
        self,
        vector_store_dir: Path,
        collection_name: str,
        embedding_model: str,
        min_similarity: float,
    ) -> None:
        self.vector_store_dir = Path(vector_store_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.min_similarity = min_similarity

        self._model: "SentenceTransformer | None" = None
        self._collection: Any = None

    # -- startup ----------------------------------------------------------

    def load(self) -> None:
        """
        Load the embedding model and open the Chroma collection.

        This is called ONCE from the FastAPI lifespan, not per request.
        Loading the model takes a few seconds and ~90 MB of RAM; doing it per
        request would make every query unusably slow.
        """
        import chromadb
        from sentence_transformers import SentenceTransformer

        if not self.vector_store_dir.is_dir():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_dir}. "
                "Run notebooks/rag_pipeline.ipynb top-to-bottom first - its last "
                "section exports the store into backend/data/vector_store/."
            )

        logger.info("Loading embedding model %s ...", self.embedding_model_name)
        self._model = SentenceTransformer(self.embedding_model_name)

        logger.info("Opening Chroma collection %r ...", self.collection_name)
        client = chromadb.PersistentClient(path=str(self.vector_store_dir))
        try:
            self._collection = client.get_collection(self.collection_name)
        except Exception as exc:  # chromadb raises different types across versions
            # list_collections() returns objects in some versions and plain
            # strings in others, so handle both rather than crash inside the
            # error handler.
            available = [
                getattr(c, "name", c) for c in client.list_collections()
            ]
            raise RuntimeError(
                f"Collection {self.collection_name!r} not found in {self.vector_store_dir}. "
                f"Collections present: {available or 'none'}."
            ) from exc

        logger.info("Vector store ready: %d chunks indexed.", self.chunk_count)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._collection is not None

    @property
    def chunk_count(self) -> int:
        return self._collection.count() if self._collection is not None else 0

    # -- query time -------------------------------------------------------

    def search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return the chunks above the similarity floor, best first."""
        if not self.is_ready:
            raise RuntimeError("Retriever.load() was never called.")

        # normalize_embeddings=True makes every vector length 1, which is what
        # makes the cosine-distance maths below valid.
        question_vector = self._model.encode(
            question, normalize_embeddings=True
        ).tolist()

        result = self._collection.query(
            query_embeddings=[question_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns one list per query; we only sent one query.
        ids = result["ids"][0]
        texts = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(ids, texts, metadatas, distances):
            # The collection was created with cosine space, where
            #     distance = 1 - cosine_similarity
            # so we invert it to get a score that is nicer to read and threshold.
            similarity = 1.0 - float(distance)

            if similarity < self.min_similarity:
                continue

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    document=str(meta.get("document", "unknown")),
                    page=int(meta["page"]) if meta.get("page") is not None else None,
                    section=meta.get("section") or None,
                    similarity=round(similarity, 4),
                )
            )

        logger.info(
            "Retrieved %d/%d chunks above the %.2f similarity floor.",
            len(chunks),
            len(ids),
            self.min_similarity,
        )
        return chunks
