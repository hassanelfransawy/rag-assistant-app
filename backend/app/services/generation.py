"""
Generation: turn retrieved chunks + the question into a grounded answer.

THE IDEA
--------
We never ask the LLM "what is X?". We ask it:

    "Here are 5 numbered passages from our documents. Using ONLY these,
     answer the question, and put [1] / [2] next to the facts you used."

That single change is the whole point of RAG. The model stops answering from
its (unreliable, frozen) memory and starts answering from our documents, and
because it cites markers, a human can verify every claim.
"""

from __future__ import annotations

import logging
import re

from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# The message we return when retrieval found nothing relevant. We do NOT call
# the LLM in that case - calling it is exactly how hallucinations get in.
NO_CONTEXT_ANSWER = (
    "I could not find anything about that in the documents I have. "
    "This assistant only answers from its indexed source documents, so I would "
    "rather say nothing than guess."
)

SYSTEM_PROMPT = """You are a careful assistant that answers questions using ONLY the numbered context passages provided by the user.

Rules you must follow:
1. Use only facts stated in the context. Never add outside knowledge.
2. After each fact, cite the passage it came from using its marker, like [1] or [2].
3. If the context does not contain the answer, reply exactly: NOT_IN_CONTEXT
4. Do not invent numbers, names, or regulations. Quote the context's figures exactly.
5. Be concise: 2-5 sentences unless the question needs a short list."""

USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer using only the passages above, citing them with [n] markers."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format chunks as a numbered block the model can cite by number."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        where = chunk.document
        if chunk.page is not None:
            where += f", page {chunk.page}"
        if chunk.section:
            where += f", section '{chunk.section}'"
        parts.append(f"[{index}] (Source: {where})\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Exposed separately so the notebook and tests can inspect the exact prompt."""
    return USER_TEMPLATE.format(
        context=build_context_block(chunks), question=question.strip()
    )


def used_markers(answer: str) -> set[int]:
    """Which [n] markers the model actually cited, so we can drop unused sources."""
    return {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}


class Generator:
    """Thin wrapper around a local Ollama chat model."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.1,
        num_ctx: int = 4096,
    ) -> None:
        # Imported here rather than at module level so the tests (which use a
        # fake generator) do not need the ollama package installed.
        import ollama

        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        # One client reused for every request (connection pooling).
        self._client = ollama.Client(host=host, timeout=timeout)

    def is_reachable(self) -> bool:
        """Used by /health so the frontend can show a useful error."""
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, bool]:
        """
        Returns (answer, grounded).

        grounded=False means we refused to answer because retrieval came back
        empty, or the model itself reported the context was insufficient.
        """
        if not chunks:
            # Short-circuit: no context means no grounded answer is possible.
            return NO_CONTEXT_ANSWER, False

        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks)},
            ],
            options={
                # Low temperature: we want faithful extraction, not creativity.
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        )

        answer = response["message"]["content"].strip()

        if "NOT_IN_CONTEXT" in answer:
            logger.info("Model reported the retrieved context was insufficient.")
            return NO_CONTEXT_ANSWER, False

        return answer, True
