"""
Streamlit chat UI for the RAG assistant.

Run with:
    streamlit run app.py

All it does is: take a question, call the backend, and render the answer
together with the sources it was grounded in.
"""

from __future__ import annotations

import streamlit as st

from api_client import API_BASE_URL, Answer, BackendError, ask, get_health

st.set_page_config(page_title="Aviation Handbook Assistant", page_icon="✈", layout="centered")

EXAMPLE_QUESTIONS = [
    "What is the difference between indicated airspeed and true airspeed?",
    "What causes a wing to stall?",
    "How do you calculate an aircraft's centre of gravity?",
    "What weather conditions produce carburettor icing?",
]


# --------------------------------------------------------------------------
# Sidebar: backend status. Showing this means a broken demo explains itself.
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Backend status")
    st.caption(f"`{API_BASE_URL}`")

    try:
        health = get_health()
        if health["status"] == "ok":
            st.success("Ready")
        else:
            st.warning("Degraded")

        st.metric("Chunks indexed", f"{health['chunk_count']:,}")
        st.write(f"**LLM:** `{health['llm_model']}`")
        st.write(f"**Embeddings:** `{health['embedding_model'].split('/')[-1]}`")

        if not health["vector_store_loaded"]:
            st.error("Vector store not loaded. Run the notebook first.")
        if not health["llm_reachable"]:
            st.error("Ollama unreachable. Run `ollama serve`.")

    except BackendError as exc:
        st.error(str(exc))

    st.divider()
    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many document chunks to feed the model as context.",
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
st.title("✈ Aviation Handbook Assistant")
st.caption(
    "Ask about flight training, aircraft systems, weather or weight & balance. "
    "Every answer is generated only from the FAA handbooks in the index, with "
    "citations you can check."
)

if "history" not in st.session_state:
    st.session_state.history = []

# Example question buttons make the live demo smooth.
if not st.session_state.history:
    st.write("**Try one of these:**")
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(example, use_container_width=True, key=f"ex{index}"):
            st.session_state.pending = example
            st.rerun()


def render_answer(answer: Answer) -> None:
    """Render one assistant reply plus its sources."""
    if answer.grounded:
        st.markdown(answer.text)
    else:
        # Visually distinct: this is the assistant refusing, not failing.
        st.info(answer.text)

    st.caption(f"{answer.model} · {answer.latency_ms / 1000:.1f}s")

    if answer.sources:
        with st.expander(f"Sources ({len(answer.sources)})", expanded=True):
            for source in answer.sources:
                location = source["document"]
                if source.get("page"):
                    location += f" — page {source['page']}"
                if source.get("section"):
                    location += f" — *{source['section']}*"

                st.markdown(f"**{source['marker']}** {location}")
                st.progress(
                    min(source["similarity"], 1.0),
                    text=f"relevance {source['similarity']:.2f}",
                )
                st.caption(f"> {source['snippet']}")
                st.divider()


# Replay the conversation so far.
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        render_answer(entry["answer"])


question = st.chat_input("Ask a question about the handbooks...")

# An example button click is handled the same way as typed input.
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the handbooks and generating an answer..."):
            try:
                answer = ask(question, top_k=top_k)
            except BackendError as exc:
                st.error(str(exc))
                answer = None

        if answer is not None:
            render_answer(answer)
            st.session_state.history.append({"question": question, "answer": answer})
