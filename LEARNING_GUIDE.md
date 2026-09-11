# Learning Guide

Everything in this project explained in plain language — the concepts, then every file, then the questions your instructors will ask.

Read this once before you run anything, and again the night before your demo.

---

## Part 1 — What problem does RAG solve?

A language model is a very well-read person with **no access to any book**. It answers from memory. That memory is:

- **frozen** at training time,
- **lossy** — it remembers the gist, not the exact figure,
- **confident regardless** — it does not know what it does not know.

Ask `llama3.2:3b` "what's the maximum altitude for a small drone under Part 107?" and it will give you a number, stated firmly. It might be right. You have no way to tell.

**RAG (Retrieval-Augmented Generation) changes the question being asked.** Instead of:

> "What's the maximum altitude under Part 107?"

we ask:

> "Here are five paragraphs from the FAA Remote Pilot Study Guide. Using **only** these, what's the maximum altitude under Part 107? Cite which paragraph each fact came from."

Now the model is doing **reading comprehension**, not recall. And because it cites, you can check it.

That is the whole idea. Everything else is plumbing to find the right five paragraphs.

---

## Part 2 — The five concepts you need

### 1. Embeddings — turning meaning into numbers

An embedding model reads text and outputs a list of numbers (here: 384 of them). Think of it as **a point in space**, where texts that mean similar things land near each other.

```
"What makes a wing stop flying?"     → [0.21, -0.04, 0.88, ...]
"critical angle of attack, stall"    → [0.19, -0.07, 0.85, ...]   ← close by
"best recipe for koshari"            → [-0.6,  0.31, 0.02, ...]   ← far away
```

This is why the search works even when **no words match**. Keyword search for "stop flying" finds nothing. Vector search finds the stall paragraph, because it understands they *mean* the same thing.

### 2. Cosine similarity — measuring "how close"

Two vectors point in some direction. Cosine similarity measures the **angle** between them:

| Similarity | Meaning |
|---|---|
| `1.0` | identical direction — same meaning |
| `0.7` | strongly related |
| `0.3` | vaguely related |
| `0.0` | unrelated |

We normalise every vector to length 1 (`normalize_embeddings=True`), which makes the maths clean: Chroma's cosine *distance* becomes exactly `1 - similarity`. **I verified this empirically** rather than trusting the docs — that check is why `similarity = 1.0 - distance` appears throughout the code.

### 3. Chunking — why we cut the books up

Two reasons, both hard limits:

- **The model can only read a few thousand characters at once.** A 520-page handbook is millions.
- **One vector = one meaning.** A vector for a whole chapter is the *average* of fifty topics, so it matches nothing sharply. A vector for one paragraph is precise.

So we split documents into ~1000-character pieces. The trade-off:

- **Too small** (200 chars) → a chunk holds a fragment with no context. "It is calibrated in knots." *What is?*
- **Too large** (5000 chars) → the vector blurs across topics, and you waste context window on irrelevant text.

**Overlap** (150 chars) exists because a fact can land exactly on a boundary. Without overlap, a definition split down the middle is retrievable from neither half.

### 4. The vector store — a database that searches by meaning

ChromaDB holds every chunk's text, its vector, and its metadata (document, page, section). Given a query vector it returns the nearest chunks, fast.

Crucially it **persists to disk**. The notebook builds it once; the backend just opens it.

### 5. Grounding — the part most projects get wrong

Retrieval **always returns something**. Ask about cooking, and you still get the five least-irrelevant aviation chunks. Hand those to the LLM and it will produce a confident, nonsense answer.

This project has **two defences**:

1. **The similarity floor** (`MIN_SIMILARITY = 0.25`). Anything below it is dropped. If nothing survives, we never call the LLM at all — we return "I don't know."
2. **The `NOT_IN_CONTEXT` instruction.** For the harder case where chunks *look* relevant but do not contain the answer, the model is told to say so explicitly.

> **This is your project's best talking point.** Most student RAG projects have neither. Demonstrate it live: ask the assistant something about football and let it refuse.

---

## Part 3 — Walking through the code

### The offline half: `notebooks/rag_pipeline.ipynb`

Runs once. Turns PDFs into a searchable index.

| Section | What it does | The idea worth understanding |
|---|---|---|
| **2.1 Load & Inspect** | Parses PDFs page by page; measures empty pages | Empty pages = a scanned PDF = you need OCR. Measuring it *proves* our corpus is clean rather than assuming it |
| **2.1 (cleaning)** | Removes headers, rejoins hyphens, fixes ligatures | The boilerplate detector is **frequency-based**: any short line appearing on >25% of pages is page furniture. No hardcoded patterns, so it works on any corpus |
| **2.2 Chunking** | Paragraph-aware splitting with section tracking | Fills chunks with *whole paragraphs*, only falling back to sentences when one paragraph is oversized. So chunks rarely start mid-sentence |
| **2.3 Embeddings** | Encodes all chunks, stores in Chroma | `metadata={"hnsw:space": "cosine"}` — the default is L2, which would silently break the similarity maths |
| **2.4 Retrieval** | Query → chunks → prompt | Mirrors `backend/app/services/` exactly, so what you validate here is what ships |
| **2.6 Evaluation** | 14 questions, results table, failure analysis | 3 questions are out-of-domain on purpose. An assistant that answers those is *worse* than one that refuses |
| **2.7 Export** | Writes `config.json` beside the store | Records which embedding model built the index, so the backend can't accidentally use a different one |

### The online half: `backend/`

```
Request → main.py → routes/query.py → retrieval.py → generation.py → Response
```

**`app/main.py`** — creates the FastAPI app. The important part is the **lifespan**: code that runs once at startup, before any request. That's where the embedding model and vector store load. Doing it per-request would add ~6 seconds to every query.

**`app/core/config.py`** — every tunable value in one place, read from `.env`. Nothing is hard-coded elsewhere. `apply_vector_store_manifest()` reads the notebook's `config.json` and adopts its settings — protection against the nastiest bug in RAG, where index-time and query-time use different embedding models and retrieval silently returns garbage with **no error**.

**`app/services/retrieval.py`** — embeds the question, searches Chroma, converts distance to similarity, drops anything below the floor. Note the heavy imports (`chromadb`, `sentence_transformers`) happen *inside* `load()`, not at the top of the file — that is why `pytest` runs in 0.06 s instead of needing a 2 GB torch download.

**`app/services/generation.py`** — builds the numbered prompt and calls Ollama. Read `SYSTEM_PROMPT` closely; those five rules are where grounding actually happens. `used_markers()` extracts which `[n]` the model cited, so we only list sources it really used.

**`app/api/routes/query.py`** — the two endpoints. Note that it reads services from `request.app.state` rather than constructing them. That one choice is what makes the tests fast and simple.

**`app/schemas/query.py`** — Pydantic models. `question: str = Field(min_length=3, max_length=1000)` is doing real work: FastAPI rejects a bad request with a `422` **before your code runs**. That's the "invalid input" test.

### The UI: `frontend/`

**`api_client.py`** — every network call. `API_BASE_URL` comes from an environment variable, never hard-coded. *(The brief lists hard-coding it as a mistake that loses points.)*

**`app.py`** — Streamlit UI only, no networking. Shows a backend status badge, example-question buttons for a smooth demo, the answer, and an expandable source panel with relevance bars.

---

## Part 4 — Your 4 days

| Day | Do this | Done when |
|---|---|---|
| **1** | Install everything. `ollama serve`, `ollama pull llama3.2:3b`. Run `download_corpus.py`. Run the notebook through §2.3 | Chroma reports ~15,000 chunks |
| **2** | Run the notebook through §2.7. Read every answer in §2.6 and **hand-correct the `correct` column**. Rewrite the failure analysis with what *you* saw | `reports/evaluation_results.csv` exists |
| **3** | Start the backend, test `/docs`, run `pytest`. Start the frontend, ask real questions. Take the 4 screenshots | End-to-end demo works |
| **4** | Paste real evaluation numbers into the README. `pip freeze > requirements.txt` in both folders. Push to GitHub. **Clone into a fresh folder and follow only your README** | A stranger could run it |

> Day 4's clone-test is the highest-value hour you will spend. It is exactly how your project will be graded, and it always finds a missing step.

---

## Part 5 — Demo questions and how to answer them

**"Why chunk size 1000?"**
> Two constraints met in the middle. The embedding model truncates past 256 tokens (~1000 characters), so anything larger would have its tail silently ignored during embedding — indexed on only part of its own content. And 1000 characters is roughly one or two paragraphs, which is one complete idea. Five of them fit comfortably in a 4096-token context alongside the prompt.

**"Why 150 overlap?"**
> 15%. A fact that straddles a chunk boundary would otherwise be retrievable from neither half. Below about 10% facts get cut; above 25% the index fills with near-duplicates that crowd out genuinely different results in the top-5.

**"How do you know it isn't hallucinating?"**
> Three things. Every answer carries `[n]` markers back to a document and page, so it's checkable. Anything below 0.25 similarity is dropped before the model ever sees it, and if nothing survives, the model isn't called at all. And the evaluation set includes three out-of-domain questions where the correct behaviour is refusal — *[then demo it live]*.

**"What if the answer is spread across two documents?"**
> It works — retrieval is over all chunks globally, so top-5 can span documents, and each gets its own citation. Where it struggles is questions needing *synthesis* across many chunks, like "compare every airspeed definition." That's a known limitation of single-pass retrieval; a reranker or query decomposition would help.

**"Why ChromaDB and not FAISS or Pinecone?"**
> Chroma stores text, vectors and metadata together and persists with one line, so the backend just opens a folder. FAISS only stores vectors — I'd have to maintain a separate metadata store and keep the two in sync. Pinecone is hosted, needs an API key, and the brief requires everything local.

**"What's your biggest weakness?"**
> Tables. `pypdf` flattens weight-and-balance tables into a run of digits with no column structure, so a retrieved table chunk is nearly useless. The honest fix is a layout-aware parser like `unstructured` or `camelot`. I documented it rather than hiding it.

**"Why is the vector store not in the repo?"**
> It's large, it's fully regenerable from the notebook, and Chroma's SQLite file changes on every read — so committing it would mean a repo full of meaningless binary diffs. The `.gitignore` excludes it and the README explains how to rebuild it.

**"Walk me through what happens when I press Enter."**
> Streamlit posts to `/query`. FastAPI validates the question against the Pydantic schema — a bad one gets a 422 before my code runs. The question is embedded with the same MiniLM model that built the index. Chroma returns the 5 nearest chunks; I convert distance to similarity and drop anything under 0.25. If nothing survives I return the refusal without calling the LLM. Otherwise I build a numbered prompt with a system message restricting the model to that context, call Ollama, extract which `[n]` markers it cited, and return the answer with only those sources attached.

---

## Part 6 — When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `/health` shows `vector_store_loaded: false` | Notebook not run, or wrote elsewhere | Run the notebook to the end; check `backend/data/vector_store/` contains `chroma.sqlite3` |
| `502` from `/query` | Ollama not running | `ollama serve`, then `ollama list` to confirm the model is pulled |
| Answers are irrelevant nonsense | Embedding model mismatch | Confirm `EMBEDDING_MODEL` in `.env` matches `embedding_model` in `vector_store/config.json` |
| Everything gets refused | Floor too high, or empty index | Check `chunk_count` on `/health`. Try `MIN_SIMILARITY=0.15` |
| Off-topic questions get answered | Floor too low | Raise `MIN_SIMILARITY` toward `0.35` |
| First query takes 60 s | Ollama loading the model into RAM | Normal. Ask one throwaway question before your demo starts |
| CORS error in the browser console | Frontend origin not allowed | Add its URL to `CORS_ORIGINS` in `backend/.env` |
| Notebook crashes on `pypdf` | A corrupt download | Delete that PDF and re-run `download_corpus.py` |

---

## Part 7 — If you want to go further

Ordered by value per hour:

1. **Rerank the top 20 → 5** with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Biggest single quality jump available.
2. **Hybrid search** — combine vector similarity with BM25 keyword matching. Fixes exact-term lookups like "Part 107" that embeddings handle poorly.
3. **A real evaluation set** — 50 hand-labelled question/answer pairs, so `auto_correct` becomes a repeatable metric instead of a heuristic.
4. **Conversation memory** — rewrite follow-ups into standalone questions before retrieving.
5. **Streaming responses** — token-by-token output makes a 5-second answer *feel* instant.
