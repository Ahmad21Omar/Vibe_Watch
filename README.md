# 🎬 Vibewatch

**A semantic recommendation system for movies & TV shows.**
Instead of searching by keywords, the user describes a *mood or theme* in natural
language (e.g. *"Survival, people fighting to stay alive"*), and Vibewatch finds the
most relevant titles via **vector similarity** and lets an LLM write a **grounded,
reasoned recommendation**.

This is a classic **RAG flow** (Retrieval-Augmented Generation):
first **retrieval** (fetch matching movies from the vector DB), then **generation**
(the LLM reasons — but *only* based on the retrieved movies, so it doesn't hallucinate).

---

## 🏛️ Architecture

```
                          ┌─────────────────────────────┐
   User query             │   1. Data pipeline (TMDb)   │   (offline, one-off)
   "Survival, dark"       │   fetch movies & TV shows   │
          │               └──────────────┬──────────────┘
          │                              │
          │                              ▼
          │               ┌─────────────────────────────┐
          │               │   2. Embeddings (Gemini)     │
          │               │   text  ->  vector           │
          │               └──────────────┬──────────────┘
          │                              ▼
          │               ┌─────────────────────────────┐
          │               │   3. Qdrant (vector DB)      │
          │               │   vectors + metadata         │
          │               └──────────────┬──────────────┘
          ▼                              │
   ┌──────────────┐   query vector       │
   │  Embed the   │──────────────────────┤
   │  user query  │                      ▼
   └──────────────┘        ┌─────────────────────────────┐
                           │   4. Retrieval (Top-K)       │
                           │   + metadata filters         │
                           └──────────────┬──────────────┘
                                          ▼
                           ┌─────────────────────────────┐
                           │   5. Generation (Gemini LLM) │
                           │   grounded recommendation    │
                           └──────────────┬──────────────┘
                                          ▼
                               Recommendation to the user
                               (Streamlit frontend)
```

The query flow (4 → 5) is orchestrated with **LangGraph** and evaluated with **RAGAS**.

---

## 🧰 Tech stack & rationale

| Component | Technology | Why |
|-----------|-------------|-----|
| Language / backend | Python, FastAPI | Standard for AI engineering, fast APIs |
| Data source | TMDb API | Movies & TV shows, multilingual descriptions, free |
| Embeddings | Gemini `gemini-embedding-001` | Strong semantic quality, free tier, no local storage cost |
| Vector DB | Qdrant (Docker) | Fast similarity search + metadata filters |
| Orchestration | LangGraph | Clear, traceable RAG flow modeled as a graph |
| Generation | Gemini | Grounded recommendation in natural language |
| Evaluation | RAGAS | Measurable retrieval quality |
| Frontend | Streamlit | Fast UI to try things out |
| Deployment | Docker | Reproducible, runs anywhere |

---

## 🗺️ Roadmap

- [x] **Step 1 — Setup & foundation:** structure, venv, config, Qdrant, README
- [x] **Step 2 — Data pipeline (TMDb):** fetch and normalize movies & TV shows
- [x] **Step 3 — Embeddings & indexing:** vectorize descriptions, store in Qdrant
- [x] **Step 4 — Retrieval:** semantic search with metadata filters
- [x] **Step 5 — Generation & orchestration:** LangGraph flow + LLM reasoning
- [ ] **Step 6 — Frontend & evaluation:** Streamlit UI + RAGAS + Docker deployment

The pipeline is end-to-end usable today:

```bash
python -m scripts.recommend "survival, dark, fighting to stay alive" --type movie
```

---

## 🚀 Setup (local)

**Prerequisites:** Python 3.12+, Docker Desktop.

```bash
# 1. Enter the repo folder and create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows (PowerShell)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configuration: copy .env.example and fill in your keys
copy .env.example .env          # Windows
#   -> set TMDB_API_KEY and GEMINI_API_KEY in .env

# 4. Start the vector DB (Qdrant)
docker compose up -d
#   -> web dashboard: http://localhost:6333/dashboard
```

---

## 📁 Project structure

```
Vibewatch/
├── vibewatch/           # Python package with the actual code
│   ├── config.py        # central, type-safe configuration
│   ├── models.py        # Title: our unified movie/TV data model
│   ├── tmdb.py          # thin TMDb API client
│   ├── embeddings.py    # text -> vector via Gemini (batched, rate-limited, resumable)
│   ├── embedding_cache.py  # on-disk cache so an interrupted run resumes
│   ├── vector_store.py  # Qdrant: collection, indexing, filtered search
│   ├── retrieval.py     # query -> ranked, grounded titles (the retrieval seam)
│   ├── generation.py    # prompt building + grounded LLM recommendation
│   └── graph.py         # LangGraph flow: retrieve -> generate (+ retry)
├── scripts/
│   ├── fetch_titles.py  # offline ingestion: TMDb -> data/titles.json
│   ├── index_titles.py  # offline indexing: embed -> Qdrant
│   └── recommend.py     # online: ask for a recommendation from the CLI
├── tests/               # fast unit tests + opt-in live integration tests
├── data/                # locally cached TMDb data (git-ignored)
├── .env.example         # template for API keys
├── requirements.txt     # Python dependencies (grouped by step)
├── docker-compose.yml   # Qdrant vector DB
└── README.md
```

---

## 🧪 Tests

```bash
pytest                  # fast, pure unit tests -- no API, no Docker, no quota
pytest -m integration   # end-to-end against live Qdrant + Gemini (opt-in)
```

The default suite covers the places where a silent bug would quietly corrupt everything
downstream: the `Title` model and its `embedding_text()`, the TMDb movie/TV field mapping,
the `point_id()` idempotency guarantee that keeps re-indexing from creating duplicates,
the metadata filters, the prompt that grounds the LLM, and the graph wiring.

Two patterns make that possible without mocking the code under test. **Dependencies are
injectable** — `retrieve()` takes its client and embedder, `build_graph()` takes its
retrieval and generation functions — so tests compile the *real* graph and run the *real*
search adapter with fakes only at the boundary. And **live services are opt-in**: the
integration tests carry a marker that a bare `pytest` deselects, and skip themselves
cleanly when Qdrant or the API key is absent, so a fresh checkout stays green.

The integration suite asserts *structure*, not specific titles — which film ranks first
shifts as the TMDb catalogue changes, so pinning one would only produce a flaky test. What
it does pin is the property RAG exists for: the answer must name a title that was actually
retrieved.

---

## 📥 Data pipeline

Ingestion runs **offline**, separated from the live query path — the standard RAG
architecture (no API limits at query time, reproducible data to embed).

```bash
python -m scripts.fetch_titles     # ~900 titles -> data/titles.json
```

It fetches the most popular movies & TV shows from TMDb, unifies the differing
movie/TV field names into one `Title` model, drops titles without a plot, and
de-duplicates (popularity shifts during paging can return the same title twice).

**Design note:** we do not embed the plot alone. `Title.embedding_text()` builds
`type + title + genres + plot`, giving the vector more context for a mood/theme query
to match against. What you embed decides what you can find.

---

## 🧠 Embeddings & indexing

```bash
python -m scripts.index_titles     # embed titles -> Qdrant collection "titles"
```

Each title becomes a **3072-dimensional vector** plus a metadata payload, stored as one
Qdrant point.

Three decisions worth calling out:

- **Asymmetric embeddings.** Documents are embedded with `task_type=RETRIEVAL_DOCUMENT`,
  queries with `RETRIEVAL_QUERY`. A short mood query and a 40-word plot are different
  kinds of text; telling the model which role a text plays measurably improves retrieval.
- **Cosine distance.** It compares the *angle* between vectors and ignores their length,
  so a three-word query and a long plot are compared by meaning, not by text volume.
- **Idempotent indexing.** Point ids are `uuid5(media_type + tmdb_id)` — deterministic,
  so re-running the script updates points instead of duplicating them.

### Rate limits (and an open constraint)

The free tier enforces **two** quotas, counted **per text** rather than per API call:
100 embeddings per minute, and **1000 per day**. `embeddings.py` therefore throttles
proactively and, on a 429, honours the retry delay the server returns instead of guessing
a backoff.

**Resumable by design.** Computed vectors are checkpointed to disk after every batch
(`embedding_cache.py`), keyed by a hash of `(model, task_type, text)`. If a run is
interrupted, re-running skips everything already embedded and only calls the API for what
is new — so a crash costs at most one batch, not the whole run.

**Standing constraint:** the catalogue has 912 titles, so a single full re-index consumes
almost the entire daily quota. The index is built and cached, so this costs nothing at
query time — but it does make iterating on `embedding_text()` a once-a-day affair. The
escape hatch, if that becomes limiting: move embeddings to a local ONNX model
(`fastembed`, no PyTorch, ~300 MB, no quota) and keep Gemini for generation, where one API
call per user query is cheap. Embeddings are the bulk operation; generation is not.

---

## 🔎 Retrieval

Two different questions, deliberately answered by two different mechanisms:

| Question | Answered by | Character |
|---|---|---|
| "What *feels* like this?" | vector similarity | soft, graded, a ranking |
| "And only movies since 2015?" | metadata filter | hard, yes/no |

A vector cannot reliably express "movies only" — the embedding of a survival *movie* is
still very close to a survival *series*. Hard constraints belong in the filter, not in the
vector and not in the prompt.

```python
retrieve("dark, hopeless survival", media_type="movie", release_year_min=2015)
```

**The decision that matters here is pre-filtering.** Qdrant applies the filter *during*
the vector search, using the payload indexes created with the collection. The naive
alternative — search top-5, then discard what does not match — would return two results
where the catalogue holds two hundred matching films. That is the kind of bug that never
looks broken in production; it just quietly answers worse.

---

## ✍️ Generation & orchestration

```
                    hits found
    query --> [retrieve] ------> [generate] --> answer
                  ^   |
                  |   | nothing found, but filters were set
                  |   v
              [relax_filters]
```

The LLM is given the retrieved titles as numbered, factual blocks and told that they are
the only titles that exist. Grounding is enforced on four levels: a static system
instruction holding the rules (separate from the volatile data, and cacheable), context
blocks built by the same `Title.as_context_block()` used everywhere else, a short-circuit
that never calls the model when retrieval came back empty (with no context, anything it
writes is invented), and a test asserting the answer names a retrieved title.

The model also acts as a **re-ranker**: it is told it may skip a higher-ranked candidate,
because similarity is not the same as suitability. In a real run, *Fight Club* ranked
second for a survival query and the model correctly passed it over.

**Why a graph for two steps?** Honestly: the linear part alone would not justify one. It
starts paying off at the conditional edge. Hard filters are unforgiving — "TV shows from
2024 tagged Western" easily matches nothing in a 900-title catalogue — so when a filtered
search comes back empty the flow retries once without the filters instead of giving up on
a mood it could have matched. The `relaxed` flag caps that at one retry and is surfaced to
the user, because silently ignoring what someone asked for is worse than showing nothing.
