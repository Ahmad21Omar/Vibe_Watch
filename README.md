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
- [ ] **Step 3 — Embeddings & indexing:** vectorize descriptions, store in Qdrant
      *(code complete; full index blocked by the embedding free-tier quota — see below)*
- [ ] **Step 4 — Retrieval:** semantic search with metadata filters
- [ ] **Step 5 — Generation & orchestration:** LangGraph flow + LLM reasoning
- [ ] **Step 6 — Frontend & evaluation:** Streamlit UI + RAGAS + Docker deployment

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
│   ├── embeddings.py    # text -> vector via Gemini (batched, rate-limited)
│   └── vector_store.py  # Qdrant: collection, indexing, search
├── scripts/
│   ├── fetch_titles.py  # offline ingestion: TMDb -> data/titles.json
│   └── index_titles.py  # offline indexing: embed -> Qdrant
├── data/                # locally cached TMDb data (git-ignored)
├── .env.example         # template for API keys
├── requirements.txt     # Python dependencies (grouped by step)
├── docker-compose.yml   # Qdrant vector DB
└── README.md
```

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

**Open constraint:** our catalogue has 912 titles, so a single full re-index consumes
almost the entire daily quota — which makes iterating on `embedding_text()` impractical.
Being evaluated: moving embeddings to a local ONNX model (`fastembed`, no PyTorch,
~300 MB, no quota) and keeping Gemini for the generation step, where one API call per
user query is cheap. Embeddings are the bulk operation; generation is not.
