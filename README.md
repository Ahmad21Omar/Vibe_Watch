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
| Embeddings | Gemini `text-embedding-004` | Multilingual, free tier, no local storage cost |
| Vector DB | Qdrant (Docker) | Fast similarity search + metadata filters |
| Orchestration | LangGraph | Clear, traceable RAG flow modeled as a graph |
| Generation | Gemini | Grounded recommendation in natural language |
| Evaluation | RAGAS | Measurable retrieval quality |
| Frontend | Streamlit | Fast UI to try things out |
| Deployment | Docker | Reproducible, runs anywhere |

---

## 🗺️ Roadmap

- [x] **Step 1 — Setup & foundation:** structure, venv, config, Qdrant, README
- [ ] **Step 2 — Data pipeline (TMDb):** fetch and normalize movies & TV shows
- [ ] **Step 3 — Embeddings & indexing:** vectorize descriptions, store in Qdrant
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
│   ├── __init__.py
│   └── config.py        # central, type-safe configuration
├── data/                # locally cached TMDb data (git-ignored)
├── .env.example         # template for API keys
├── requirements.txt     # Python dependencies (grouped by step)
├── docker-compose.yml   # Qdrant vector DB
└── README.md
```
