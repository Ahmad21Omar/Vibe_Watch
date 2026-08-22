# Deploying to Streamlit Community Cloud

Free, deploys straight from GitHub, ~10 minutes.

## Why this shape

Community Cloud runs **one process** and exposes **one port** — there is nowhere to put a
second service. The tempting shortcut is to let the UI import the pipeline directly when
deployed. That would mean the demo runs a *different architecture* than the repository
describes and the tests cover, and the first bug that only appears in the demo would be
impossible to reproduce locally.

So `EMBEDDED_API=true` starts the same API on loopback **inside** the UI process, and the
UI keeps talking HTTP to it — same client, same contract, same status codes:

```
   local / docker            Streamlit Community Cloud
   ┌──────┐   ┌─────┐        ┌───────────────────────────┐
   │  ui  │──►│ api │        │ ui ──HTTP──► api (127.0.0.1)│   one process
   └──────┘   └─────┘        └───────────────────────────┘
```

It is opt-in, never automatic: a UI that silently falls back to a hidden second copy of the
pipeline would hide exactly the outage an operator needs to see.

## 1. Prerequisites

- The index must live in **Qdrant Cloud** (a Community Cloud app has no database beside it
  and no disk that survives a restart). See `deploy/huggingface/DEPLOY.md` step 1–2 — the
  cluster setup is identical.
- The repository must be on GitHub, public or private.

## 2. Deploy

1. Sign in at https://share.streamlit.io with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `Ahmad21Omar/Vibe_Watch`
   - Branch: `main`
   - Main file path: `app.py`
   - Python version (under *Advanced settings*): **3.12**

## 3. Secrets — before you click Deploy

Open **Advanced settings → Secrets** and paste this, filling in your values:

```toml
GEMINI_API_KEY = "your-gemini-key"
QDRANT_URL = "https://xxxxx.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-key"
EMBEDDED_API = "true"
```

Two things worth knowing:

- **`EMBEDDED_API = "true"` is what makes it work at all.** Without it the UI looks for an
  API on `localhost:8000` that nobody started, and every request reports "cannot reach the
  API".
- The `:6333` on the Qdrant URL is not optional. Without a port the client tries 443.
- `TMDB_API_KEY` is deliberately **not** needed: TMDb is only used by the offline ingestion
  scripts, which the deployed app never runs.

Community Cloud hands these to the app as `st.secrets`, not as environment variables —
`app.py` bridges them into the environment before anything reads configuration.

## 4. Check it

First load takes a minute (installing dependencies), and the very first request also waits
~2 s for the embedded API to bind. Then ask for something.

If it errors, the app's **Manage app → logs** panel shows why. The usual cause is a missing
or misspelled secret; the message names which one.

## 5. Link it

- GitHub repo → ⚙️ next to **About** → **Website** → the app URL.
- Then tell me the URL and I will put it at the top of the README.

## Notes

- **Sleeping.** A Community Cloud app sleeps after a stretch of inactivity and wakes on the
  next visit. Worth mentioning yourself in an interview rather than being surprised by it.
- **Your Gemini quota is public.** Anyone using the demo spends it.
- **Updating.** Push to `main`; the app redeploys itself.
