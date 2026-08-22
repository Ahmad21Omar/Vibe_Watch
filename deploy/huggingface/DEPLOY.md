# Deploying to a Hugging Face Space

Step by step. Roughly 30 minutes, most of it waiting for uploads.

The Space runs the API and the UI in **one container** (a Space has one public port), and
talks to a **Qdrant Cloud** cluster for the vectors — a Space has no database beside it,
and its disk does not survive a restart.

```
   HF Space (one container)              Qdrant Cloud            Google
   ┌───────────────────────────┐
   │ streamlit :7860 (public)  │
   │      ↓ 127.0.0.1:8000     │ ──────► vector search    ──────► embeddings + LLM
   │ uvicorn (internal only)   │
   └───────────────────────────┘
```

---

## 1. Create the Qdrant Cloud cluster (~10 min)

1. Sign up at https://cloud.qdrant.io (the free tier gives 1 GB — this index needs ~11 MB).
2. Create a cluster; any region will do, pick the one nearest to you.
3. Copy the **cluster URL** (looks like `https://xyz.eu-central.aws.cloud.qdrant.io:6333`)
   and create an **API key**.

## 2. Upload the index (~5 min, no LLM quota spent)

Vectors already exist in `data/embedding_cache.json`, so this re-uses them and calls no
API. Point the indexer at the cloud cluster for one run:

```bash
# PowerShell
$env:QDRANT_URL     = "https://xyz.eu-central.aws.cloud.qdrant.io:6333"
$env:QDRANT_API_KEY = "your-qdrant-key"
python -m scripts.index_titles
```

Check it landed:

```bash
python -c "from vibewatch.vector_store import get_client, COLLECTION_NAME; print(get_client().count(COLLECTION_NAME))"
```

Then **close that shell** so the variables do not linger and point your local runs at the
cloud by accident.

## 3. Create the Space (~5 min)

1. https://huggingface.co/new-space → name `vibewatch`, SDK **Docker**, template **Blank**,
   visibility **Public**.
2. Clone it and copy these files in:

```bash
git clone https://huggingface.co/spaces/<your-user>/vibewatch
cd vibewatch

cp -r ../Vibewatch/vibewatch .
cp ../Vibewatch/app.py ../Vibewatch/requirements.txt .
cp ../Vibewatch/deploy/huggingface/Dockerfile .
cp ../Vibewatch/deploy/huggingface/start.sh .
cp ../Vibewatch/deploy/huggingface/README.md .        # carries the Space config header

git add -A && git commit -m "Deploy Vibewatch" && git push
```

`README.md` matters: its YAML header is how a Space learns it is a Docker app on port 7860.

## 4. Set the secrets (~2 min)

In the Space: **Settings → Variables and secrets**, add three **secrets**:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | your Gemini key |
| `QDRANT_URL` | the cluster URL from step 1 |
| `QDRANT_API_KEY` | the Qdrant key from step 1 |

Secrets, not variables — variables are visible to anyone who opens the Space.

`TMDB_API_KEY` is deliberately **not** needed: TMDb is only used by the offline ingestion
scripts, which are not part of this image.

## 5. Check it

The Space rebuilds automatically. When it says *Running*:

- open the Space and ask for something — *"a korean thriller with a plot twist"*;
- if it errors, the **Logs** tab shows why. The usual cause is a missing or misspelled
  secret; the message names which one.

## 6. Link it from the repo

On GitHub: repo → ⚙️ next to **About** → **Website** → the Space URL. Then add it to the
top of the README so nobody has to hunt for it.

---

## Notes

- **Cold starts.** A free Space sleeps after inactivity and takes ~30 s to wake. Fine for
  a portfolio link; worth saying out loud in an interview rather than being surprised by it.
- **Your Gemini quota is public.** Anyone using the Space spends it. That is the trade for
  a link people can click without asking you for a key.
- **Updating.** Re-copy the changed files and push again; the Space rebuilds itself.
