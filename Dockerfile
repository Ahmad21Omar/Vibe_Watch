# ONE image, two roles: the API and the Streamlit UI run from it with different commands
# (see docker-compose.yml). Building it twice would let the two drift onto different
# versions of the same code -- the exact bug a service boundary is supposed to prevent.
# Qdrant runs as its own official image.
#
# Built so `docker compose up` is the only command a stranger needs: no Python install,
# no virtualenv, no version mismatch. The offline ingestion scripts stay OUT of this
# image on purpose (they are a one-off developer task, not part of the running service).

FROM python:3.12-slim

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the code, so
# rebuilding after an edit reuses the cached pip install instead of redoing it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vibewatch/ ./vibewatch/
COPY app.py .

EXPOSE 8000 8501

# Default role is the API; compose overrides `command` for the UI container.
# 0.0.0.0 rather than the default localhost: inside a container, "localhost" is the
# container itself, so binding there would make the service unreachable from the host.
CMD ["uvicorn", "vibewatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
