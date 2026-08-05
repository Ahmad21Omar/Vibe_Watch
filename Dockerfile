# Image for the Streamlit app. Qdrant runs as its own official image -- see
# docker-compose.yml, which wires the two together.
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

EXPOSE 8501

# 0.0.0.0 rather than the default localhost: inside a container, "localhost" is the
# container itself, so binding there would make the app unreachable from the host.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
