# Image for the Hugging Face Space. Same application, different shape of host.
#
# WHY THIS IS NOT THE ROOT Dockerfile.
# Locally, docker compose runs three containers: qdrant, api, ui. A Space is ONE container
# with ONE public port, and no database beside it. So this image:
#   - runs the API and the UI as two processes side by side (see start.sh),
#   - exposes only the UI, on 7860 (what Spaces routes to by default),
#   - keeps the API on 127.0.0.1:8000, reachable by the UI and by nobody else,
#   - expects Qdrant to live elsewhere (Qdrant Cloud), via QDRANT_URL + QDRANT_API_KEY.
#
# The API staying internal is not a compromise -- it is the correct shape here. Nothing
# outside needs it, and exposing an unauthenticated pipeline that spends LLM quota on
# every request would be an open invitation.

FROM python:3.12-slim

# Spaces run as a non-root user; writing into /app as root would leave the runtime unable
# to create Streamlit's config and cache directories.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HOME=/home/user

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user vibewatch/ ./vibewatch/
COPY --chown=user app.py .
COPY --chown=user start.sh .

# The UI talks to the API over loopback inside this container.
ENV API_URL=http://127.0.0.1:8000

EXPOSE 7860

CMD ["bash", "start.sh"]
