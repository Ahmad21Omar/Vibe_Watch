#!/usr/bin/env bash
# Start the API and the UI in one container (the shape a Hugging Face Space requires).
#
# `set -e` plus the wait/kill pair below is the whole supervision story: if either process
# dies the container exits, so the platform restarts it. A half-dead container that still
# answers the health check is worse than a restart -- it serves a UI whose backend is gone.
set -e

uvicorn vibewatch.api:app --host 127.0.0.1 --port 8000 &
api_pid=$!

# Streamlit is the public face, on the port Spaces routes to.
streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --browser.gatherUsageStats=false &
ui_pid=$!

# Wait for whichever exits first, then take the other one down with it.
wait -n "$api_pid" "$ui_pid"
kill "$api_pid" "$ui_pid" 2>/dev/null || true
exit 1
