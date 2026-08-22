"""Run the API inside the UI's own process, for hosts that give you exactly one.

Locally and in Docker the API and the UI are separate services and talk over HTTP. Some
free hosting (Streamlit Community Cloud, for one) runs a single Python process and exposes
a single port -- there is nowhere to put a second service.

The tempting shortcut is to let the UI import `recommend()` directly when it is deployed
there. That would mean the demo runs a DIFFERENT architecture than the one the repository
describes and the tests cover, and the first bug that only appears in the demo would be
impossible to reproduce locally.

So instead the same API is started on loopback inside this process, and the UI keeps
talking HTTP to it. Same client, same contract, same status codes -- one process. The
only thing that changes is where the server lives.

This is opt-in via `EMBEDDED_API=true`, never automatic. An automatic fallback would mean
that a UI which cannot reach its API silently starts serving from a second, hidden copy of
the pipeline -- hiding exactly the outage an operator needs to see.
"""

import threading
import time

import httpx
import uvicorn

# Loopback only. This server exists for the process it lives in; binding it to 0.0.0.0
# would publish an unauthenticated pipeline that spends LLM quota per request.
HOST = "127.0.0.1"
PORT = 8000

STARTUP_TIMEOUT_SECONDS = 30.0

# NOTE: call this ONCE per process. uvicorn cannot bind a port twice, and Streamlit re-runs
# its script on every interaction -- so the caller is expected to memoise it
# (`@st.cache_resource` in app.py). Guarding in here instead would hide the mistake from a
# caller who genuinely wanted a second server on another port.


def serve_api_in_background(host: str = HOST, port: int = PORT) -> str:
    """Start the API on a daemon thread and return its base URL once it answers.

    Blocking until it responds matters: Streamlit renders immediately after this, and a
    first request against a socket that is not listening yet would surface to the user as
    "cannot reach the API" on a perfectly healthy app.
    """
    from vibewatch.api import app

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
    # Daemon: the process must be able to exit even though this loop never finishes.
    threading.Thread(target=server.run, daemon=True).start()

    base_url = f"http://{host}:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            # `/openapi.json` proves the SERVER is up. `/health` would also check Qdrant,
            # which can legitimately be down while the API itself is fine -- and then we
            # would wait out the timeout for no reason.
            httpx.get(f"{base_url}/openapi.json", timeout=2.0)
            return base_url
        except httpx.RequestError:
            time.sleep(0.2)

    raise RuntimeError(
        f"the embedded API did not start within {STARTUP_TIMEOUT_SECONDS:.0f}s"
    )
