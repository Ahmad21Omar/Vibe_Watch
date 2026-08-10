"""HTTP client for the Vibewatch API -- what the UI uses instead of calling the pipeline.

Why the UI does not simply import `recommend()` any more, now that a service exists:
if the frontend keeps its own private path into the pipeline, the API is decoration. The
UI is then not proof that the service works, and the two ways of asking can drift apart.
Making the UI an ordinary client means every request goes through the same contract, and
the UI becomes the API's most-used integration test.

There is deliberately NO fallback to the in-process pipeline. A silent fallback would hide
exactly the failure the operator needs to see -- the app would look healthy while the
service it depends on is down.

Errors are translated into one exception type carrying a message a UI can show. A
Streamlit page has no use for an httpx traceback; it needs a sentence.
"""

import httpx

from vibewatch.config import settings

# Generous: a request runs an LLM extraction, an embedding, a vector search and a
# generation call. Anything less and a normal slow-but-fine request looks like a failure.
TIMEOUT_SECONDS = 120.0


class ApiError(RuntimeError):
    """The API could not answer. The message is meant to be shown to a person."""


def _post(path: str, payload: dict) -> dict:
    try:
        response = httpx.post(
            f"{settings.api_url.rstrip('/')}{path}", json=payload, timeout=TIMEOUT_SECONDS
        )
    except httpx.RequestError as error:
        raise ApiError(f"Cannot reach the API at {settings.api_url}: {error}") from error

    if response.is_error:
        # FastAPI puts the reason in `detail`; fall back to the raw body if it is missing
        # (a proxy error, for instance, will not follow that shape).
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"API error {response.status_code}: {detail}")

    return response.json()


def _get(path: str) -> dict:
    try:
        response = httpx.get(f"{settings.api_url.rstrip('/')}{path}", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ApiError(f"Cannot reach the API at {settings.api_url}: {error}") from error
    return response.json()


def recommend(
    query: str, *, limit: int = 5, history: list[str] | None = None, **filters
) -> dict:
    """Ask the API for a recommendation. Same shape as the in-process pipeline returns.

    `history` is this client's own conversation. Sending it -- rather than relying on the
    server to remember -- is what keeps the service stateless and the client in control of
    what it is asking about.
    """
    payload = {"query": query, "limit": limit, **filters}
    if history:
        payload["history"] = history
    return _post("/recommend", payload)


def genres() -> list[str]:
    """The catalogue's genres, for building the filter UI."""
    return _get("/genres")["genres"]
