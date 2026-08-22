"""Central configuration for Vibewatch.

Why a dedicated config class instead of os.getenv() scattered across the code?
- Type safety: pydantic validates at startup that all values exist and are correct.
- Single source of truth for all settings: easy to find and to explain.
- One clear, actionable error when a key is missing -- not a cryptic crash somewhere
  deep in the program.

WHERE THE FAIL-FAST HAPPENS -- a decision worth explaining.
The keys used to be REQUIRED fields, so a missing one raised at import time. That reads
like the strictest option, but it made the package unimportable without secrets: cloning
the repo and running `pytest` crashed, even though all 78 unit tests are pure and need no
key at all. Worse, the error was a pydantic ValidationError -- technically precise, and
useless to someone who just wants to run the tests.

So the keys are now optional fields, and `require()` enforces them at the moment they are
actually used, with a message that says what to do. Fail-fast is preserved where it
matters (nobody reaches a live API call with an empty key) and dropped where it only hurt.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingConfiguration(RuntimeError):
    """Raised when a setting is needed but was never provided."""


# How to obtain each key -- included in the error so the fix needs no README lookup.
_KEY_HELP = {
    "tmdb_api_key": "free at https://www.themoviedb.org/settings/api",
    "gemini_api_key": "free at https://aistudio.google.com/apikey",
}


class Settings(BaseSettings):
    # Automatically reads from the .env file in the project root.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Empty default = "not configured". Checked by require() at the point of use, so
    # importing the package (and running the pure tests) works without any secrets.
    tmdb_api_key: str = ""
    gemini_api_key: str = ""

    # Optional values with sensible defaults for local Docker.
    qdrant_url: str = "http://localhost:6333"
    # Only managed Qdrant (Qdrant Cloud) requires this. Empty means "no auth", which is
    # exactly right for the local container -- and wrong to demand, since the whole test
    # suite and the local setup would then need a credential that does not exist.
    qdrant_api_key: str = ""
    # Where the UI finds the API. Inside docker compose this is overridden to the service
    # name (http://api:8000), because "localhost" in a container is the container itself.
    api_url: str = "http://localhost:8000"

    def require(self, name: str) -> str:
        """Return the setting `name`, or explain exactly what is missing and how to fix it."""
        value = getattr(self, name)
        if not value:
            raise MissingConfiguration(
                f"{name.upper()} is not set. Copy .env.example to .env and add it "
                f"({_KEY_HELP.get(name, 'see .env.example')})."
            )
        return value


# A single, project-wide importable instance: `from vibewatch.config import settings`
settings = Settings()
