"""Central configuration for Vibewatch.

Why a dedicated config class instead of os.getenv() scattered across the code?
- Type safety: pydantic validates at startup that all values exist and are correct.
- Fail-fast: if a key is missing you get a clear error immediately,
  not a cryptic crash somewhere deep in the program.
- Single source of truth for all settings: easy to find and to explain.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Automatically reads from the .env file in the project root.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Required values (no default) -> the app won't start if they are missing.
    tmdb_api_key: str
    gemini_api_key: str

    # Optional value with a sensible default for local Docker.
    qdrant_url: str = "http://localhost:6333"


# A single, project-wide importable instance: `from vibewatch.config import settings`
settings = Settings()
