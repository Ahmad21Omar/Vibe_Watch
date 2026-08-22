"""Tests for configuration loading.

The property under test is one nobody notices until it bites a stranger: **the package
must import, and the pure test suite must run, on a fresh clone with no secrets at all.**
Every unit test here is pure -- requiring an API key just to import them would turn
"clone and run pytest" into a crash, which is the worst possible first impression.

The trade is that a missing key must then be caught at the point of USE, with a message
that says what to do about it. That is what `require()` is for, and what is pinned here.
"""

import pytest

from vibewatch.config import MissingConfiguration, Settings


def _blank_settings(**overrides) -> Settings:
    """Settings as a fresh clone sees them: no .env, nothing configured."""
    return Settings(_env_file=None, tmdb_api_key="", gemini_api_key="", **overrides)


def test_settings_load_without_any_keys():
    # The whole point: no secrets, no crash. If this ever raises again, `pytest` on a
    # fresh checkout is broken for everyone who does not have API keys.
    settings = _blank_settings()

    assert settings.gemini_api_key == ""
    assert settings.qdrant_url  # the local-Docker default still applies


def test_require_returns_a_configured_value():
    settings = _blank_settings()
    settings.gemini_api_key = "abc123"

    assert settings.require("gemini_api_key") == "abc123"


def test_require_explains_which_variable_is_missing_and_how_to_fix_it():
    # An error a beginner can act on beats a technically precise one they cannot. The
    # message must name the variable, the file to create, and where to get the key.
    with pytest.raises(MissingConfiguration) as error:
        _blank_settings().require("gemini_api_key")

    message = str(error.value)
    assert "GEMINI_API_KEY" in message
    assert ".env" in message
    assert "aistudio.google.com" in message


def test_require_covers_the_tmdb_key_too():
    with pytest.raises(MissingConfiguration) as error:
        _blank_settings().require("tmdb_api_key")

    assert "TMDB_API_KEY" in str(error.value)
    assert "themoviedb.org" in str(error.value)


# --- managed Qdrant (Qdrant Cloud) --------------------------------------------------------


def test_qdrant_needs_no_api_key_by_default():
    # The local container has no auth. Requiring a key would break every local run and
    # the entire test suite for the sake of a deployment target.
    assert _blank_settings().qdrant_api_key == ""


def test_a_configured_qdrant_key_is_passed_to_the_client(monkeypatch):
    # Qdrant Cloud rejects unauthenticated requests, so the key has to reach the client --
    # and an empty one must NOT be sent, since some local setups reject an empty
    # Authorization header outright.
    import vibewatch.vector_store as vector_store

    seen = {}
    monkeypatch.setattr(vector_store, "QdrantClient", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(vector_store.settings, "qdrant_api_key", "cloud-key")
    monkeypatch.setattr(vector_store.settings, "qdrant_url", "https://cluster:6333")

    vector_store.get_client()

    assert seen == {"url": "https://cluster:6333", "api_key": "cloud-key"}


def test_no_key_means_no_auth_header(monkeypatch):
    import vibewatch.vector_store as vector_store

    seen = {}
    monkeypatch.setattr(vector_store, "QdrantClient", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(vector_store.settings, "qdrant_api_key", "")

    vector_store.get_client()

    assert seen["api_key"] is None
