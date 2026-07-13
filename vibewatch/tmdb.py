"""A thin client for the TMDb REST API.

Kept deliberately small: it only knows how to talk to TMDb (HTTP, auth, paging).
Turning the raw responses into our own `Title` model happens in `to_title()` --
that separation keeps each piece easy to read and to test.
"""

import httpx

from vibewatch.config import settings
from vibewatch.models import Title

BASE_URL = "https://api.themoviedb.org/3"

# TMDb returns 20 results per page.
RESULTS_PER_PAGE = 20

# The app is English-only, so we always ask TMDb for English text.
LANGUAGE = "en-US"


class TMDbClient:
    def __init__(self) -> None:
        # One reusable HTTP connection instead of a new one per request (much faster).
        self._client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    def _get(self, path: str, **params) -> dict:
        params["api_key"] = settings.tmdb_api_key
        response = self._client.get(path, params=params)
        response.raise_for_status()  # fail loudly on 401 (bad key) / 404 / 429
        return response.json()

    def genre_map(self, media_type: str) -> dict[int, str]:
        """TMDb only sends genre IDs with each title -- we need id -> name."""
        data = self._get(f"/genre/{media_type}/list", language=LANGUAGE)
        return {genre["id"]: genre["name"] for genre in data["genres"]}

    def discover(self, media_type: str, page: int) -> list[dict]:
        """One page of the most popular movies / TV shows."""
        data = self._get(
            f"/discover/{media_type}",
            language=LANGUAGE,
            sort_by="popularity.desc",
            include_adult=False,
            page=page,
        )
        return data["results"]

    def close(self) -> None:
        self._client.close()


def to_title(raw: dict, media_type: str, genre_names: dict[int, str]) -> Title:
    """Map one raw TMDb result onto our unified `Title` model.

    This is where the movie/TV field differences are ironed out.
    """
    if media_type == "movie":
        title = raw.get("title", "")
        date = raw.get("release_date", "")
    else:
        title = raw.get("name", "")
        date = raw.get("first_air_date", "")

    # Dates come as "1999-10-15"; they can also be missing or empty.
    release_year = int(date[:4]) if date[:4].isdigit() else None

    return Title(
        tmdb_id=raw["id"],
        media_type=media_type,
        title=title,
        overview=raw.get("overview", ""),
        original_language=raw.get("original_language", ""),
        genres=[genre_names[gid] for gid in raw.get("genre_ids", []) if gid in genre_names],
        release_year=release_year,
        popularity=raw.get("popularity", 0.0),
        vote_average=raw.get("vote_average", 0.0),
        vote_count=raw.get("vote_count", 0),
        poster_path=raw.get("poster_path"),
    )
