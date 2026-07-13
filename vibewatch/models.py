"""The data model of our domain: one movie or TV show.

Why normalize into our own model instead of passing raw TMDb JSON around?
- TMDb uses different field names for movies and TV ("title" vs "name",
  "release_date" vs "first_air_date"). We unify them once, here.
- The rest of the pipeline (embedding, indexing, retrieval) then works with
  ONE stable shape and does not care where the data came from.
- If we ever swap the data source, only this boundary changes.
"""

from typing import Literal

from pydantic import BaseModel


class Title(BaseModel):
    """A single movie or TV show, normalized from the TMDb API."""

    tmdb_id: int
    media_type: Literal["movie", "tv"]

    title: str
    overview: str

    # The title's ORIGINAL language (e.g. "ja" for anime). Not the app language --
    # we keep it as a metadata field so we can filter on it later.
    original_language: str

    genres: list[str]
    release_year: int | None

    popularity: float
    vote_average: float
    vote_count: int
    poster_path: str | None

    def embedding_text(self) -> str:
        """The text we turn into a vector.

        We do not embed the overview alone: adding title, genres and type gives the
        vector more context to match a mood/theme query against ("Survival, dark").
        This is a deliberate, tunable choice -- what you embed decides what you can find.
        """
        kind = "Movie" if self.media_type == "movie" else "TV show"
        genres = ", ".join(self.genres) if self.genres else "unknown"
        return (
            f"{kind}: {self.title}\n"
            f"Genres: {genres}\n"
            f"Plot: {self.overview}"
        )
