"""Dataset loading and normalization for common movie-metadata CSV formats."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "original_title", "movie_title", "name"),
    "genres": ("genres", "genre"),
    "keywords": ("keywords", "keyword", "tags"),
    "tagline": ("tagline", "slogan"),
    "overview": ("overview", "plot", "description", "summary"),
    "cast": ("cast", "actors", "actor", "starring"),
    "director": ("director", "filmmaker"),
    "crew": ("crew",),
}


def _find_column(frame: pd.DataFrame, choices: tuple[str, ...]) -> str | None:
    """Return the first matching column, ignoring capitalization and spaces."""
    available = {str(column).strip().casefold(): str(column) for column in frame.columns}
    for choice in choices:
        match = available.get(choice.casefold())
        if match is not None:
            return match
    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "nan", "none", "null", "<na>"}
    try:
        missing = pd.isna(value)
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _parse_structured(value: Any) -> Any:
    """Parse JSON/Python-list cells from TMDB exports without rejecting plain text."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return value


def _metadata_text(value: Any, *, limit: int | None = None) -> str:
    """Convert plain text or a TMDB list of ``{"name": ...}`` records to text."""
    if _is_missing(value):
        return ""

    parsed = _parse_structured(value)
    if isinstance(parsed, dict):
        parsed = parsed.get("name") or parsed.get("title") or parsed.get("original_name") or ""

    if isinstance(parsed, (list, tuple, set)):
        values: list[str] = []
        for item in parsed:
            if isinstance(item, dict):
                item = item.get("name") or item.get("title") or item.get("original_name")
            if _is_missing(item):
                continue
            text = str(item).strip()
            if text and text not in values:
                values.append(text)
            if limit is not None and len(values) >= limit:
                break
        return " ".join(values)

    text = str(parsed).strip()
    if text.casefold() in {"nan", "none", "null", "<na>"}:
        return ""
    return " ".join(text.split())


def _director_text(director: Any, crew: Any) -> str:
    """Prefer a director column, otherwise extract the Director from TMDB crew."""
    if not _is_missing(director):
        parsed = _parse_structured(director)
        if isinstance(parsed, (list, tuple)) and any(
            isinstance(person, dict) and "job" in person for person in parsed
        ):
            crew = parsed
        else:
            direct_text = _metadata_text(director, limit=1)
            if direct_text:
                return direct_text

    parsed_crew = _parse_structured(crew) if not _is_missing(crew) else []
    if isinstance(parsed_crew, dict):
        parsed_crew = [parsed_crew]
    if isinstance(parsed_crew, (list, tuple)):
        directors = [
            person.get("name")
            for person in parsed_crew
            if isinstance(person, dict)
            and str(person.get("job", "")).strip().casefold() == "director"
            and not _is_missing(person.get("name"))
        ]
        if directors:
            return _metadata_text(directors, limit=1)
        # A crew list without job fields is not a reliable director signal.
        return ""
    return _metadata_text(parsed_crew, limit=1)


def _normalized_key(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip().casefold()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _merge_credits(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    """Attach cast/crew/director fields from a separate TMDB credits CSV."""
    movies = movies.copy()
    left_key = _find_column(movies, ("id", "movie_id", "tmdb_id"))
    right_key = _find_column(credits, ("movie_id", "id", "tmdb_id"))
    if left_key is None or right_key is None:
        left_key = _find_column(movies, ("title", "original_title", "movie_title"))
        right_key = _find_column(credits, ("title", "original_title", "movie_title"))
    if left_key is None or right_key is None:
        raise ValueError(
            "Could not match the movies and credits files. Include an id/movie_id column "
            "in both files, or a title column in both."
        )

    credit_fields: dict[str, str] = {}
    for canonical, aliases in (
        ("cast", TEXT_ALIASES["cast"]),
        ("crew", TEXT_ALIASES["crew"]),
        ("director", TEXT_ALIASES["director"]),
    ):
        column = _find_column(credits, aliases)
        if column is not None:
            credit_fields[canonical] = column

    if not credit_fields:
        return movies

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in credits.iterrows():
        key = _normalized_key(row[right_key])
        if key and key not in lookup:
            lookup[key] = {name: row[column] for name, column in credit_fields.items()}

    movie_keys = movies[left_key].map(_normalized_key)
    for canonical in credit_fields:
        existing = _find_column(movies, TEXT_ALIASES[canonical])
        incoming = movie_keys.map(lambda key: lookup.get(key, {}).get(canonical))
        if existing is None:
            movies[canonical] = incoming
        else:
            missing = movies[existing].map(_is_missing)
            movies.loc[missing, existing] = incoming.loc[missing]
    return movies


def normalize_movies(
    movies: pd.DataFrame,
    credits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize raw movie metadata into the columns used by the recommender.

    Supports the two-file TMDB 5000 export (movies + credits), a pre-merged CSV,
    and simpler CSVs with plain-text fields. The returned frame is safe for use
    directly by :class:`movie_recommender.engine.MovieRecommender`.
    """
    if not isinstance(movies, pd.DataFrame):
        raise TypeError("movies must be a pandas DataFrame")
    if movies.empty:
        raise ValueError("The movies dataset is empty.")

    source = movies.copy()
    if credits is not None:
        if not isinstance(credits, pd.DataFrame):
            raise TypeError("credits must be a pandas DataFrame")
        source = _merge_credits(source, credits)
    source = source.reset_index(drop=True)

    title_column = _find_column(source, TEXT_ALIASES["title"])
    if title_column is None:
        raise ValueError("The movies CSV needs a title column (for example, 'title').")

    normalized = pd.DataFrame(index=source.index)
    id_column = _find_column(source, ("id", "movie_id", "tmdb_id"))
    if id_column is None:
        normalized["movie_id"] = [f"movie-{index + 1:05d}" for index in source.index]
    else:
        ids = source[id_column].map(_metadata_text)
        normalized["movie_id"] = [
            movie_id if movie_id else f"movie-{index + 1:05d}"
            for index, movie_id in enumerate(ids)
        ]

    normalized["title"] = source[title_column].map(_metadata_text)
    for field in ("genres", "keywords", "tagline", "overview", "cast"):
        column = _find_column(source, TEXT_ALIASES[field])
        if column is None:
            normalized[field] = ""
        else:
            limit = 5 if field == "cast" else None
            normalized[field] = source[column].map(lambda value: _metadata_text(value, limit=limit))

    director_column = _find_column(source, TEXT_ALIASES["director"])
    crew_column = _find_column(source, TEXT_ALIASES["crew"])
    director_values = source[director_column] if director_column else [None] * len(source)
    crew_values = source[crew_column] if crew_column else [None] * len(source)
    normalized["director"] = [
        _director_text(director, crew)
        for director, crew in zip(director_values, crew_values)
    ]

    rating_column = _find_column(source, ("vote_average", "rating", "average_rating", "imdb_rating"))
    votes_column = _find_column(source, ("vote_count", "votes", "rating_count"))
    popularity_column = _find_column(source, ("popularity", "view_count"))

    def numeric_values(column: str | None, *, default: float) -> np.ndarray:
        if column is None:
            return np.full(len(source), default, dtype=np.float64)
        values = pd.to_numeric(source[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(default)
        return values.to_numpy(dtype=np.float64)

    normalized["vote_average"] = np.clip(numeric_values(rating_column, default=0.0), 0.0, 10.0)
    normalized["vote_count"] = np.maximum(numeric_values(votes_column, default=0.0), 0.0)
    normalized["popularity"] = np.maximum(numeric_values(popularity_column, default=0.0), 0.0)

    release_column = _find_column(source, ("release_date", "year", "release_year"))
    if release_column is None:
        normalized["release_date"] = ""
    else:
        normalized["release_date"] = source[release_column].map(_metadata_text)

    normalized = normalized.loc[normalized["title"].str.strip().ne("")].reset_index(drop=True)
    if normalized.empty:
        raise ValueError("No movies with a usable title were found in the dataset.")
    return normalized


def load_movie_data(
    movies_path: str | Path,
    credits_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read one or two CSV files and return normalized movie metadata."""
    movies = pd.read_csv(movies_path, low_memory=False)
    credits = pd.read_csv(credits_path, low_memory=False) if credits_path is not None else None
    return normalize_movies(movies, credits)
