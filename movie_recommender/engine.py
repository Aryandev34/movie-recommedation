"""TF-IDF, cosine-similarity, fuzzy title matching, and hybrid ranking."""

from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data import normalize_movies

RecommendationMode = Literal["content", "hybrid"]


def normalize_title(title: str) -> str:
    """Normalize a title for exact and approximate matching."""
    ascii_title = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_title.casefold()))


class MovieNotFoundError(ValueError):
    """Raised when a movie title cannot be matched to the catalog."""

    def __init__(self, query: str, suggestions: list[str] | None = None) -> None:
        self.query = query
        self.suggestions = suggestions or []
        message = f"No movie title closely matches {query!r}."
        if self.suggestions:
            message += " Did you mean: " + ", ".join(self.suggestions) + "?"
        super().__init__(message)


class MovieRecommender:
    """Content and popularity-based recommendations for a movie catalog.

    Text fields are weighted and joined before TF-IDF vectorization. For a
    selected film, a single cosine-similarity vector is computed on demand;
    the implementation does not allocate a dense all-pairs similarity matrix.
    """

    def __init__(self, movies: pd.DataFrame) -> None:
        """Build a recommender from a raw or already-normalized DataFrame."""
        self._initialize(normalize_movies(movies))

    def _initialize(self, normalized: pd.DataFrame) -> None:
        self._movies = normalized.reset_index(drop=True).copy()
        self._normalized_titles = [normalize_title(title) for title in self._movies["title"]]
        self._title_indices: dict[str, list[int]] = {}
        for index, normalized_title in enumerate(self._normalized_titles):
            if normalized_title:
                self._title_indices.setdefault(normalized_title, []).append(index)
        self._unique_title_keys = list(self._title_indices)

        feature_texts = [self._build_feature_text(row) for row in self._movies.itertuples(index=False)]
        feature_texts = [text if text.strip() else f"catalogueitem{index}" for index, text in enumerate(feature_texts)]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=100_000,
            dtype=np.float32,
        )
        try:
            self._tfidf_matrix = self.vectorizer.fit_transform(feature_texts)
        except ValueError as error:
            # A tiny/custom catalog might contain only stop words. Keep it usable
            # rather than failing during index construction.
            if "empty vocabulary" not in str(error).casefold():
                raise
            self.vectorizer = TfidfVectorizer(
                stop_words=None,
                strip_accents="unicode",
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=100_000,
                dtype=np.float32,
            )
            self._tfidf_matrix = self.vectorizer.fit_transform(feature_texts)

        self._popularity_scores = self._calculate_popularity_scores()

    @classmethod
    def from_dataframes(
        cls,
        movies: pd.DataFrame,
        credits: pd.DataFrame | None = None,
    ) -> "MovieRecommender":
        """Create an engine from movies and (optionally) a separate credits frame."""
        recommender = cls.__new__(cls)
        recommender._initialize(normalize_movies(movies, credits))
        return recommender

    @classmethod
    def from_csv(
        cls,
        movies_path: str | Path,
        credits_path: str | Path | None = None,
    ) -> "MovieRecommender":
        """Load a CSV catalog, with optional separate TMDB credits CSV."""
        movies = pd.read_csv(movies_path, low_memory=False)
        credits = pd.read_csv(credits_path, low_memory=False) if credits_path is not None else None
        return cls.from_dataframes(movies, credits)

    @staticmethod
    def _build_feature_text(movie: object) -> str:
        """Boost distinctive structured metadata while retaining overview context."""
        weights = {
            "genres": 3,
            "keywords": 2,
            "director": 2,
            "cast": 2,
            "tagline": 2,
            "overview": 1,
        }
        parts: list[str] = []
        for field, weight in weights.items():
            value = getattr(movie, field, "")
            if value and str(value).strip():
                parts.extend([str(value)] * weight)
        return " ".join(parts)

    def _calculate_popularity_scores(self) -> np.ndarray:
        """Compute a smoothed rating prior combined with scaled catalog popularity."""
        ratings = self._movies["vote_average"].to_numpy(dtype=np.float64)
        votes = self._movies["vote_count"].to_numpy(dtype=np.float64)
        popularity = self._movies["popularity"].to_numpy(dtype=np.float64)

        known_ratings = ratings[ratings > 0]
        global_mean = float(known_ratings.mean()) if known_ratings.size else 0.0
        adjusted_ratings = np.where(ratings > 0, ratings, global_mean)
        nonzero_votes = votes[votes > 0]
        prior_strength = max(
            10.0,
            float(np.quantile(nonzero_votes, 0.60)) if nonzero_votes.size else 10.0,
        )
        weighted_rating = (
            votes / (votes + prior_strength) * adjusted_ratings
            + prior_strength / (votes + prior_strength) * global_mean
        )
        rating_score = np.clip(weighted_rating / 10.0, 0.0, 1.0)

        log_popularity = np.log1p(np.maximum(popularity, 0.0))
        max_log_popularity = float(log_popularity.max()) if log_popularity.size else 0.0
        scaled_popularity = log_popularity / max_log_popularity if max_log_popularity > 0 else log_popularity
        return (0.80 * rating_score + 0.20 * scaled_popularity).astype(np.float32)

    @property
    def movies(self) -> pd.DataFrame:
        """Return a copy of the normalized catalog."""
        return self._movies.copy()

    @property
    def number_of_movies(self) -> int:
        return len(self._movies)

    @property
    def feature_matrix_shape(self) -> tuple[int, int]:
        """Number of catalog rows and TF-IDF terms in the sparse matrix."""
        return self._tfidf_matrix.shape

    def suggest_titles(self, query: str, *, limit: int = 5, cutoff: float = 0.34) -> list[str]:
        """Return likely catalog titles for an approximate or misspelled query."""
        query_key = normalize_title(query)
        if not query_key or limit < 1:
            return []
        matches = difflib.get_close_matches(
            query_key,
            self._unique_title_keys,
            n=limit,
            cutoff=cutoff,
        )
        return [self._movies.iloc[self._title_indices[key][0]]["title"] for key in matches]

    def _resolve_index(self, query: str) -> tuple[int, bool]:
        query_key = normalize_title(query)
        if not query_key:
            raise MovieNotFoundError(query)

        exact_matches = self._title_indices.get(query_key)
        if exact_matches:
            # If a catalog has multiple releases with the same title, prefer the
            # better-known entry while still matching the user's exact title.
            index = max(
                exact_matches,
                key=lambda item: (
                    self._movies.iloc[item]["vote_count"],
                    self._movies.iloc[item]["vote_average"],
                ),
            )
            return index, True

        matches = difflib.get_close_matches(
            query_key,
            self._unique_title_keys,
            n=1,
            cutoff=0.43,
        )
        if matches:
            return self._title_indices[matches[0]][0], False
        raise MovieNotFoundError(query, self.suggest_titles(query))

    def find_movie(self, query: str) -> pd.Series:
        """Find an exact title or the closest fuzzy match in the catalog."""
        index, _ = self._resolve_index(query)
        return self._movies.iloc[index].copy()

    def similarity_vector(self, title: str) -> pd.Series:
        """Return the cosine-similarity score from one title to every catalog row."""
        index, _ = self._resolve_index(title)
        scores = linear_kernel(self._tfidf_matrix, self._tfidf_matrix[index]).ravel()
        return pd.Series(scores, index=self._movies["title"].to_list(), name="similarity")

    def recommend(
        self,
        title: str,
        *,
        top_n: int = 10,
        mode: RecommendationMode = "hybrid",
        popularity_weight: float = 0.15,
    ) -> pd.DataFrame:
        """Recommend movies similar to ``title``, ranked by a content/hybrid score.

        ``content`` sorts by cosine similarity alone. ``hybrid`` blends content
        similarity with the smoothed popularity prior; ``popularity_weight`` is
        the prior's share of the final score (0 to 1).
        """
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if mode not in ("content", "hybrid"):
            raise ValueError("mode must be either 'content' or 'hybrid'")
        if not 0.0 <= popularity_weight <= 1.0:
            raise ValueError("popularity_weight must be between 0 and 1")

        seed_index, _ = self._resolve_index(title)
        similarity = linear_kernel(self._tfidf_matrix, self._tfidf_matrix[seed_index]).ravel()
        seed_title_key = self._normalized_titles[seed_index]
        candidate_indices = np.asarray(
            [
                index
                for index, key in enumerate(self._normalized_titles)
                if key != seed_title_key
            ],
            dtype=np.int64,
        )
        if not candidate_indices.size:
            return pd.DataFrame(
                columns=[*self._movies.columns, "similarity", "popularity_score", "recommendation_score"]
            )

        results = self._movies.iloc[candidate_indices].reset_index(drop=True).copy()
        content_scores = np.clip(similarity[candidate_indices], 0.0, 1.0)
        prior_scores = self._popularity_scores[candidate_indices]
        weight = 0.0 if mode == "content" else float(popularity_weight)
        results["similarity"] = content_scores
        results["popularity_score"] = prior_scores
        results["recommendation_score"] = (1.0 - weight) * content_scores + weight * prior_scores
        results = results.sort_values(
            ["recommendation_score", "similarity", "popularity_score", "title"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        return results.head(top_n).reset_index(drop=True)

    def popular_movies(self, *, top_n: int = 10) -> pd.DataFrame:
        """Return catalog-wide popular picks ordered by the smoothed prior."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        results = self._movies.copy()
        results["similarity"] = np.nan
        results["popularity_score"] = self._popularity_scores
        results["recommendation_score"] = self._popularity_scores
        results = results.sort_values(
            ["popularity_score", "vote_count", "title"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        return results.head(top_n).reset_index(drop=True)
