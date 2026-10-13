"""A lightweight movie discovery and recommendation toolkit."""

from .engine import MovieNotFoundError, MovieRecommender, normalize_title

__all__ = ["MovieNotFoundError", "MovieRecommender", "normalize_title"]
