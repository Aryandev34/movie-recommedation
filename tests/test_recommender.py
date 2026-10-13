from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from movie_recommender import MovieNotFoundError, MovieRecommender


SAMPLE = Path(__file__).resolve().parents[1] / "movie_recommender" / "sample_movies.csv"


@pytest.fixture(scope="module")
def recommender() -> MovieRecommender:
    return MovieRecommender.from_csv(SAMPLE)


def test_sample_catalog_builds_sparse_tfidf_features(recommender: MovieRecommender) -> None:
    assert recommender.number_of_movies == 40
    rows, terms = recommender.feature_matrix_shape
    assert rows == 40
    assert terms > 100


def test_fuzzy_title_matching_handles_spelling_variations(recommender: MovieRecommender) -> None:
    assert recommender.find_movie("inception")["title"] == "Inception"
    assert recommender.find_movie("Incepton")["title"] == "Inception"
    assert recommender.suggest_titles("Incepton")[0] == "Inception"


def test_recommendations_exclude_seed_and_sort_by_rank_score(recommender: MovieRecommender) -> None:
    results = recommender.recommend("Inception", top_n=8, mode="hybrid", popularity_weight=0.15)

    assert len(results) == 8
    assert "Inception" not in results["title"].tolist()
    assert results["recommendation_score"].is_monotonic_decreasing
    assert results["similarity"].between(0, 1).all()
    assert results["popularity_score"].between(0, 1).all()


def test_content_mode_is_not_affected_by_popularity_weight(recommender: MovieRecommender) -> None:
    content_a = recommender.recommend("The Matrix", top_n=5, mode="content", popularity_weight=0.0)
    content_b = recommender.recommend("The Matrix", top_n=5, mode="content", popularity_weight=0.5)
    assert content_a["title"].tolist() == content_b["title"].tolist()
    assert content_a["recommendation_score"].tolist() == pytest.approx(content_a["similarity"].tolist())


def test_popular_list_is_sorted(recommender: MovieRecommender) -> None:
    results = recommender.popular_movies(top_n=6)
    assert len(results) == 6
    assert results["popularity_score"].is_monotonic_decreasing
    assert results["recommendation_score"].tolist() == pytest.approx(results["popularity_score"].tolist())


def test_unknown_title_has_actionable_suggestions(recommender: MovieRecommender) -> None:
    with pytest.raises(MovieNotFoundError) as error:
        recommender.recommend("zzzzzzzzzz")
    assert error.value.query == "zzzzzzzzzz"
    assert isinstance(error.value.suggestions, list)


def test_tmdb_movies_and_credits_json_are_normalized_and_joined() -> None:
    movies = pd.DataFrame(
        {
            "id": [101, 202],
            "title": ["Asteria", "Solar Drift"],
            "genres": ['[{"id": 1, "name": "Science Fiction"}]', '[{"name": "Adventure"}]'],
            "keywords": ['[{"name": "first contact"}]', '[{"name": "space mission"}]'],
            "tagline": ["A signal from beyond.", "A long way from home."],
            "overview": ["A linguist studies an alien transmission.", "A pilot navigates a distant star system."],
            "vote_average": [7.8, 7.2],
            "vote_count": [250, 80],
            "popularity": [12.0, 4.0],
        }
    )
    credits = pd.DataFrame(
        {
            "movie_id": [101, 202],
            "cast": [
                '[{"name": "Ava Ray"}, {"name": "Noah Vale"}]',
                '[{"name": "Mina Sol"}]',
            ],
            "crew": [
                '[{"job": "Producer", "name": "R. Lee"}, {"job": "Director", "name": "Mira Tan"}]',
                '[{"job": "Director", "name": "O. Park"}]',
            ],
        }
    )

    engine = MovieRecommender.from_dataframes(movies, credits)
    first = engine.find_movie("Asteria")
    assert first["movie_id"] == "101"
    assert first["genres"] == "Science Fiction"
    assert first["keywords"] == "first contact"
    assert first["cast"] == "Ava Ray Noah Vale"
    assert first["director"] == "Mira Tan"
    assert len(engine.recommend("Asteria", top_n=1)) == 1


def test_title_column_is_required() -> None:
    with pytest.raises(ValueError, match="title column"):
        MovieRecommender(pd.DataFrame({"genre": ["Drama"]}))
