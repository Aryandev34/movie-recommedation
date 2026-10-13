"""Command-line interface for the movie recommender."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import MovieNotFoundError, MovieRecommender


DEFAULT_DATA = Path(__file__).resolve().parent / "sample_movies.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movie-recommender",
        description="Find movie recommendations with TF-IDF, cosine similarity, and popularity ranking.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help=f"movies CSV path (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--credits",
        default=None,
        help="optional separate TMDB credits CSV containing cast and crew",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    recommend_parser = commands.add_parser("recommend", help="recommend movies similar to a title")
    recommend_parser.add_argument("--title", required=True, help="movie title to look up")
    recommend_parser.add_argument("--top", type=int, default=10, help="number of results (default: 10)")
    recommend_parser.add_argument(
        "--mode",
        choices=("content", "hybrid"),
        default="hybrid",
        help="rank by text similarity only or blend in popularity (default: hybrid)",
    )
    recommend_parser.add_argument(
        "--popularity-weight",
        type=float,
        default=0.15,
        help="popularity share of the hybrid score, from 0 to 1 (default: 0.15)",
    )

    popular_parser = commands.add_parser("popular", help="show the most popular catalog titles")
    popular_parser.add_argument("--top", type=int, default=10, help="number of results (default: 10)")
    return parser


def _print_results(results, *, show_similarity: bool) -> None:
    for rank, (_, movie) in enumerate(results.iterrows(), start=1):
        year = str(movie.get("release_date", ""))[:4]
        rating = float(movie.get("vote_average", 0.0))
        genres = str(movie.get("genres", "")).replace("|", ", ")
        details = [item for item in (year if year.isdigit() else "", genres) if item]
        detail_text = f" ({' · '.join(details)})" if details else ""
        score = float(movie.get("recommendation_score", 0.0)) * 100
        if show_similarity:
            similarity = float(movie.get("similarity", 0.0)) * 100
            print(
                f"{rank:>2}. {movie['title']}{detail_text}  "
                f"| rank {score:5.1f}% · content match {similarity:5.1f}% · rating {rating:.1f}/10"
            )
        else:
            print(
                f"{rank:>2}. {movie['title']}{detail_text}  "
                f"| popularity score {score:5.1f}% · rating {rating:.1f}/10"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        recommender = MovieRecommender.from_csv(args.data, args.credits)
        print(f"Loaded {recommender.number_of_movies:,} movies from {args.data}\n")
        if args.command == "recommend":
            match = recommender.find_movie(args.title)
            if match["title"].casefold() != args.title.strip().casefold():
                print(f"Closest title match: {match['title']}\n")
            results = recommender.recommend(
                match["title"],
                top_n=args.top,
                mode=args.mode,
                popularity_weight=args.popularity_weight,
            )
            print(f"Because you searched for {match['title']}:\n")
            _print_results(results, show_similarity=True)
        else:
            results = recommender.popular_movies(top_n=args.top)
            print("Popular picks:\n")
            _print_results(results, show_similarity=False)
    except MovieNotFoundError as error:
        print(error)
        return 2
    except (OSError, ValueError) as error:
        print(f"Could not build recommendations: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
