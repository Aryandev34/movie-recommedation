# Reel Finder — Movie Recommendation System

A Python movie-discovery project that combines **content-based recommendations** with a **popularity prior**. It uses pandas and NumPy to normalize movie metadata, scikit-learn TF-IDF vectors to represent each film, cosine similarity to find related titles, and Python's `difflib` to handle approximate title searches. An interactive Streamlit app and a command-line interface are included.

> The repository ships with a compact 40-title sample catalog so the project works immediately. To recommend across 4,800+ films, upload or point the app/CLI at the full TMDB 5000 metadata files; the full dataset is intentionally not bundled.

## Features

- Builds movie fingerprints from **genres, keywords, tagline, cast, director, and overview**.
- Gives structured genres and keywords extra weight during TF-IDF vectorization; uses unigrams and bigrams.
- Ranks similar films with cosine similarity. A single sparse similarity vector is calculated per query instead of storing a dense all-pairs matrix.
- Supports a **content-only** ranking and a **hybrid** ranking that blends similarity with a smoothed rating/popularity score.
- Uses `difflib.get_close_matches` to resolve misspelled or approximate movie titles and offers likely matches when a search fails.
- Reads the common two-file TMDB layout (`tmdb_5000_movies.csv` + `tmdb_5000_credits.csv`) as well as merged or simpler CSVs.
- Includes an interactive Streamlit front end, CLI, and unit tests.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

The app starts with the included sample catalog. Use the sidebar to upload a movies CSV and, optionally, a separate credits CSV. For the TMDB 5000 export, upload `tmdb_5000_movies.csv` as the movies file and `tmdb_5000_credits.csv` as the credits file.

## Command line

Recommend titles similar to a movie:

```bash
python -m movie_recommender recommend --title "Incepton" --top 5
```

The misspelled title resolves to the closest sample-catalog match. Choose a pure content ranking or adjust the popularity blend:

```bash
python -m movie_recommender recommend --title "Inception" --mode content --top 10
python -m movie_recommender recommend --title "Inception" --mode hybrid --popularity-weight 0.20
```

Show the most popular titles in the catalog:

```bash
python -m movie_recommender popular --top 10
```

Use your own dataset (global options go before the subcommand):

```bash
python -m movie_recommender \
  --data data/tmdb_5000_movies.csv \
  --credits data/tmdb_5000_credits.csv \
  recommend --title "Inception" --top 10
```

The CSVs can be downloaded from the [TMDB Movie Metadata dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). Large or private datasets should stay local; the repository ignores the `data/` directory by default.

## Data format

At minimum, provide a `title` column. The engine recognizes common aliases such as `original_title`, `genre`, `keyword`, `plot`, `actors`, `rating`, and `votes`. TMDB JSON-style `genres`, `keywords`, `cast`, and `crew` cells are parsed automatically. When two files are supplied, movie `id` is joined to credits `movie_id` (other ID aliases or matching title columns are supported).

Useful optional columns:

| Field | Used for |
| --- | --- |
| `genres`, `keywords` | Core content features; weighted more heavily |
| `tagline`, `overview` | Story and theme features |
| `cast`, `director` or TMDB `crew` | People-based similarities |
| `vote_average`, `vote_count` | Smoothed audience rating prior |
| `popularity` | Additional catalog-popularity signal |
| `release_date` | Result display |

Missing optional fields are allowed. Rows with blank titles are skipped. Downloaded datasets are not committed to Git.

## How ranking works

1. Text metadata is cleaned into a combined feature document per movie. Genres are repeated three times; keywords, director, cast, and tagline twice; overview once. Repetition is a simple way to boost those signals without adding another dependency.
2. A scikit-learn `TfidfVectorizer` creates sparse vectors using English stop-word removal and one-/two-word features.
3. The cosine similarity of the selected movie's vector against every catalog vector provides the content score.
4. **Content mode** sorts by cosine similarity. **Hybrid mode** uses:

   `rank_score = (1 − popularity_weight) × content_similarity + popularity_weight × popularity_score`

   The default popularity weight is `0.15`. The popularity prior combines a Bayesian-style vote-adjusted rating with log-scaled catalog popularity. Both component scores are exposed in the results for interpretation.

These scores are ranking signals, not calibrated probabilities or guarantees of personal preference. The project does not use user histories or collaborative filtering.

## Python API

```python
from movie_recommender import MovieRecommender

engine = MovieRecommender.from_csv(
    "data/tmdb_5000_movies.csv",
    "data/tmdb_5000_credits.csv",
)

matches = engine.recommend(
    "Incepton",       # exact or fuzzy title lookup
    top_n=10,
    mode="hybrid",    # or "content"
    popularity_weight=0.15,
)
print(matches[["title", "similarity", "popularity_score", "recommendation_score"]])
```

Other useful methods are `find_movie(title)`, `suggest_titles(query)`, `similarity_vector(title)`, and `popular_movies(top_n=10)`. You can also construct an engine from pandas DataFrames with `MovieRecommender.from_dataframes(movies, credits=None)`.

## Run tests

```bash
pip install -e ".[test]"
pytest
```
