"""Streamlit front end for the movie recommender."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from movie_recommender import MovieNotFoundError, MovieRecommender, normalize_title


ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = ROOT / "movie_recommender" / "sample_movies.csv"

st.set_page_config(
    page_title="Reel Finder · Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      [data-testid="stHeader"] { background: rgba(245,247,251,.88); }
      [data-testid="stSidebar"] { background: #111827; }
      [data-testid="stSidebar"] * { color: #edf2f7; }
      .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero {
        padding: 2.1rem 2.3rem; border-radius: 24px; color: #fff;
        background: radial-gradient(circle at 85% 5%, rgba(255,184,107,.32), transparent 30%),
                    linear-gradient(125deg, #172554 0%, #312e81 56%, #7c2d52 100%);
        box-shadow: 0 18px 50px rgba(42,47,100,.17); margin-bottom: 1.2rem;
      }
      .hero-kicker { color: #fcd9a7; letter-spacing: .16em; text-transform: uppercase;
                     font-size: .75rem; font-weight: 800; margin-bottom: .65rem; }
      .hero h1 { color: #fff; font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1.06;
                 margin: 0 0 .7rem 0; letter-spacing: -.04em; }
      .hero p { color: #e0e7ff; font-size: 1.05rem; max-width: 720px; margin: 0; }
      [data-testid="stMetric"] { background: #fff; padding: 1rem; border: 1px solid #e7eaf2;
                                  border-radius: 16px; box-shadow: 0 5px 18px rgba(24,39,75,.04); }
      [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; }
      div.stButton > button { border-radius: 10px; font-weight: 700; }
      .small-label { color: #64748b; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _build_engine(movies_bytes: bytes, credits_bytes: bytes | None) -> MovieRecommender:
    movies = pd.read_csv(io.BytesIO(movies_bytes), low_memory=False)
    credits = pd.read_csv(io.BytesIO(credits_bytes), low_memory=False) if credits_bytes else None
    return MovieRecommender.from_dataframes(movies, credits)


def _pretty_list(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return "Not listed"
    return " · ".join(part.strip() for part in text.replace("|", ",").split(",") if part.strip())


def _render_cards(movies: pd.DataFrame, *, show_similarity: bool = True) -> None:
    if movies.empty:
        st.info("No other movies are available to recommend from this catalog.")
        return

    columns = st.columns(3, gap="medium")
    for position, (_, movie) in enumerate(movies.iterrows()):
        with columns[position % 3]:
            with st.container(border=True):
                release = str(movie.get("release_date", ""))[:4]
                if not release.isdigit():
                    release = "Year unknown"
                st.markdown(f"#### {movie['title']}")
                st.caption(f"{release}  ·  {_pretty_list(movie.get('genres', ''))}")
                director = str(movie.get("director", "")).strip()
                if director and director.casefold() not in {"nan", "none"}:
                    st.write(f"**Director:** {director.replace('|', ', ')}")
                overview = str(movie.get("overview", "")).strip()
                if overview and overview.casefold() not in {"nan", "none"}:
                    st.write(overview)
                rating = float(movie.get("vote_average", 0.0))
                votes = int(float(movie.get("vote_count", 0.0)))
                st.caption(f"★ {rating:.1f}/10  ·  {votes:,} ratings")
                if show_similarity:
                    similarity = float(movie.get("similarity", 0.0))
                    rank_score = float(movie.get("recommendation_score", 0.0))
                    st.progress(max(0.0, min(1.0, similarity)))
                    st.caption(
                        f"Content match {similarity * 100:.1f}%  ·  "
                        f"Rank score {rank_score * 100:.1f}%"
                    )
                else:
                    score = float(movie.get("popularity_score", 0.0))
                    st.caption(f"Smoothed popularity score {score * 100:.1f}%")


with st.sidebar:
    st.markdown("## 🎞️ Reel Finder")
    st.caption("A metadata-powered movie discovery lab")
    st.divider()
    st.markdown("### Movie catalog")
    movies_upload = st.file_uploader(
        "Upload a movies CSV",
        type=("csv",),
        help="Accepts TMDB 5000 movies data or a CSV with title and metadata columns.",
    )
    credits_upload = st.file_uploader(
        "Optional credits CSV",
        type=("csv",),
        help="For the separate TMDB credits file, upload it here to add cast and director features.",
    )
    if movies_upload is None:
        st.info("Using the included 40-title sample catalog. Upload your full dataset to explore more movies.")
        movies_bytes = SAMPLE_DATA.read_bytes()
        credits_bytes = None
        data_label = "Built-in sample"
    else:
        movies_bytes = movies_upload.getvalue()
        credits_bytes = credits_upload.getvalue() if credits_upload else None
        data_label = movies_upload.name
    if credits_upload is not None and movies_upload is None:
        st.warning("Upload the movies CSV as well as the credits CSV.")

    st.divider()
    st.caption("Engine: weighted TF-IDF · cosine similarity · fuzzy title matching · Bayesian popularity")

source_key = hashlib.sha256(movies_bytes + (credits_bytes or b"")).hexdigest()
if st.session_state.get("_source_key") != source_key:
    for key in ("_recommend_query", "_recommend_mode", "_recommend_weight", "_recommend_count"):
        st.session_state.pop(key, None)
    st.session_state["_source_key"] = source_key

try:
    with st.spinner("Preparing movie features and recommendations…"):
        recommender = _build_engine(movies_bytes, credits_bytes)
except Exception as error:  # show actionable dataset errors instead of a blank app
    st.error(f"Could not load this catalog: {error}")
    st.stop()

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">A better next-watch list</div>
      <h1>Find your next favorite film.</h1>
      <p>Search a movie you love. Reel Finder reads its genres, story, cast, keywords, and filmmaker
      to surface titles with a similar feel—then balances the results with audience popularity.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(3)
metric_columns[0].metric("Movies indexed", f"{recommender.number_of_movies:,}")
metric_columns[1].metric("Text features", f"{recommender.feature_matrix_shape[1]:,}")
metric_columns[2].metric("Data source", data_label)

recommend_tab, popular_tab, method_tab = st.tabs(["✨ Discover", "🔥 Popular picks", "🧠 How it works"])

with recommend_tab:
    st.subheader("Recommendations based on a movie")
    st.write("Enter a title—even an approximate spelling works—and choose how much popularity should influence the ranking.")
    with st.form("recommendation-search"):
        search_col, mode_col = st.columns([2, 1])
        with search_col:
            query = st.text_input("Movie title", placeholder="Try Inception, Spirited Away, or The Martian")
        with mode_col:
            mode_label = st.selectbox(
                "Ranking style",
                ("Balanced · content + popularity", "Content similarity only"),
            )
        settings_col, weight_col = st.columns([1, 2])
        with settings_col:
            result_count = st.slider("Number of recommendations", min_value=3, max_value=15, value=9)
        with weight_col:
            popularity_weight = st.slider(
                "Popularity influence",
                min_value=0.0,
                max_value=0.5,
                value=0.15,
                step=0.05,
                help="Used in balanced mode. The remaining weight comes from TF-IDF cosine similarity.",
            )
        submitted = st.form_submit_button("Find recommendations", type="primary", use_container_width=True)

    if submitted:
        st.session_state["_recommend_query"] = query.strip()
        st.session_state["_recommend_mode"] = "content" if mode_label.startswith("Content") else "hybrid"
        st.session_state["_recommend_weight"] = float(popularity_weight)
        st.session_state["_recommend_count"] = int(result_count)

    saved_query = st.session_state.get("_recommend_query", "")
    if saved_query:
        try:
            matched_movie = recommender.find_movie(saved_query)
            if normalize_title(saved_query) != normalize_title(matched_movie["title"]):
                st.info(f"Closest match in this catalog: **{matched_movie['title']}**")
            selected_mode = st.session_state.get("_recommend_mode", "hybrid")
            selected_weight = st.session_state.get("_recommend_weight", 0.15)
            selected_count = st.session_state.get("_recommend_count", 9)
            recommendations = recommender.recommend(
                matched_movie["title"],
                top_n=selected_count,
                mode=selected_mode,
                popularity_weight=selected_weight,
            )
            if selected_mode == "hybrid":
                st.caption(f"Hybrid ranking: {(1 - selected_weight) * 100:.0f}% content match · {selected_weight * 100:.0f}% popularity")
            else:
                st.caption("Ranked by text-based content similarity only.")
            _render_cards(recommendations)
        except MovieNotFoundError as error:
            st.warning(str(error))
            suggestions = recommender.suggest_titles(saved_query)
            if suggestions:
                suggestion = st.selectbox("Closest titles", suggestions)
                if st.button("Recommend from this title", type="primary"):
                    st.session_state["_recommend_query"] = suggestion
                    st.rerun()
    elif submitted:
        st.info("Enter a movie title to get started.")
    else:
        st.markdown("**Try a sample search:** `Inception` · `The Matrix` · `Coco` · `Parasite`")

with popular_tab:
    st.subheader("Popular across this catalog")
    st.write("A catalog-wide list ranked with a smoothed audience rating and a popularity signal.")
    popular_count = st.slider("Movies to show", min_value=3, max_value=18, value=9, key="popular-count")
    _render_cards(recommender.popular_movies(top_n=popular_count), show_similarity=False)

with method_tab:
    st.subheader("A small, explainable recommendation pipeline")
    method_columns = st.columns(3)
    method_columns[0].markdown(
        "**01 · Prepare metadata**\n\n"
        "TMDB-style JSON fields are parsed into readable genres, keywords, cast, and director text. "
        "The sample also uses taglines and story overviews."
    )
    method_columns[1].markdown(
        "**02 · Build movie fingerprints**\n\n"
        "A weighted TF-IDF vectorizer emphasizes genres and keywords, then represents every title as "
        "a sparse text vector with unigram and bigram features."
    )
    method_columns[2].markdown(
        "**03 · Rank recommendations**\n\n"
        "Cosine similarity finds content neighbors. Hybrid mode blends similarity with a Bayesian-style "
        "rating prior and scaled catalog popularity; `difflib` handles approximate title searches."
    )
    st.info(
        "The engine computes one sparse similarity vector per search instead of storing a dense all-pairs matrix. "
        "This keeps the workflow practical for a 4,800+ movie catalog. Scores are ranking signals, not calibrated probabilities."
    )
    st.markdown("#### Bring your own data")
    st.write(
        "Upload a movies CSV in the sidebar. For TMDB 5000, upload `tmdb_5000_movies.csv` and "
        "`tmdb_5000_credits.csv`; the engine joins them by movie ID. A single merged CSV works too."
    )
    st.code(
        'python -m movie_recommender --data data/tmdb_5000_movies.csv '
        '--credits data/tmdb_5000_credits.csv recommend --title "Inception" --top 10',
        language="bash",
    )
