# app.py

import streamlit as st
import pandas as pd
from src.vector_search import SearchEngine
from src.llm_handler import get_explanation

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CineSearch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHING ---
@st.cache_resource
def load_search_engine():
    """Load the SearchEngine and cache it for the session."""
    print("Loading Search Engine for the first time...")
    return SearchEngine()

# --- INITIALIZATION ---
engine = load_search_engine()
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
# NEW: Initialize state for button clicks
if 'active_tconst' not in st.session_state:
    st.session_state.active_tconst = None

# --- UI LAYOUT ---
st.title("🎬 CineSearch: Semantic Movie Recommender")
st.markdown("Describe a plot, filter by details, or find movies with a twist to discover your next favorite film.")

# --- SIDEBAR FOR FILTERS ---
with st.sidebar:
    st.header("🔍 Search Filters")
    query_text = st.text_area("Describe a plot, theme, or character:", height=100, placeholder="e.g., a group of thieves attempts to plant an idea into a CEO's mind...")
    st.markdown("---")
    actor_filter = st.text_input("Actor", placeholder="e.g., Leonardo DiCaprio")
    director_filter = st.text_input("Director", placeholder="e.g., Christopher Nolan")
    genre_filter = st.text_input("Genre", placeholder="e.g., Sci-Fi")
    min_year, max_year = int(engine.df['startYear'].min()), int(engine.df['startYear'].max())
    year_range = st.slider("Release Year Range", min_year, max_year, (1990, 2023))
    rating_threshold = st.slider("Minimum IMDb Rating", 0.0, 10.0, 7.5, 0.1)
    
    if st.button("Search Movies", type="primary", use_container_width=True):
        filters = {
            'actors': actor_filter,
            'directors': director_filter,
            'genres': genre_filter,
            'startYear': year_range,
            'averageRating': rating_threshold
        }
        with st.spinner("Searching for movies..."):
            st.session_state.results_df = engine.search(query_text=query_text, filters=filters)
        # Reset active button state on new search
        st.session_state.active_tconst = None

# --- REVAMPED RESULT DISPLAY ---
if not st.session_state.results_df.empty:
    results_df = st.session_state.results_df
    st.success(f"Found {len(results_df)} matching movies. Displaying top results:")
    
    for _, movie in results_df.iterrows():
        tconst = movie['tconst']
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.metric("Rating", f"{movie['averageRating']:.1f} ⭐")
                st.metric("Year", str(movie['startYear']))
            with col2:
                st.subheader(movie['primaryTitle'])
                meta_info = [f"**Genres:** {movie.get('genres', 'N/A')}", f"**Directors:** {movie.get('directors', 'N/A')}", f"**Top Actors:** {movie.get('actors', 'N/A')}"]
                st.markdown(" | ".join(meta_info))
                
                with st.expander("Show Plot and Advanced Features"):
                    st.markdown(f"**Plot:** {movie['plot']}")
                    st.divider()
                    
                    # --- Buttons in their own row ---
                    c1, c2, _ = st.columns([1, 1, 2])
                    if c1.button("💡 Why is this a match?", key=f"explain_{tconst}"):
                        st.session_state.active_tconst = f"explain_{tconst}"
                    if c2.button("🎭 Find similar plot twists", key=f"twist_{tconst}"):
                        st.session_state.active_tconst = f"twist_{tconst}"
                    
                    # --- Output display area (full width) ---
                    if st.session_state.active_tconst == f"explain_{tconst}":
                        if not query_text:
                            st.error("Please enter a plot description in the sidebar to use this feature.")
                        else:
                            with st.spinner("Generating explanation..."):
                                explanation = get_explanation(recommended_movie=movie, query_text=query_text)
                                st.info(explanation)
                    
                    if st.session_state.active_tconst == f"twist_{tconst}":
                        with st.spinner("Finding movies with a different ending..."):
                            twist_df = engine.find_plot_twist(tconst=tconst)
                            if not twist_df.empty:
                                st.success("Found movies that start similarly but end differently:")
                                for _, twist_movie in twist_df.iterrows():
                                    st.markdown(f"- **{twist_movie['primaryTitle']}** ({twist_movie['startYear']})")
                            else:
                                st.warning("Couldn't find any plot twist recommendations for this movie.")

elif st.session_state.get('results_df') is not None and st.session_state.results_df.empty:
    st.warning("No movies found matching your criteria. Please try different filters.")