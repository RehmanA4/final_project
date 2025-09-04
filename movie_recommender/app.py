# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# -----------------------------
# Content-Based Filtering
# -----------------------------
def content_based_recommender(movie_title, movies, top_n=5):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    if movie_title not in indices:
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

# -----------------------------
# Collaborative Filtering (Simple User Avg Ratings)
# -----------------------------
def collaborative_recommender(user_id, ratings, movies, top_n=5):
    user_ratings = ratings[ratings['userId'] == user_id]
    top_rated = user_ratings.sort_values("rating", ascending=False).head(top_n)
    top_movies = pd.merge(top_rated, movies, on="movieId")
    return top_movies['title'].tolist()

# -----------------------------
# Hybrid Recommender
# -----------------------------
def hybrid_recommender(movie_title, user_id=None, top_n=5):
    content_recs = content_based_recommender(movie_title, movies, top_n=top_n)
    collab_recs = []

    if user_id:
        collab_recs = collaborative_recommender(user_id, ratings, movies, top_n=top_n)

    # Merge recommendations
    final_recs = list(dict.fromkeys(content_recs + collab_recs))[:top_n]
    return final_recs

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Suggesting movies based on user preferences using **ML**.")

st.sidebar.header("User Input")
movie_list = movies['title'].values
selected_movie = st.sidebar.selectbox("Choose a movie you like:", movie_list)

user_id_input = st.sidebar.text_input("Enter User ID (optional):")

if st.sidebar.button("Get Recommendations"):
    if selected_movie:
        user_id = int(user_id_input) if user_id_input.isdigit() else None
        recommendations = hybrid_recommender(selected_movie, user_id=user_id, top_n=5)

        st.subheader("✅ Top 5 Recommendations:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("Please select a movie.")
