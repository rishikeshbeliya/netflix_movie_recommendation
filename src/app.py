import streamlit as st
import pandas as pd
from hybrid_model import HybridRecommender


# =========================
# Load Data (cache this)
# =========================
@st.cache_data
def load_data():
    content = pd.read_csv("../data/content_based_df.csv")
    content = content.rename(columns={"id": "movieId"})

    collab = pd.read_csv("../data/collaborative_df.csv")
    return content, collab


content_df, collab_df = load_data()


# =========================
# Build Model (cache)
# =========================
@st.cache_resource
def build_model(content, collab):
    model = HybridRecommender(content, collab, alpha=0.5)
    model.build()
    return model


model = build_model(content_df, collab_df)

# =========================
# UI
# =========================
st.title("🎬 Hybrid Movie Recommender System")

st.markdown("""
This system combines:

- **Content-Based Filtering** (movie similarity)
- **Collaborative Filtering** (user preferences)

Final Score:
`Hybrid = α * Collaborative + (1 - α) * Content`
""")

# =========================
# Inputs
# =========================
st.sidebar.header("Controls")

movie_list = sorted(content_df["title"].dropna().unique())

selected_movie = st.sidebar.selectbox("Select a Movie", movie_list)

user_id = st.sidebar.number_input("User ID", min_value=1, max_value=1000, value=1)

alpha = st.sidebar.slider("Alpha (Collaborative Weight)", 0.0, 1.0, 0.5)

top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Update alpha dynamically
model.alpha = alpha

# =========================
# Recommendation Button
# =========================
if st.button("Get Recommendations"):
    results = model.hybrid_recommend(selected_movie, user_id, top_n)

    if not results:
        st.warning("No recommendations found. Try a different input.")
    else:
        st.subheader("📌 Recommendations")

        df_results = pd.DataFrame(results, columns=["movieId", "title", "score"])
        st.dataframe(df_results)

        # Show explanation
        st.subheader("🧠 How it works")
        st.markdown(f"""
        - Content similarity based on **plot, genre, keywords**
        - Collaborative filtering based on **user {user_id}'s preferences**
        - α = **{alpha}**
        """)

# =========================
# Debug / Transparency
# =========================
with st.expander("🔍 Show Raw Data"):
    st.write(content_df.head())
