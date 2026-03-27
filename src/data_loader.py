# Dataset = "https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset"
import pandas as pd

#!==============================
#! Content Based Filtering
#!==============================
movies = pd.read_csv("../data/movies_metadata.csv", low_memory=False)
movies = movies[
    ["id", "title", "overview", "genres", "tagline"]
]  # ?Keeping relevant columns only

movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
movies["id"] = movies["id"].dropna().astype(int)

# Todo: Importing keywords
keywords = pd.read_csv("../data/keywords.csv", low_memory=False)
keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")
keywords["id"] = keywords["id"].dropna().astype(int)

# Todo: Creating content based filtering DF
content_based_df = pd.merge(movies, keywords, on="id", how="inner")

content_based_df.to_csv("../data/content_based_df_new.csv", index=False)


#!==============================
#! Collaborative Filtering
#!==============================
ratings = pd.read_csv("../data/ratings_small.csv", low_memory=False)
links = pd.read_csv("../data/links_small.csv", low_memory=False)
collaborative_df = pd.merge(ratings, links, how="inner", on="movieId")
collaborative_df = collaborative_df.drop(columns=["timestamp", "imdbId"])

collaborative_df.to_csv("../data/collaborative_df_new.csv", index=False)


#!==============================
#! Hybrid Filtering
#!==============================
hybrid_based_df = pd.merge(
    collaborative_df, content_based_df, left_on="tmdbId", right_on="id"
)

hybrid_based_df = hybrid_based_df.drop(columns=["tmdbId"])

hybrid_based_df.to_csv("../data/hybrid.csv", index=False)
