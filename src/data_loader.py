# Dataset = "https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset"

import pandas as pd


#!==============================
#! Content Based Filtering
#!==============================

movies = pd.read_csv("../data/movies_metadata.csv", low_memory=False)
movies = movies[["id", "title", "overview", "genres", "tagline", "imdb_id"]]
movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
movies["id"] = movies["id"].dropna().astype(int)

keywords = pd.read_csv("../data/keywords.csv", low_memory=False)
keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")
keywords["id"] = keywords["id"].dropna().astype(int)

content_based_df = pd.merge(movies, keywords, on="id", how="inner")

content_based_df.to_csv("../data/content_based_df.csv", index=False)


#!==============================
#! Collaborative Filtering
#!==============================

ratings = pd.read_csv("../data/ratings_small.csv", low_memory=False)
links = pd.read_csv("../data/links_small.csv", low_memory=False)

temp = pd.merge(ratings, links, how="inner", on="movieId")
temp = temp.drop(columns=["timestamp", "tmdbId"])
movies["imdb_id"] = pd.to_numeric(movies["imdb_id"], errors="coerce")
movies["imdb_id"] = movies["imdb_id"].dropna().astype(int)

collaborative_df = pd.merge(
    movies[["title", "imdb_id"]],
    temp,
    left_on="imdb_id",
    right_on="imdbId",
    how="inner",
)
collaborative_df.to_csv("../data/collaborative_df.csv", index=False)
