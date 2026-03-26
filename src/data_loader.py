# Dataset = "https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset"
import pandas as pd

#!==============================
#! Content Based Filtering
#!==============================

movies = pd.read_csv("../data/movies_metadata.csv", low_memory=False)
movies = movies[["id", "title", "overview", "genres", "tagline"]]
movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
movies["id"] = movies["id"].dropna().astype(int)
movies = movies.rename(columns={"id": "movieId"})

keywords = pd.read_csv("../data/keywords.csv", low_memory=False)
keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")
keywords["id"] = keywords["id"].dropna().astype(int)
keywords = keywords.rename(columns={"id": "movieId"})

content_based_df = pd.merge(movies, keywords, on="movieId", how="inner")

content_based_df.to_csv("../data/content_based_df.csv", index=False)


#!==============================
#! Collaborative Filtering
#!==============================

ratings = pd.read_csv("../data/ratings_small.csv", low_memory=False)
links = pd.read_csv("../data/links_small.csv", low_memory=False)
temp = pd.merge(ratings, links, how="inner", on="movieId")
temp = temp.drop(columns=["timestamp", "imdbId", "movieId"])
temp = temp.rename(columns={"tmdbId": "movieId"})


collaborative_df = pd.merge(
    movies[["movieId", "title"]],
    temp,
    left_on="movieId",
    right_on="movieId",
    how="left",
)
collaborative_df.to_csv("../data/collaborative_df.csv", index=False)
