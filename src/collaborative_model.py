import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split
from surprise import SVD, Dataset, Reader, accuracy


class CollaborativeRecommend:
    def __init__(
        self,
        ratings,
    ):
        self.ratings = ratings
        self.algo = SVD()
        self.prediction = dict()

    def build(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings[["userId", "movieId", "rating"]], reader
        )

        self.train, self.test = train_test_split(data, test_size=0.2)
        self.algo.fit(self.train)

        for u in self.ratings["userId"].unique():
            self.prediction[u] = {}

            for m in self.ratings["movieId"].unique():
                est = self.algo.predict(u, m).est
                self.prediction[u][m] = est

    def collab_recommend(self, user_id, top_n):
        if user_id not in self.prediction:
            print(f"User {user_id} not found in predictions.")
            return []

        # Movies user has already seen
        seen_movies = set(self.ratings[self.ratings["userId"] == user_id]["movieId"])
        # Candidate movies = all - seen
        all_movies = set(self.ratings["movieId"].unique())
        candidates = list(all_movies - seen_movies)

        predictions = []

        for m in candidates:
            est = self.algo.predict(user_id, m).est
            predictions.append((m, est))

        # Sort and return top_n
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

    def evaluate(self):
        predictions = self.algo.test(self.test)
        return accuracy.rmse(predictions)
