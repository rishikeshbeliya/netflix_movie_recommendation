import pandas as pd
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommend


class HybridRecommender:
    def __init__(self, content_df, collab_df, alpha=0.5):
        self.content = ContentRecommender(content_df)
        self.collab = CollaborativeRecommend(collab_df)
        self.movie_lookup = content_df.set_index("movieId")["title"].to_dict()
        self.alpha = alpha

    def build(self):
        self.content.build()
        self.collab.build()

    def hybrid_recommend(self, title, userId, top_n):
        content_recommend = self.content.content_recommend(title, top_n * 2)
        content_score = dict(
            zip(content_recommend["movieId"], content_recommend["similarity_score"])
        )

        collab_recommend = self.collab.collab_recommend(userId, top_n * 2)
        collab_score = {movieId: score for movieId, score in collab_recommend}

        seen = set(
            self.collab.ratings[self.collab.ratings["userId"] == userId]["movieId"]
        )
        all_movies = set(content_score) | set(collab_score)
        all_movies -= seen

        hybrid_scores = {}
        for key in all_movies:
            c_score = collab_score.get(key, 2.5)  # Default neutral rating
            t_score = content_score.get(key, 0)
            hybrid_scores[key] = (
                self.alpha * ((c_score - 1) / 4) + (1 - self.alpha) * t_score
            )

        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_hybrid[:top_n]
        return [(mid, self.movie_lookup[mid], score) for mid, score in top_movies]
