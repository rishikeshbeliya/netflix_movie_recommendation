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
        col_re = self.collab.collab_recommend(userId, top_n * 2)
        collab_score = {movie_id: score for movie_id, score in col_re} if col_re else {}
        con_re = self.content.content_recommend(title, top_n * 2)
        con_score = dict(zip(con_re["movieId"], con_re["similarity_score"]))

        all_movies = set(collab_score) | set(con_score)

        hybrid_scores = {}
        for mid in all_movies:
            c_score = collab_score.get(mid, 2.5)  # Default neutral rating
            t_score = con_score.get(mid, 0)
            hybrid_scores[mid] = (
                self.alpha * ((c_score - 1) / 4) + (1 - self.alpha) * t_score
            )

        # Top N with titles
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_hybrid[:top_n]
        return [(mid, self.movie_lookup[mid], score) for mid, score in top_movies]


content = pd.read_csv("../data/content_based_df.csv")
collab = pd.read_csv("../data/collaborative_df.csv")
x = HybridRecommender(content, collab)

# print(collab["userId"].unique())

print(x.hybrid_recommend("The Godfather", 154, 5))
