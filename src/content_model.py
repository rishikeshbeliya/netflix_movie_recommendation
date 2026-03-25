import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentRecommender:
    def __init__(self, df):
        self.df = df.copy()

    def build(self):
        self.df["genres"] = self.df["genres"].fillna("")
        self.df["tagline"] = self.df["tagline"].fillna("")
        self.df["overview"] = self.df["overview"].fillna("")
        self.df["keywords"] = self.df["keywords"].fillna("")
        self.df["title"] = self.df["title"].fillna("")

        self.df["content_filter"] = (
            self.df["genres"]
            + ""
            + self.df["tagline"]
            + ""
            + self.df["overview"]
            + ""
            + self.df["keywords"]
            + ""
            + self.df["title"]
        ).str.lower()

        tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000
        )

        self.tfidf_ = tfidf.fit_transform(self.df["content_filter"])

        # TODO: Fast Look-Up Table
        self.indices = pd.Series(
            self.df.index, index=self.df["title"]
        ).drop_duplicates()

    def recommend(self, title, top_n):
        idx = self.indices.get(title)

        if idx is None:
            print(f"No result found for {title} :(")
            return pd.DataFrame(columns=["title", "id"])

        cosine_similarty = linear_kernel(self.tfidf_[idx], self.tfidf_).flatten()
        similarity_score = list(enumerate(cosine_similarty))
        similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        top_ns = [i for i, _ in similarity_score[1 : top_n + 1]]
        return self.df.iloc[top_ns][["title", "id"]]
