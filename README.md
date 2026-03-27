# 🎬 Netflix Movie Recommendation System

A hybrid movie recommendation system built using both **Content-Based Filtering** and **Collaborative Filtering**, combined into a **Hybrid Model** for improved recommendation accuracy.

---

## 📌 Overview

This project leverages the **Movies Dataset from Kaggle** to build three types of recommendation systems:

1. **Content-Based Filtering**
2. **Collaborative Filtering (SVD-based)**
3. **Hybrid Recommendation System**

The goal is to recommend movies based on:

* Movie content (genres, overview, keywords, etc.)
* User behavior (ratings)
* A weighted combination of both approaches

---

## 📂 Dataset

Dataset used:
👉 https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Files used:

* `movies_metadata.csv`
* `keywords.csv`
* `ratings_small.csv`
* `links_small.csv`

---

## ⚙️ Data Processing

### Content-Based Data

* Merged `movies_metadata` with `keywords`
* Selected relevant features:

  * genres
  * tagline
  * overview
  * keywords
  * title
* Created a unified text feature: `content_filter`

### Collaborative Data

* Combined:

  * ratings
  * links
  * movie metadata
* Cleaned unnecessary columns and aligned movie IDs

---

## 🧠 Models

### 1. Content-Based Recommender

* Uses **TF-IDF Vectorization**
* Applies **Truncated SVD (Dimensionality Reduction)**
* Computes similarity using **Cosine Similarity**

**Key Features:**

* Handles high-dimensional text efficiently
* Fast lookup using precomputed vectors
* Recommends movies similar to a given title

---

### 2. Collaborative Filtering

* Built using **SVD (Singular Value Decomposition)** from `surprise`
* Learns user-item interaction patterns

**Key Features:**

* Predicts ratings for unseen movies
* Generates personalized recommendations per user
* Evaluated using RMSE

---

### 3. Hybrid Recommender

Combines both approaches:

[
Hybrid Score = \alpha \cdot Collaborative + (1 - \alpha) \cdot Content
]

**Key Features:**

* Balances personalization and similarity
* Handles cold-start problems better than pure collaborative filtering
* Configurable weight (`alpha`)

---

## 🚀 How to Run

### 1. Prepare Data

```bash
python data_preprocessing.py
```

This generates:

* `content_based_df.csv`
* `collaborative_df.csv`

---

### 2. Build Models

```python
from hybrid_model import HybridRecommender
import pandas as pd

content_df = pd.read_csv("data/content_based_df.csv")
collab_df = pd.read_csv("data/collaborative_df.csv")

model = HybridRecommender(content_df, collab_df, alpha=0.5)
model.build()
```

---

### 3. Get Recommendations

```python
model.hybrid_recommend(
    title="The Dark Knight",
    userId=1,
    top_n=10
)
```

---

## 📊 Output Format

The hybrid recommender returns:

```
(movieId, title, hybrid_score)
```

---

## ⚠️ Limitations (Be Honest)

* **Memory-heavy**: Precomputing predictions for all users × movies is inefficient
* **Scalability issue**: Not suitable for large-scale production systems
* **Cold start still partially unresolved**
* **Content quality depends heavily on text preprocessing**

---

## 🔧 Possible Improvements

* Use **Approximate Nearest Neighbors (FAISS)** for faster similarity search
* Replace SVD with **Neural Collaborative Filtering**
* Add **user embeddings**
* Optimize prediction storage (avoid full matrix precomputation)
* Deploy using **FastAPI + Docker**

---

## 🧪 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Surprise (SVD)
* TF-IDF + Cosine Similarity
* Truncated SVD

---

## 💡 Key Learning Outcomes

* Built recommender systems from scratch
* Understood trade-offs between content vs collaborative approaches
* Learned hybrid modeling techniques
* Dealt with real-world messy datasets

---

## 📌 Final Note

This project is a **solid intermediate-level implementation**, not production-ready.
If you're aiming for real-world systems, focus next on:

* scalability
* latency optimization
* model serving

---
