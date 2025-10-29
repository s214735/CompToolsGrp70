import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecommenderSystem:
    def __init__(self, ratings_df):
        self.ratings = ratings_df.pivot(index="userId", columns="title", values="rating").fillna(0)
        self.similarity = cosine_similarity(self.ratings.T)
        self.sim_df = pd.DataFrame(self.similarity, index=self.ratings.columns, columns=self.ratings.columns)

    def recommend(self, movie_title, n=5):
        return self.sim_df[movie_title].sort_values(ascending=False)[1:n+1]
