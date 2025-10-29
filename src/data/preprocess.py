import pandas as pd
from pathlib import Path

def load_data(bronze_path: str):
    movies = pd.read_csv(Path(bronze_path) / "movies.csv.zip", compression='zip')
    ratings = pd.read_csv(Path(bronze_path) / "ratings.csv.zip", compression='zip')
    return movies, ratings

def preprocess_data(movies, ratings):
    df = ratings.merge(movies, on="movieId")
    df.dropna(subset=["rating"], inplace=True)
    return df

def save_preprocessed(df, silver_path: str):
    Path(silver_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(silver_path) / "cleaned_ratings.csv", index=False)
