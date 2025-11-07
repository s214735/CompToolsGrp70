# src/data/preprocess.py
import json
from pathlib import Path
import pandas as pd

BRONZE = Path("data/bronze")
SILVER = Path("data/silver")

def read_bronze():
    movies = pd.read_csv(BRONZE / "movies.csv.zip", compression="zip")
    ratings = pd.read_csv(BRONZE / "ratings.csv.zip", compression="zip")
    # standardize dtypes
    movies["movieId"] = movies["movieId"].astype("int64")
    ratings["userId"] = ratings["userId"].astype("int64")
    ratings["movieId"] = ratings["movieId"].astype("int64")
    return movies, ratings

def clean_merge(movies, ratings):
    # drop obvious NA/duplicates
    ratings = ratings.dropna(subset=["rating"]).drop_duplicates(subset=["userId","movieId"], keep="last")
    # join titles/genres for convenience
    df = ratings.merge(movies, on="movieId", how="left")
    # timestamp → datetime (optional)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    return df

def apply_min_activity_filters(df, min_user_ratings=5, min_movie_ratings=5, max_users=None, max_movies=None):
    # filter sparse tails (tune if you want more/less aggressive)
    user_counts = df.groupby("userId").size()
    movie_counts = df.groupby("movieId").size()
    keep_users = user_counts[user_counts >= min_user_ratings].index
    keep_movies = movie_counts[movie_counts >= min_movie_ratings].index
    df = df[df["userId"].isin(keep_users) & df["movieId"].isin(keep_movies)]
    # optional caps (handy for first experiments)
    if max_users:
        keep_u = df["userId"].drop_duplicates().sort_values().head(max_users)
        df = df[df["userId"].isin(keep_u)]
    if max_movies:
        keep_m = df["movieId"].drop_duplicates().sort_values().head(max_movies)
        df = df[df["movieId"].isin(keep_m)]
    return df

def save_silver(df):
    SILVER.mkdir(parents=True, exist_ok=True)
    # core clean table
    df.to_parquet(SILVER / "cleaned_ratings.parquet", index=False)
    df.to_csv(SILVER / "cleaned_ratings.csv", index=False)

    # slim helper tables
    users = df[["userId"]].drop_duplicates().sort_values("userId")
    movies = df[["movieId","title","genres"]].drop_duplicates().sort_values("movieId")
    users.to_csv(SILVER / "users.csv", index=False)
    movies.to_csv(SILVER / "movies.csv", index=False)

    # quick stats (nice for report)
    stats = {
        "n_rows": int(len(df)),
        "n_users": int(users.shape[0]),
        "n_movies": int(movies.shape[0]),
        "density": float(len(df) / (users.shape[0] * max(1, movies.shape[0]))),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "rating_mean": float(df["rating"].mean()),
    }
    (SILVER / "stats.json").write_text(json.dumps(stats, indent=2))

def main():
    movies, ratings = read_bronze()
    df = clean_merge(movies, ratings)
    df = apply_min_activity_filters(df, min_user_ratings=5, min_movie_ratings=5)
    save_silver(df)

if __name__ == "__main__":
    main()
