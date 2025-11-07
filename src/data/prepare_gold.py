from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

SILVER = Path("data/silver")
GOLD = Path("data/gold")


def main():
    GOLD.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load encoded silver data ----------
    df = pd.read_parquet(SILVER / "interactions_encoded.parquet")
    logging.info(f"Loaded {len(df):,} interactions from silver")

    n_users = df["user_idx"].nunique()
    n_movies = df["movie_idx"].nunique()
    logging.info(f"Users: {n_users:,}, Movies: {n_movies:,}")

    # ---------- 2. Train/validation/test split ----------
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    logging.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    train.to_parquet(GOLD / "train.parquet", index=False)
    val.to_parquet(GOLD / "val.parquet", index=False)
    test.to_parquet(GOLD / "test.parquet", index=False)

    # ---------- 3. Sparse user–movie matrix for clustering ----------
    logging.info("Building sparse user–movie matrix (this can take a bit)...")
    rows = df["user_idx"].to_numpy()
    cols = df["movie_idx"].to_numpy()
    vals = df["rating"].astype(np.float32).to_numpy()
    user_movie_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))
    sp.save_npz(GOLD / "user_movie_matrix_sparse.npz", user_movie_sparse)
    logging.info(f"Sparse matrix saved to {GOLD / 'user_movie_matrix_sparse.npz'}")

    # ---------- 4. Small dense sample for quick visualization ----------
    sample_users = np.random.choice(n_users, size=min(2000, n_users), replace=False)
    sample = df[df["user_idx"].isin(sample_users)]
    sample_matrix = sample.pivot_table(
        index="user_idx", columns="movie_idx", values="rating", fill_value=0
    ).astype("float32")
    sample_matrix.to_parquet(GOLD / "user_movie_matrix_sample.parquet")
    logging.info("Saved small dense sample matrix for EDA.")

    # ---------- 5. Save statistics ----------
    stats = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "density": float(len(df) / (n_users * n_movies)),
    }
    with open(GOLD / "gold_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logging.info("Gold layer ready!")


if __name__ == "__main__":
    main()
