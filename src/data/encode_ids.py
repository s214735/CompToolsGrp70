from pathlib import Path
import pandas as pd
import numpy as np

SILVER = Path("data/silver")

def main():
    df = pd.read_csv(SILVER / "cleaned_ratings.csv")

    # enforce base dtypes
    df["userId"] = df["userId"].astype("int64")
    df["movieId"] = df["movieId"].astype("int64")
    df["rating"] = df["rating"].astype("float32")

    # timestamp -> int64 epoch (or drop it if you don't need it)
    if "timestamp" in df.columns:
        if not np.issubdtype(df["timestamp"].dtype, np.integer):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["timestamp"] = df["timestamp"].view("int64")  # ns since epoch

    # safe strings
    for c in ["title", "genres"]:
        if c in df.columns:
            df[c] = df[c].astype("string")  # will serialize as plain strings

    # contiguous indices for embeddings
    df["user_idx"] = df["userId"].astype("category").cat.codes.astype("int32")
    df["movie_idx"] = df["movieId"].astype("category").cat.codes.astype("int32")

    cols = ["userId","movieId","user_idx","movie_idx","rating","timestamp","title","genres"]
    cols_to_keep = [c for c in cols if c in df.columns]
    out = df[cols_to_keep].copy()

    # one last guard: make sure object/extension dtypes are gone
    # cast strings to Python str so pyarrow doesn't store extension metadata
    for c in ["title","genres"]:
        if c in out.columns:
            out[c] = out[c].astype(object)

    # write fresh file
    (SILVER / "interactions_encoded.parquet").unlink(missing_ok=True)
    out.to_parquet(SILVER / "interactions_encoded.parquet", engine="pyarrow", index=False)

    # id maps
    df[["userId","user_idx"]].drop_duplicates().sort_values("user_idx").to_csv(SILVER / "user_id_map.csv", index=False)
    df[["movieId","movie_idx","title","genres"]].drop_duplicates().sort_values("movie_idx").to_csv(SILVER / "movie_id_map.csv", index=False)

if __name__ == "__main__":
    main()
