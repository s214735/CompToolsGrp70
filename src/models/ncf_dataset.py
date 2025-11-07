# src/models/ncf_dataset.py
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    def __init__(self, parquet_path: Path, user_col="user_idx", movie_col="movie_idx", rating_col="rating"):
        self.df = pd.read_parquet(parquet_path)
        self.user_col = user_col
        self.movie_col = movie_col
        self.rating_col = rating_col

        # ensure dtype for torch
        self.users = self.df[self.user_col].astype("int64").to_numpy()
        self.movies = self.df[self.movie_col].astype("int64").to_numpy()
        self.ratings = self.df[self.rating_col].astype("float32").to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        u = torch.tensor(self.users[idx], dtype=torch.long)
        i = torch.tensor(self.movies[idx], dtype=torch.long)
        r = torch.tensor(self.ratings[idx], dtype=torch.float32)
        return u, i, r
