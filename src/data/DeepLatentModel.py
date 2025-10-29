# ==============================  autoencoder_training.py  ==============================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import random


# ------------------ Utility ------------------
def sample_fraction(df, fraction: float = 0.1, seed: int = 42):
    return df.sample(frac=fraction, random_state=seed)


# ------------------ Dataset ------------------
class MovieRatingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path="src/data/movie_ratings.csv",
                 movie_path="src/data/movies.csv",
                 keep_fraction: float = 0.10,
                 max_allowed_len: int = 5000,
                 target_length: int = 500,
                 seed: int = 42):
        data_path = Path(data_path)
        movie_path = Path(movie_path)

        ratings_df = pd.read_csv(data_path, usecols=['movieId', 'rating'])
        movies_df  = pd.read_csv(movie_path, usecols=['movieId', 'genres'])

        grouped = ratings_df.groupby('movieId')['rating'].agg(
            ratings_list=lambda x: x.tolist(),
            rating_count='count',
            rating_mean='mean'
        ).reset_index()

        merged = grouped.merge(movies_df, on='movieId', how='left')
        merged['genres_list'] = merged['genres'].str.split('|')

        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(merged['genres_list'])
        print(f"Found {len(self.mlb.classes_)} unique genres")

        before = len(merged)
        merged = merged[merged['ratings_list'].apply(len) <= max_allowed_len]
        print(f"Removed {before - len(merged)} movies longer than {max_allowed_len}")

        merged = sample_fraction(merged, fraction=keep_fraction, seed=seed)
        print(f"Reduced dataset: {len(merged)} movies")

        self.target_length = target_length
        merged['ratings_list'] = merged['ratings_list'].apply(self._pad_or_truncate)
        merged['genres_onehot'] = self.mlb.transform(merged['genres_list']).tolist()

        self.movie_ids = merged['movieId'].values
        self.ratings   = merged['ratings_list'].values
        self.genres    = merged['genres_onehot'].values
        self.counts    = merged['rating_count'].values
        self.means     = merged['rating_mean'].values

        print(f"Final dataset size: {len(self)} movies")

    def _pad_or_truncate(self, ratings):
        if len(ratings) < self.target_length:
            return ratings + [0.0] * (self.target_length - len(ratings))
        else:
            return ratings[:self.target_length]

    def __len__(self):
        return len(self.movie_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ratings[idx], dtype=torch.float),
            torch.tensor(self.genres[idx], dtype=torch.float)
        )


# ------------------ Model ------------------
class RatingsGenreAutoencoder(nn.Module):
    def __init__(self, rating_dim: int, genre_dim: int, latent_dim: int = 128):
        super().__init__()
        input_dim = rating_dim + genre_dim

        # Encoder: compress concatenated input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Decoder: reconstruct ratings
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, rating_dim)
        )

    def forward(self, ratings, genres):
        x = torch.cat([ratings, genres], dim=1)
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z


# ------------------ Training Loop ------------------
def train_autoencoder(dataset, epochs=10, batch_size=64, lr=1e-3, latent_dim=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    rating_dim = dataset.target_length
    genre_dim = len(dataset.mlb.classes_)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RatingsGenreAutoencoder(rating_dim, genre_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for ratings, genres in dataloader:
            ratings, genres = ratings.to(device), genres.to(device)

            optimizer.zero_grad()
            reconstructed, _ = model(ratings, genres)
            loss = criterion(reconstructed, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * ratings.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.6f}")

    return model


# ------------------ Run Example ------------------
if __name__ == "__main__":
    dataset = MovieRatingDataset(
        keep_fraction=0.05,
        target_length=200,
        max_allowed_len=200
    )

    model = train_autoencoder(dataset, epochs=1000, batch_size=64, lr=1e-4, latent_dim=32)
