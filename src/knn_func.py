import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import random
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from tqdm import tqdm
# --------------------------
# Dataset: user-item matrix
# --------------------------
class UserRatingDataset(Dataset):
    def __init__(self,
                 ratings_csv="src/data/movie_ratings.csv",
                 movies_csv="src/data/movies.csv",
                 min_movie_count=5,
                 debias_years=True,
                 downsample_overrated_years_frac=0.4,
                 min_year=1920,
                 max_year=2025):
        """
        Anti-1995 Apocalypse Edition
        """
        ratings = pd.read_csv(ratings_csv)
        movies = pd.read_csv(movies_csv)

        # Merge to get titles and extract year
        ratings = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')
        ratings['year'] = ratings['title'].str.extract(r'\((\d{4})\)')
        ratings['year'] = pd.to_numeric(ratings['year'], errors='coerce')
        ratings = ratings.dropna(subset=['year'])
        ratings['year'] = ratings['year'].astype(int)

        # === 1. HARD YEAR FILTER (remove junk eras) ===
        ratings = ratings[ratings['year'].between(min_year, max_year)]

        # === 2. PRUNE RARE MOVIES ===
        movie_counts = ratings['movieId'].value_counts()
        keep_movies = movie_counts[movie_counts >= min_movie_count].index
        ratings = ratings[ratings['movieId'].isin(keep_movies)].copy()
        print(f"Movies kept after min_count={min_movie_count}: {len(keep_movies)}")

        # === 3. TEMPORAL DEBIASING: Downsample 1993–1998 (the MovieLens plague) ===
        if debias_years:
            plague_years = (ratings['year'] >= 1993) & (ratings['year'] <= 1998)
            plague_count = plague_years.sum()
            if plague_count > 0:
                keep = ratings[plague_years].sample(frac=downsample_overrated_years_frac, random_state=42)
                drop = ratings[plague_years].drop(keep.index)
                ratings = pd.concat([ratings[~plague_years], keep], ignore_index=True)
            else:
                print("No 1993–1998 ratings found. You're using a clean dataset!")

        # Drop temp columns
        ratings = ratings.drop(columns=['title', 'year'])

        # === 4. REINDEX USERS AND MOVIES ===
        self.user_ids = np.sort(ratings['userId'].unique())
        self.movie_ids = np.sort(ratings['movieId'].unique())
        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie2idx = {m: i for i, m in enumerate(self.movie_ids)}

        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)
        print(f"Final → Users: {n_users}, Movies: {n_movies}, Ratings: {len(ratings):,}")

        # Build dense matrix
        ratings_matrix = np.zeros((n_users, n_movies), dtype=np.float32)
        mask = np.zeros_like(ratings_matrix, dtype=np.float32)

        for row in ratings.itertuples(index=False):
            u_idx = self.user2idx[row.userId]
            m_idx = self.movie2idx[row.movieId]
            ratings_matrix[u_idx, m_idx] = row.rating
            mask[u_idx, m_idx] = 1.0

        # === 5. USER NORMALIZATION (mean/std) ===
        sums = (ratings_matrix * mask).sum(axis=1, keepdims=True)
        counts = mask.sum(axis=1, keepdims=True)
        counts[counts == 0] = 1.0
        mean_user = sums / counts

        centered = (ratings_matrix - mean_user) * mask
        var = ((centered**2) * mask).sum(axis=1, keepdims=True) / counts
        std_user = np.sqrt(var)
        std_user[std_user == 0.0] = 1.0
        normalized = centered / std_user

        self.ratings = normalized.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.mean_user = mean_user.astype(np.float32)
        self.std_user = std_user.astype(np.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        real_ratings = (self.ratings[idx] * self.std_user[idx]) + self.mean_user[idx]
        real_ratings = real_ratings * self.mask[idx]
        return (
            torch.from_numpy(real_ratings),
            torch.from_numpy(self.mask[idx])
        )
    
# --------------------------
# Simple user autoencoder
# --------------------------
class UserVAE(nn.Module):
    def __init__(self, num_movies, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_movies, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, latent_dim * 2)  # mean + logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, num_movies)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        rec = self.decoder(z)
        return rec, mu, logvar, z

    def encode(self, x):
            self.eval()
            with torch.no_grad():
                h = self.encoder(x)
                mu, _ = h.chunk(2, dim=-1)
                return mu  # Use mean only — stable, no sampling noise
            
# --------------------------
# Training routine
# --------------------------
def train_user_ae(train_dataset, val_dataset=None, latent_dim=64,
                  batch_size=128, epochs=100,
                  lr=1e-4, device=None, save_path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_movies = train_dataset[0][0].shape[0]
    print(num_movies)
    model = UserVAE(num_movies=num_movies, latent_dim=latent_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.L1Loss()

    # Precompile model
    model = torch.compile(model)

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for r, mask in train_loader:
            r = r.to(device)          # normalized ratings with zeros where missing
            mask = mask.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda"):
              rec, mu, logvar, z = model(r, mask)
              recon_loss = (F.mse_loss(rec, r, reduction='none') * mask).sum() / mask.sum()
              kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
              loss = recon_loss + 0.0005 * kl_loss  # anneal if needed
            loss.backward()
            opt.step()
            total_loss += loss.item() * r.size(0)
        avg_train = total_loss / len(train_dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} train_mse={avg_train:.6f}")
        if save_path and epoch % 50 == 0:
            torch.save(model.state_dict(), save_path)
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model

# --------------------------
# Encode users
# --------------------------
@torch.no_grad()
def build_user_latents(model, dataset, device=None, batch_size=512):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    latents = []
    for r, mask in loader:
        r = r.to(device)
        z = model.encode(r)
        latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    return latents

# --------------------------
# Recommend users
# --------------------------
def recommend_by_similar_users(user_id, user_latents, dataset,
                               movies_csv="src/data/movies.csv",
                               ratings_csv="src/data/movie_ratings.csv",
                               top_k_users=20, top_n_items=10):
    """
    Given a user_id (actual id from dataset.user_ids), find nearest users and aggregate their ratings
    to recommend top_n_items (excluding items the user already rated).
    """
    # find user index in dataset.user_ids
    if user_id not in dataset.user2idx:
        raise ValueError("User id not found in dataset (maybe pruned).")
    user_idx = dataset.user2idx[user_id]

    # nearest neighbors among users
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(user_latents)
    distances, indices = nn.kneighbors(user_latents[user_idx: user_idx+1], n_neighbors=top_k_users+1)
    neighbor_idxs = indices[0][1:]  # exclude self

    # aggregate neighbors' original (denormalized) ratings
    # we have dataset.ratings = normalized centered by mean/std, need original ratings.
    # reconstruct original neighbor ratings approx: rating ≈ (normalized * std) + mean  (for observed entries)
    neighbor_mask = dataset.mask[neighbor_idxs]           # shape [k, num_movies]
    neighbor_norm = dataset.ratings[neighbor_idxs]        # normalized
    # denormalize
    means = dataset.mean_user[neighbor_idxs]              # [k,1]
    stds  = dataset.std_user[neighbor_idxs]               # [k,1]
    neighbor_ratings = neighbor_norm * stds + means       # [k, num_movies]
    neighbor_ratings = neighbor_ratings * neighbor_mask  # set unrated to zero

    # score each movie by average rating across neighbors who rated it
    # sum ratings and count ratings
    rating_sums = neighbor_ratings.sum(axis=0)
    rating_counts = neighbor_mask.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_rating = np.divide(rating_sums, rating_counts)
        avg_rating[np.isnan(avg_rating)] = 0.0

    # exclude movies the target user already rated
    user_mask = dataset.mask[user_idx]  # 1 for rated movies
    avg_rating[user_mask == 1.0] = -np.inf

    # top N indices
    top_indices = np.argpartition(-avg_rating, range(top_n_items))[:top_n_items]
    top_indices = top_indices[np.argsort(-avg_rating[top_indices])]  # sort by score desc

    # map back to movie IDs and titles
    movie_ids = dataset.movie_ids[top_indices]
    movies_df = pd.read_csv(movies_csv)
    results = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids]
    out = results[['title', 'genres']].copy()
    out.insert(0, 'movieId', movie_ids)
    out['score'] = avg_rating[top_indices]
    return out

def get_user_liked_movies(user_id, dataset, movies_csv="src/data/movies.csv", min_rating=4.0):
    """Return movies rated >= min_rating by a user."""
    if user_id not in dataset.user2idx:
        raise ValueError("User id not found in dataset.")
    user_idx = dataset.user2idx[user_id]

    ratings_norm = dataset.ratings[user_idx]
    mask = dataset.mask[user_idx]
    mean = dataset.mean_user[user_idx]
    std = dataset.std_user[user_idx]

    ratings_real = ratings_norm * std + mean
    ratings_real = ratings_real * mask

    liked_indices = np.where(ratings_real >= min_rating)[0]
    if len(liked_indices) == 0:
        print(f"User {user_id} has no movies rated >= {min_rating}")
        return pd.DataFrame(columns=["movieId", "title", "genres", "rating"])

    movie_ids = dataset.movie_ids[liked_indices]
    movies_df = pd.read_csv(movies_csv)
    results = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids]
    out = results[['title', 'genres']].copy()
    out.insert(0, 'movieId', movie_ids)
    out['rating'] = ratings_real[liked_indices].round(2)
    return out.sort_values('rating', ascending=False).reset_index(drop=True)

def plot_latent_space(latent_path = "src/data/user_latents.npy"):
    plt.rcParams.update({'font.size': 22})

    latents = np.load(latent_path)  # Shape: (n_users, latent_dim)
    print(f"Loaded user latents: {latents.shape}")

    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained by 2 components: {pca.explained_variance_ratio_.sum():.4f}")

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=range(len(latents)),  # color by user index for pattern visibility
        cmap='viridis',
        s=30,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.3
    )

    plt.title('2D PCA Projection of User Latent Embeddings', fontsize=22, pad=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=22)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=22)
    plt.colorbar(scatter, label='User Index (ordered)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig("user_latents_pca_2d.png", dpi=300, bbox_inches='tight')
    print("Plot saved as 'user_latents_pca_2d.png'")

    plt.show()

def compute_metric_latent_knn(ratings_csv, movies_csv, model_path, latents_path):
    ds = UserRatingDataset(
        ratings_csv=ratings_csv,
        movies_csv=movies_csv,
        min_movie_count=10,
        debias_years=False,
        downsample_overrated_years_frac=0.35
    )
    RNG_SEED = 42
    rng = np.random.RandomState(RNG_SEED)
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 124
    num_movies = ds.ratings.shape[1]

    model = UserVAE(num_movies=num_movies, latent_dim=latent_dim).to(device)

    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    full_latents = np.load(latents_path).astype(np.float32)
    full_latents = normalize(full_latents, norm='l2')

    user_pos_info = {}
    MIN_RATING = 4.0
    MIN_POS = 20

    for idx in range(len(ds)):
        user_id = ds.user_ids[idx]
        real_ratings = (ds.ratings[idx] * ds.std_user[idx] + ds.mean_user[idx]) * ds.mask[idx]
        pos_mask = real_ratings >= MIN_RATING
        if pos_mask.sum() < MIN_POS:
            continue
        pos_movie_ids = ds.movie_ids[pos_mask]
        pos_ratings = real_ratings[pos_mask]
        user_pos_info[user_id] = list(zip(pos_movie_ids, pos_ratings))

    eligible_user_ids = list(user_pos_info.keys())

    MAX_USERS = 100000
    eval_user_ids = rng.choice(eligible_user_ids, size=min(MAX_USERS, len(eligible_user_ids)), replace=False)

    TOP_K_NEIGHBORS = 30
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(full_latents)

    results_ndcg = []
    results_recall = []
    results_hit = []

    for user_id in tqdm(eval_user_ids, desc="Final Eval (Full KNN)"):
        user_idx = ds.user2idx[user_id]
        pos_items = user_pos_info[user_id]
        if len(pos_items) < 10:
            continue

        # Random 10 held-out
        pos_array = np.array(pos_items, dtype=object)
        held_out_idx = rng.choice(len(pos_array), size=10, replace=False)
        held_out = pos_array[held_out_idx]
        held_out_movie_ids = held_out[:, 0].astype(int)
        held_out_set = set(held_out_movie_ids)

        try:
            held_out_cols = [ds.movie2idx[mid] for mid in held_out_movie_ids]
        except KeyError:
            continue

        # Partial profile
        full_real = (ds.ratings[user_idx] * ds.std_user[user_idx] + ds.mean_user[user_idx]) * ds.mask[user_idx]
        partial_real = full_real.copy()
        partial_mask = ds.mask[user_idx].copy()
        partial_real[held_out_cols] = 0.0
        partial_mask[held_out_cols] = 0.0

        obs = partial_mask.sum()
        if obs == 0:
            continue
        pmean = (partial_real * partial_mask).sum() / obs
        centered = (partial_real - pmean) * partial_mask
        pstd = np.sqrt((centered**2 * partial_mask).sum() / obs) or 1.0
        partial_norm = centered / pstd

        # Encode
        with torch.no_grad():
            z = model.encode(torch.from_numpy(partial_norm).float().unsqueeze(0).to(device))
            z = normalize(z.cpu().numpy(), norm='l2')

        # KNN
        distances, neigh_idx = knn.kneighbors(z, n_neighbors=TOP_K_NEIGHBORS + 1)
        neighbor_idxs = neigh_idx[0][1:]  # skip self

        # Aggregate
        neigh_real = (ds.ratings[neighbor_idxs] * ds.std_user[neighbor_idxs] + ds.mean_user[neighbor_idxs]) * ds.mask[neighbor_idxs]
        sums = neigh_real.sum(axis=0)
        counts = ds.mask[neighbor_idxs].sum(axis=0)
        scores = np.zeros_like(sums)
        valid = counts > 0
        scores[valid] = sums[valid] / counts[valid]
        scores[partial_mask == 1] = -np.inf

        # Top 30
        top_cols = np.argpartition(-scores, 29)[:30]
        top_cols = top_cols[np.argsort(-scores[top_cols])]
        rec_ids = ds.movie_ids[top_cols]
        rec_scores = scores[top_cols]

        # Relevance
        relevance = np.zeros(len(ds.movie_ids))
        for mid in held_out_movie_ids:
            if mid in ds.movie2idx:
                relevance[ds.movie2_idx[mid]] = 1.0

        # Metrics
        recall = len(held_out_set & set(rec_ids)) / 10.0
        hit = 1.0 if recall > 0 else 0.0

        results_hit.append(hit)

    print(f"Hit Rate@30 : {np.mean(results_hit):.1%}")