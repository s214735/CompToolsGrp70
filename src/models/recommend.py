# src/models/recommend.py
from pathlib import Path
import torch
import pandas as pd
import numpy as np

from src.models.ncf_model import NeuralCF


def evaluate_leave_one_out(user_ids, test_path="data/gold/test.parquet",
                           ckpt_path="models/ncf.pt", n_samples_per_user=1,
                           random_state=None):
    """
    Randomly removes one (or more) movies per user that they have rated, 
    and predicts the missing rating using a pseudo-user embedding.

    Args:
        user_ids (int | list[int]): One or more user_idx values from the dataset.
        test_path (str): Path to test parquet file.
        ckpt_path (str): Path to trained model checkpoint.
        n_samples_per_user (int): How many rated movies to remove/predict per user.
        random_state (int): Optional random seed for reproducibility.

    Returns:
        pd.DataFrame: [user_idx, movie_idx, true_rating, pred_rating, abs_error]
    """
    if isinstance(user_ids, int):
        user_ids = [user_ids]

    rng = np.random.default_rng(random_state)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = NeuralCF(
        n_users=ckpt["n_users"],
        n_items=ckpt["n_items"],
        emb_dim=ckpt["emb_dim"],
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load test data
    test_df = pd.read_parquet(test_path)
    results = []

    for uid in user_ids:
        user_data = test_df[test_df["user_idx"] == uid]
        if len(user_data) < n_samples_per_user + 1:
            print(f"⚠️  Skipping user {uid} (not enough ratings)")
            continue

        sample_rows = user_data.sample(n=n_samples_per_user, random_state=rng.integers(1e9))

        for _, row in sample_rows.iterrows():
            true_movie, true_rating = int(row.movie_idx), float(row.rating)
            known = user_data[user_data["movie_idx"] != true_movie]
            if known.empty:
                continue

            # Build pseudo-user embedding from remaining known movies
            km_idx = torch.tensor(known["movie_idx"].values, dtype=torch.long, device=device)
            km_wts = torch.tensor(known["rating"].values, dtype=torch.float32, device=device).unsqueeze(1)
            item_embs = model.item_emb(km_idx)
            pseudo_user = (item_embs * km_wts).sum(0) / (km_wts.sum() + 1e-8)

            # Predict for the held-out movie
            m_idx = torch.tensor([true_movie], dtype=torch.long, device=device)
            held_item = model.item_emb(m_idx)
            x = torch.cat([pseudo_user.unsqueeze(0), held_item], dim=1)
            pred = model.mlp(x).squeeze(-1).item()

            results.append((uid, true_movie, true_rating, pred, abs(true_rating - pred)))

    res_df = pd.DataFrame(results, columns=["user_idx", "movie_idx", "true_rating", "pred_rating", "abs_error"])

    if not res_df.empty:
        print(f"\nAverage absolute error: {res_df['abs_error'].mean():.3f}")
        print(res_df.head(10))

    return res_df

SILVER = Path("data/silver")
MODELS = Path("models")

def load_model(ckpt_path=MODELS / "ncf.pt", device=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_users = ckpt["n_users"]
    n_items = ckpt["n_items"]
    model = NeuralCF(
        n_users=n_users,
        n_items=n_items,
        emb_dim=ckpt["emb_dim"],
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if device:
        model.to(device)
    return model, n_users, n_items, ckpt

def _id_maps():
    users = pd.read_csv(SILVER / "user_id_map.csv")   # columns: userId, user_idx
    movies = pd.read_csv(SILVER / "movie_id_map.csv") # columns: movieId, movie_idx, title, genres
    return users, movies

def recommend_for_existing_user(user_id: int, top_n=10, device=None):
    users_map, movies_map = _id_maps()
    # map to user_idx
    row = users_map[users_map["userId"] == user_id]
    if row.empty:
        raise ValueError(f"userId {user_id} not found (treat as new user).")

    user_idx = int(row["user_idx"].iloc[0])

    model, n_users, n_items, _ = load_model(device=device)

    # predict for all items
    with torch.no_grad():
        items = torch.arange(n_items, dtype=torch.long, device=device or "cpu")
        users = torch.full_like(items, fill_value=user_idx)
        preds = model(users, items).cpu().numpy()

    # mask already seen: load silver interactions for this user
    df = pd.read_parquet("data/gold/train.parquet")  # quick source of interactions
    seen_idx = set(df.loc[df["user_idx"] == user_idx, "movie_idx"].astype(int).tolist())

    # rank
    candidates = [(i, p) for i, p in enumerate(preds) if i not in seen_idx]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:top_n]

    # map back to movieId/title
    inv = movies_map.set_index("movie_idx")
    results = []
    for midx, score in top:
        m = inv.loc[midx]
        results.append({
            "movieId": int(m["movieId"]),
            "title": m.get("title", None),
            "score": float(score)
        })
    return results

def recommend_for_new_user(rated_pairs, top_n=10, device=None):
    """
    rated_pairs: list of (movieId, rating) from a brand new user
    """
    _, movies_map = _id_maps()
    model, n_users, n_items, _ = load_model(device=device)
    device = device or "cpu"

    # map movieIds to movie_idx
    m2idx = movies_map.set_index("movieId")["movie_idx"].to_dict()
    rated = [(m2idx[m], r) for (m, r) in rated_pairs if m in m2idx]

    if not rated:
        raise ValueError("None of the provided movieIds exist in the map.")

    # build pseudo-user embedding by rating-weighted average of item embeddings
    with torch.no_grad():
        items = torch.tensor([midx for (midx, _) in rated], dtype=torch.long, device=device)
        weights = torch.tensor([r for (_, r) in rated], dtype=torch.float32, device=device)
        item_embs = model.item_emb(items)  # (k, emb)
        weights = (weights / (weights.sum() + 1e-8)).unsqueeze(1)
        pseudo_user = (item_embs * weights).sum(dim=0, keepdim=True)  # (1, emb)

        # score all items
        all_items = torch.arange(n_items, dtype=torch.long, device=device)
        all_item_embs = model.item_emb(all_items)
        # pass through MLP by concatenating pseudo-user with each item embedding
        # reuse internal MLP: create batched features
        feats = torch.cat([pseudo_user.repeat(n_items, 1), all_item_embs], dim=1)
        scores = model.mlp(feats).squeeze(-1).cpu().numpy()

    # mask the ones the new user already rated
    rated_idx = set([midx for (midx, _) in rated])
    candidates = [(i, s) for i, s in enumerate(scores) if i not in rated_idx]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:top_n]

    inv = movies_map.set_index("movie_idx")
    results = []
    for midx, score in top:
        row = inv.loc[midx]
        results.append({
            "movieId": int(row["movieId"]),
            "title": row.get("title", None),
            "score": float(score),
        })
    return results

if __name__ == "__main__":
    # tiny demo (adjust ids to your data)
    # print(recommend_for_existing_user(user_id=1, top_n=10))
    # print(recommend_for_new_user([(1, 5.0), (296, 4.5), (593, 4.0)], top_n=10))
    pass
