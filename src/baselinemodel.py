import numpy as np
import pandas as pd
from tqdm import tqdm

def baseline_model_old(ratings_df, 
                   threshold=5, 
                   min_ratings_per_user=20, 
                   top_k_recs=30, 
                   max_users_to_eval=100, 
                   rng_seed=42):

    rng = np.random.RandomState(rng_seed)
    
    # Compute global top liked movies
    liked = ratings_df[ratings_df['rating'] == threshold]
    top_movies = liked['movieId'].value_counts().index.tolist()
    
    # Find qualifying users with enough high ratings
    high_ratings = ratings_df[ratings_df['rating'] >= 4]
    user_high_rating_counts = high_ratings.groupby('userId').size()
    qualifying_users = user_high_rating_counts[user_high_rating_counts >= min_ratings_per_user].index.tolist()
    
    def naive_person_accuracy(user_id):
        user_ratings = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4)]
        if len(user_ratings) < 5:
            return 0.0
        
        sample_df = user_ratings.sample(n=10, random_state=rng_seed)
        rest_df = user_ratings.drop(index=sample_df.index)
        
        rest_movie_ids = set(rest_df['movieId'].values)
        top_movies_new = [m for m in top_movies if m not in rest_movie_ids][:top_k_recs]
       
        common_values = set(sample_df['movieId'].values).intersection(set(top_movies_new))
        overlap_count = len(common_values)
        
        return overlap_count 
    
    # Select users
    selected_users = rng.choice(qualifying_users, size=min(max_users_to_eval, len(qualifying_users)), replace=False)
    accuracies = []
    for user_id in tqdm(selected_users, desc="Evaluating users"):
        accuracy = naive_person_accuracy(user_id)
        accuracies.append(accuracy)
    
    average_accuracy = np.mean(accuracies)
    return average_accuracy, len(selected_users)

def baseline_model(
    ratings_df: pd.DataFrame,
    like_threshold: float = 4.0,
    min_user_likes: int = 20,
    holdout_per_user: int = 10,
    top_k: int = 30,
    users_to_eval: int | None = 100,
    rng_seed: int = 42,
):
    """
    Baseline hit-rate model based on global frequency of positive ratings (>=4).

    Steps:
      1) Rank all movies by count of ratings >= like_threshold (descending).
      2) Keep only users with >= min_user_likes movies rated >= like_threshold.
      3) Optionally sample up to `users_to_eval` of those users for efficiency.
      4) For each selected user:
         - Randomly hold out `holdout_per_user` of their positive movies.
         - Treat all remaining movies (any rating) as 'seen' and remove them from the top list,
           preserving top_k length by filling with lower-ranked movies.
         - Compute hit rate = (# of held-out positives in top_k) / holdout_per_user.
      5) Return the average hit rate across evaluated users.
    """
    rng = np.random.RandomState(rng_seed)

    # 1) Compute global ranking by count of positive ratings
    pos_counts = (
        ratings_df.loc[ratings_df["rating"] >= like_threshold, "movieId"]
        .value_counts()
        .index
        .tolist()
    )

    # 2) Identify qualifying users
    user_pos_counts = (
        ratings_df.loc[ratings_df["rating"] >= like_threshold]
        .groupby("userId")
        .size()
    )
    qualifying_users = user_pos_counts[user_pos_counts >= min_user_likes].index.to_numpy()

    if len(qualifying_users) == 0:
        return 0.0

    # 3) Sample users if requested
    if users_to_eval is not None and users_to_eval < len(qualifying_users):
        selected_users = rng.choice(qualifying_users, size=users_to_eval, replace=False)
    else:
        print("Evaluating all qualifying users:", len(qualifying_users))
        selected_users = qualifying_users

    # 4) Compute hit rate
    hit_rates = []

    for user_id in tqdm(selected_users, desc="Evaluating users"):
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        user_pos = user_ratings[user_ratings["rating"] >= like_threshold]["movieId"].values

        # Randomly hold out positive movies
        holdout = rng.choice(user_pos, size=holdout_per_user, replace=False)
        holdout_set = set(holdout)
        seen_set = set(user_ratings["movieId"].values) - holdout_set

        # Build top-K list excluding seen movies, maintaining list length
        recs = []
        for m in pos_counts:
            if m in seen_set:
                continue
            recs.append(m)
            if len(recs) == top_k:
                break

        hits = len(holdout_set.intersection(recs))
        hit_rates.append(hits / holdout_per_user)

    return float(np.mean(hit_rates)) if hit_rates else 0.0