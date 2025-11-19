from datasketch import MinHash, MinHashLSH
import pandas as pd

import numpy as np
from tqdm import tqdm

# Find movieid based on partial title
def find_movie_by_title(partial_title: str, df_movies: pd.DataFrame) -> pd.DataFrame:
    mask = df_movies['title'].str.contains(partial_title, case=False, na=False)
    print(df_movies[mask][['title', 'genres']])

def create_movie_tokens_lsh(df_movies, movie_genres, movie_users_pos, movie_users_neg):
    movie_tokens_lsh = {}
    for mid in df_movies.index:
        tokens = set()
        # genre tokens
        for g in movie_genres.loc[mid]:
            tokens.add(f"g:{g}")
        # positive user tokens
        for u in movie_users_pos.loc[mid]:
            tokens.add(f"u+:{u}")
        # negative user tokens
        for u in movie_users_neg.loc[mid]:
            tokens.add(f"u-:{u}")
        movie_tokens_lsh[mid] = tokens 
    return movie_tokens_lsh

def minhash_signatures(movie_tokens_lsh, num_perm=128):
    minhashes = {}
    for mid, tokens in movie_tokens_lsh.items():
        m = MinHash(num_perm=num_perm)
        for t in tokens:
            m.update(t.encode("utf-8"))
        minhashes[mid] = m
    return minhashes

def minhash_lsh(minhashes, lsh_threshold=0.3, num_perm=128):
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    for mid, m in minhashes.items():
        lsh.insert(str(mid), m)
    return lsh

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def double_jaccard(ap: set, bp: set, am: set, bm: set) -> float:
    if not ap and not bp and not am and not bm:
        return 0.0
    inter = len(ap & bp) + len(am & bm)
    union = len(ap | bp) + len(am | bm)
    return inter / union if union else 0.0

def combined_similarity(
    genres_a: set, genres_b: set,
    users_pos_a: set, users_pos_b: set,
    users_neg_a: set, users_neg_b: set,
    LAMBDA_GENRE=0.4
) -> dict:
    """
    Compute:
      J_g      : genre Jaccard
      J_pos    : Jaccard on liked users
      J_neg    : Jaccard on disliked users
      S_user   : combined user similarity in [0,1]
      S_final  : final similarity mixing genre and user channels
    """
    J_g = jaccard(genres_a, genres_b)
    J_rating = double_jaccard(users_pos_a, users_pos_b,users_neg_a, users_neg_b)

    # Normalize to [0,1]
    S_user = J_rating

    # Final similarity: mix genres and user-based
    S_final = LAMBDA_GENRE * J_g + (1.0 - LAMBDA_GENRE) * S_user

    return {
        "J_genre": J_g,
        "J_rating": J_rating,
        "S_user": S_user,
        "similarity": S_final,
    }


def build_profile_sets(
    movie_ids,
    df_movies,
    movie_genres,
    movie_users_pos,
    movie_users_neg,
    use_genre: bool = True,
    use_user_pos: bool = True,
    use_user_neg: bool = True,
    
):
    """
    Build profile sets (genres, positive users, negative users)
    by unioning information from a list of movies.
    """
    profile_genres = set()
    profile_users_pos = set()
    profile_users_neg = set()

    for mid in movie_ids:
        if mid not in df_movies.index:
            continue
        if use_genre:
            profile_genres |= movie_genres.loc[mid]
        if use_user_pos:
            profile_users_pos |= movie_users_pos.loc[mid]
        if use_user_neg:
            profile_users_neg |= movie_users_neg.loc[mid]

    return profile_genres, profile_users_pos, profile_users_neg

def build_lsh_tokens_from_sets(
    genres: set,
    users_pos: set,
    users_neg: set,
) -> set:
    """
    Convert genre / user sets into the token format used by LSH.
    """
    tokens = set()
    for g in genres:
        tokens.add(f"g:{g}")
    for u in users_pos:
        tokens.add(f"u+:{u}")
    for u in users_neg:
        tokens.add(f"u-:{u}")
    return tokens

def profile_minhash_from_sets(
    genres: set,
    users_pos: set,
    users_neg: set,
    num_perm: int = 128,
) -> MinHash:
    """
    Build a MinHash signature for a profile defined by
    (genres, users_pos, users_neg) sets.
    """
    tokens = build_lsh_tokens_from_sets(genres, users_pos, users_neg)
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m

def similar_movies_for_profile(
    movie_ids,
    lsh_index,
    movies_df: pd.DataFrame,
    movie_genres: pd.DataFrame,
    movie_users_pos: pd.DataFrame,
    movie_users_neg: pd.DataFrame,
    top_k: int = 15,
    use_genre: bool = True,
    use_user_pos: bool = True,
    use_user_neg: bool = True,
):
    """
    Recommend movies for a 'virtual user' whose taste is defined
    by a list of liked movie_ids, using a single LSH index and
    a combined (genre + user likes/dislikes) similarity.
    """
    # 1) Build profile sets (genres, pos users, neg users)
    g_prof, up_prof, un_prof = build_profile_sets(
        movie_ids,
        movies_df,
        movie_genres,
        movie_users_pos,
        movie_users_neg,
        use_genre=use_genre,
        use_user_pos=use_user_pos,
        use_user_neg=use_user_neg,
    )

    if not g_prof and not up_prof and not un_prof:
        raise ValueError("Profile is empty; check input movie_ids or channel flags.")

    # 2) Build profile MinHash and query LSH
    q_sig = profile_minhash_from_sets(g_prof, up_prof, un_prof)
    candidates = lsh_index.query(q_sig)

    liked_set = set(movie_ids)
    cand_ids = [int(cid) for cid in candidates if int(cid) not in liked_set]

    rows = []
    for cid in cand_ids:
        g_b = movie_genres.loc[cid]
        up_b = movie_users_pos.loc[cid]
        un_b = movie_users_neg.loc[cid]

        sim = combined_similarity(g_prof, g_b, up_prof, up_b, un_prof, un_b)

        title = movies_df.loc[cid, "title"]
        year = movies_df.loc[cid, "year"]

        rows.append({
            "movieId": cid,
            "title": title,
            "year": int(year) if pd.notna(year) else None,
            **sim,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "movieId", "title", "year",
            "similarity", "J_genre", "J_pos", "J_neg", "S_user",
        ])

    df = pd.DataFrame(rows).sort_values("similarity", ascending=False)
    return df.head(top_k)



def validate(ratings_eval, 
             lsh_index,
             movies_df: pd.DataFrame,
             movie_genres: pd.DataFrame,
             movie_users_pos: pd.DataFrame,
             movie_users_neg: pd.DataFrame,
             RNG_SEED = 42,
             TOP_K_EVAL = 30,
             MASK_USER_MOVIES = 10,
             MIN_RATING_SCORE = 4.0,
             MIN_AMOUNT_MOVIE_OVER_THRESHOLD = 20,
             MAX_USERS_TO_EVAL = 100):
    
    rng = np.random.RandomState(RNG_SEED)
    
    # Keep only "positive" ratings (>= MIN_RATING_SCORE)
    df_pos_eval = ratings_eval[ratings_eval["rating"] >= MIN_RATING_SCORE]

    # Collect positive movies per user
    user_pos_movies = (
        df_pos_eval.groupby("userId")["movieId"]
        .apply(lambda s: sorted(set(s.astype(int))))
    )

    # Users with at least MIN_AMOUNT_MOVIE_OVER_THRESHOLD positive movies
    eligible_users = [
        user_id
        for user_id, movies in user_pos_movies.items()
        if len(movies) >= MIN_AMOUNT_MOVIE_OVER_THRESHOLD
    ]

    print(f"Eligible users (>= {MIN_AMOUNT_MOVIE_OVER_THRESHOLD} positive movies): {len(eligible_users)}")

    if not eligible_users:
        raise ValueError("No eligible users in the specified range.")

    # Sample subset of users for evaluation
    n_users_target = min(MAX_USERS_TO_EVAL, len(eligible_users))
    user_subset = rng.choice(eligible_users, size=n_users_target, replace=False)

    per_user_results = []

    for user_id in tqdm(user_subset):
        movies = np.array(user_pos_movies[user_id])

        # Shuffle to randomize which movies are masked
        rng.shuffle(movies)

        # Mask MASK_USER_MOVIES movies, input the rest
        if len(movies) <= MASK_USER_MOVIES:
            continue  # safety check; shouldn't happen given the threshold

        masked_movies = set(movies[:MASK_USER_MOVIES])
        input_movies = movies[MASK_USER_MOVIES:].tolist()

        if len(input_movies) == 0:
            continue  # nothing to build a profile from

        try:
            recs = similar_movies_for_profile(
                input_movies,
                lsh_index=lsh_index,
                movies_df=movies_df,
                movie_genres=movie_genres,
                movie_users_pos=movie_users_pos,
                movie_users_neg=movie_users_neg,                
                top_k=TOP_K_EVAL,
                use_genre=True,
                use_user_pos=True,
                use_user_neg=True,
            )
        except ValueError:
            continue

        if recs.empty:
            user_hits = 0
        else:
            rec_ids = recs["movieId"].head(TOP_K_EVAL).tolist()
            # Count how many of the masked movies are in the top-K recommendations
            user_hits = sum(1 for m in masked_movies if m in rec_ids)

        per_user_results.append({
            "userId": user_id,
            "hits": user_hits,
            "n_masked": len(masked_movies),  # should be MASK_USER_MOVIES
            "hit_rate_user": user_hits / len(masked_movies) if masked_movies else 0,
        })

    # Aggregate results
    df_eval = pd.DataFrame(per_user_results)

    avg_hits = df_eval["hits"].mean() if not df_eval.empty else float("nan")
    avg_hit_rate = df_eval["hit_rate_user"].mean() if not df_eval.empty else float("nan")

    print(f"Users requested              : {n_users_target}")
    print(f"Users evaluated              : {len(df_eval)}")
    print(f"Average hits per user (0–{MASK_USER_MOVIES}) : {avg_hits:.3f}")
    print(f"Average per-user hit rate    : {avg_hit_rate:.3f}")