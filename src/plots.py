import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.stats import binned_statistic

def hello():
    print("Hello, world!")


def users_per_genre(df_ratings, df_movies, K=10, random_state=42):
    # Ensure merge is unambiguous
    df_movies_reset = df_movies.reset_index(drop=True)

    # Expand multi-genre rows; count watches per (user, genre)
    merged = df_ratings.merge(df_movies_reset, on="movieId").explode("genres")

    ugc = (
        merged.groupby(["userId", "genres"])["movieId"]
        .count()
        .unstack(fill_value=0)
    )
    
    row_sums = ugc.sum(axis=1).replace(0, np.nan)
    ugc_norm = ugc.div(row_sums, axis=0).fillna(0.0)


    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(ugc_norm.values)

    ugc_norm_clusters = ugc_norm.copy()
    ugc_norm_clusters["cluster"] = labels

    # Cluster sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    print("Cluster sizes:\n", cluster_sizes.to_string())

    centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=ugc_norm.columns,
        index=[f"Cluster {i}" for i in range(K)]
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        centroids,
        cmap="viridis",
        cbar_kws={"label": "Avg. watch share per genre"},
        annot=False
    )
    plt.title("User clusters by watched-genre distribution")
    plt.xlabel("Genre")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()


def popularity_rating(df_ratings, df_movies):
    movie_stats = (
    df_ratings
    .groupby("movieId")["rating"]
    .agg(["mean", "count", "std"])
    .rename(columns={"mean": "avg_rating", "count": "num_ratings"})
    )

    # Merge titles for readability
    movie_stats = movie_stats.join(df_movies.set_index("movieId")["title"])

    plt.figure(figsize=(10, 6))
    plt.scatter(
        movie_stats["num_ratings"],
        movie_stats["avg_rating"],
        alpha=0.4,
        s=15
    )

    plt.xscale("log")
    plt.xlabel("Number of Ratings (log scale)")
    plt.ylabel("Average Rating")
    plt.title("Movie Popularity vs Average Rating")

    bins = np.logspace(0, np.log10(movie_stats["num_ratings"].max()), 40)
    bin_means, _, _ = binned_statistic(
        movie_stats["num_ratings"], movie_stats["avg_rating"], statistic="mean", bins=bins
    )
    plt.plot(bins[:-1], bin_means, color="red", linewidth=2, label="Average rating")
    plt.legend()
    plt.show()


def movies_per_year(movies_df):

    movies_df = movies_df.dropna(subset=['year'])

    yearly_movie_count = movies_df.groupby('year').size().reset_index(name='movie_count')

    min_year = yearly_movie_count['year'].min()
    max_year = yearly_movie_count['year'].max()
    full_years = np.arange(min_year, max_year + 1)

    yearly_movie_count = yearly_movie_count.set_index('year').reindex(full_years, fill_value=0).reset_index()
    yearly_movie_count.columns = ['year', 'movie_count']

    plt.figure(figsize=(12, 6))
    plt.bar(yearly_movie_count['year'], yearly_movie_count['movie_count'], 
        edgecolor='black', alpha=0.7, width=0.8)

    xticks = np.arange(1880, max_year + 1, 20)
    plt.xticks(xticks, rotation=45)

    plt.title('Number of Movies by Release Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def ratings_per_year(movies_df, ratings_df):
    movies_df = movies_df.reset_index(drop=True)
    ratings_with_year = ratings_df.merge(movies_df[['movieId', 'year']], on='movieId', how='left')

    ratings_with_year = ratings_with_year.dropna(subset=['year'])

    yearly_ratings_count = ratings_with_year.groupby('year').size().reset_index(name='total_ratings')
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_ratings_count['year'], yearly_ratings_count['total_ratings'], 
            edgecolor='black', alpha=0.7)
    plt.title('Total Number of Ratings by Movie Release Year')
    plt.xlabel('Year')
    plt.ylabel('Total Ratings')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def ratings_per_movie(ratings_df):
    ratings_per_movie = ratings_df.groupby('movieId').size().reset_index(name='rating_count')

    ratings_sorted = ratings_per_movie.sort_values('rating_count', ascending=False).reset_index(drop=True)
    ratings_sorted['rank'] = ratings_sorted.index + 1

    print("Top 10 movies by rating count:\n", ratings_sorted.head(10)[['movieId', 'rating_count']])

    plt.figure(figsize=(12, 6))
    x = ratings_sorted['rank']
    y = ratings_sorted['rating_count']

    plt.plot(x, y, marker='o', markersize=2, linewidth=1.5, alpha=0.8, color='blue')

    plt.yscale('log')
    plt.title('Number of ratings per movie sorted (loglog)')
    plt.xlabel('Movie Rank (1 = Most Rated)')
    plt.ylabel('Number of Ratings (Log Scale)')
    plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()
