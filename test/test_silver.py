from pathlib import Path
import pandas as pd

def test_silver_exists():
    for f in ["cleaned_ratings.csv","users.csv","movies.csv","stats.json"]:
        assert (Path("data/silver")/f).exists(), f"Missing {f}"

def test_basic_columns():
    df = pd.read_csv("data/silver/cleaned_ratings.csv")
    for c in ["userId","movieId","rating","title","genres"]:
        assert c in df.columns
    assert df["rating"].between(0.5, 5).all()
