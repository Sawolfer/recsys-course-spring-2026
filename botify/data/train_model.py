import json
import pickle
import re
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

DATA_DIR = Path(__file__).parent
TRAINING_DATA = DATA_DIR / "training_data.csv"
TRACKS_PATH = DATA_DIR / "tracks.json"
MODEL_PATH = DATA_DIR / "ml_model.joblib"
FEATURES_PATH = DATA_DIR / "track_features.pkl"

FEATURE_NAMES = [
    "genre_jaccard",
    "same_artist",
    "same_mood",
    "year_diff_norm",
    "log_artist_fans",
    "in_sasrec_i2i",
    "in_lightfm_i2i",
]


def parse_year(val):
    if val is None:
        return 1990
    match = re.search(r"\d{4}", str(val))
    if match:
        return int(match.group(0))
    return 1990


def parse_float(val, default=10.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_track_features(tracks_path):
    with open(tracks_path) as f:
        tracks = [json.loads(line) for line in f]

    track_features = {}
    for t in tracks:
        track_id = int(t["track"])
        genres = set(t.get("genres", []))
        mood = t.get("mood", "")
        year = parse_year(t.get("year"))
        artist = t["artist"]
        artist_fans = parse_float(t.get("artist_fans"))

        track_features[track_id] = {
            "genres": genres,
            "mood": mood,
            "year": year,
            "artist": artist,
            "artist_fans": artist_fans,
        }

    print(f"Built features for {len(track_features)} tracks")
    return track_features


def main():
    import pandas as pd

    print("Loading training data...")
    df = pd.read_csv(TRAINING_DATA)
    print(f"Loaded {len(df)} training examples")

    X = df[FEATURE_NAMES].values
    y = df["listen_time"].values

    print(f"Feature matrix: {X.shape}")
    print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    print("Training...")
    model.fit(X, y)

    train_score = model.score(X, y)
    print(f"Train R²: {train_score:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=3, scoring="r2", n_jobs=-1)
    print(f"CV R² (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: -x[1],
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name}: {imp:.4f}")

    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH, protocol=4)
    print(f"Model saved ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")

    print(f"\nBuilding track features...")
    track_features = build_track_features(TRACKS_PATH)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(track_features, f, protocol=4)
    print(f"Track features saved ({FEATURES_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
