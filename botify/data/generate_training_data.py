import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

TRACKS_PATH = Path(__file__).parent / "tracks.json"
SASREC_PATH = Path(__file__).parent / "sasrec_i2i.jsonl"
LIGHTFM_PATH = Path(__file__).parent / "lightfm_i2i.jsonl"
OUTPUT_PATH = Path(__file__).parent / "training_data.csv"
N_USERS = 2500
SESSIONS_PER_USER = 8
TRACKS_PER_SESSION = (8, 20)
TOP_GENRES_N = 40

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


def load_tracks(path):
    tracks = []
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            tracks.append({
                "track": int(t["track"]),
                "artist": t["artist"],
                "genres": set(t.get("genres", [])),
                "mood": t.get("mood", ""),
                "year": parse_year(t.get("year")),
                "artist_fans": parse_float(t.get("artist_fans")),
            })
    return tracks


def load_i2i_mapping(path, key_object="item_id", key_recommendations="recommendations"):
    mapping = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            mapping[int(rec[key_object])] = set(rec[key_recommendations])
    return mapping


def build_lookups(tracks):
    genre_freq = Counter()
    for t in tracks:
        for g in t["genres"]:
            genre_freq[g] += 1
    top_genres = {g for g, _ in genre_freq.most_common(TOP_GENRES_N)}

    all_moods = sorted(set(t["mood"] for t in tracks if t["mood"]))
    all_artists = sorted(set(t["artist"] for t in tracks))

    tracks_by_artist = {}
    for t in tracks:
        tracks_by_artist.setdefault(t["artist"], []).append(t["track"])

    tracks_by_genre = {}
    for t in tracks:
        for g in t["genres"] & top_genres:
            tracks_by_genre.setdefault(g, []).append(t["track"])

    popular = [t["track"] for t in sorted(tracks, key=lambda x: -x["artist_fans"])[:1000]]

    return {
        "top_genres": top_genres,
        "all_moods": all_moods,
        "all_artists": all_artists,
        "tracks_by_artist": tracks_by_artist,
        "tracks_by_genre": tracks_by_genre,
        "popular_tracks": popular,
        "n_tracks": len(tracks),
    }


def generate_user_profile(lookups):
    top_genres = list(lookups["top_genres"])
    all_moods = lookups["all_moods"]
    all_artists = lookups["all_artists"]

    n_genres = random.randint(2, 8)
    preferred_genres = set(random.sample(top_genres, min(n_genres, len(top_genres))))
    n_artists = random.randint(3, 12)
    preferred_artists = set(random.sample(all_artists, min(n_artists, len(all_artists))))
    n_moods = random.randint(1, 4)
    preferred_moods = set(random.sample(all_moods, min(n_moods, len(all_moods))))

    return {
        "preferred_genres": preferred_genres,
        "preferred_artists": preferred_artists,
        "preferred_moods": preferred_moods,
        "genre_weight": random.uniform(0.2, 0.6),
        "artist_weight": random.uniform(0.15, 0.5),
        "mood_weight": random.uniform(0.05, 0.25),
        "base_time": random.uniform(0.08, 0.3),
    }


def compute_listen_time(track, user_profile, prev_artist=None):
    genre_score = len(track["genres"] & user_profile["preferred_genres"]) / max(
        1, len(track["genres"] | user_profile["preferred_genres"])
    )
    artist_score = 1.0 if track["artist"] in user_profile["preferred_artists"] else 0.0
    mood_score = 1.0 if track["mood"] in user_profile["preferred_moods"] else 0.0
    same_artist_bonus = 0.15 if prev_artist and track["artist"] == prev_artist else 0.0

    time = (
        user_profile["base_time"]
        + user_profile["genre_weight"] * genre_score
        + user_profile["artist_weight"] * artist_score
        + user_profile["mood_weight"] * mood_score
        + same_artist_bonus
        + np.random.normal(0, 0.07)
    )
    return max(0.01, min(0.99, time))


def generate_session(user_profile, tracks, lookups, sasrec_map, lightfm_map):
    session_history = []
    session_len = random.randint(*TRACKS_PER_SESSION)

    for _ in range(session_len):
        candidate = pick_candidate(user_profile, tracks, lookups, session_history)
        prev = session_history[-1] if session_history else None
        prev_artist = prev["artist"] if prev else None
        listen_time = compute_listen_time(candidate, user_profile, prev_artist)

        history_slice = session_history[-10:] if session_history else []
        features = extract_features(history_slice, candidate, sasrec_map, lightfm_map)
        features["listen_time"] = round(listen_time, 4)

        candidate_with_time = {**candidate, "listen_time": listen_time}
        session_history.append(candidate_with_time)
        yield features


def pick_candidate(user_profile, tracks, lookups, session_history):
    seen = set(t["track"] for t in session_history)
    roll = random.random()

    pref_artists = list(user_profile["preferred_artists"])
    pref_genres = user_profile["preferred_genres"]

    if roll < 0.45 and pref_artists:
        artist = random.choice(pref_artists)
        artist_tracks = lookups["tracks_by_artist"].get(artist, [])
        unseen = [tid for tid in artist_tracks if tid not in seen]
        if unseen:
            return get_track_by_id(random.choice(unseen), tracks)
    elif roll < 0.72 and pref_genres:
        genre = random.choice(list(pref_genres))
        genre_tracks = lookups["tracks_by_genre"].get(genre, [])
        unseen = [tid for tid in genre_tracks if tid not in seen]
        if unseen:
            return get_track_by_id(random.choice(unseen), tracks)
    elif roll < 0.87:
        popular_tracks = lookups["popular_tracks"]
        unseen = [tid for tid in popular_tracks if tid not in seen]
        if unseen:
            return get_track_by_id(random.choice(unseen), tracks)

    unseen_all = [t for t in tracks if t["track"] not in seen]
    if unseen_all:
        return random.choice(unseen_all)
    return random.choice(tracks)


def get_track_by_id(tid, tracks):
    for t in tracks:
        if t["track"] == tid:
            return t
    return tracks[tid] if tid < len(tracks) else random.choice(tracks)


def extract_features(history, candidate, sasrec_map, lightfm_map):
    if not history:
        return {name: 0.0 for name in FEATURE_NAMES}

    hgen = set()
    hart = set()
    hmoods = []
    hyears = []
    for h in history:
        hgen |= h["genres"]
        hart.add(h["artist"])
        if h["mood"]:
            hmoods.append(h["mood"])
        hyears.append(h["year"])

    cgen = candidate["genres"]
    union = hgen | cgen
    genre_jaccard = len(hgen & cgen) / max(1, len(union))

    same_artist = 1.0 if candidate["artist"] in hart else 0.0
    same_mood = 1.0 if candidate["mood"] and candidate["mood"] in hmoods else 0.0

    avg_year = np.mean(hyears) if hyears else 2000
    year_diff_norm = min(abs(candidate["year"] - avg_year) / 50.0, 1.0)

    log_fans = min(math.log10(candidate["artist_fans"] + 1) / 6.0, 1.0)

    in_sasrec = 0.0
    in_lightfm = 0.0
    for h in history:
        if candidate["track"] in sasrec_map.get(h["track"], set()):
            in_sasrec = 1.0
        if candidate["track"] in lightfm_map.get(h["track"], set()):
            in_lightfm = 1.0

    return {
        "genre_jaccard": round(genre_jaccard, 4),
        "same_artist": same_artist,
        "same_mood": same_mood,
        "year_diff_norm": round(year_diff_norm, 4),
        "log_artist_fans": round(log_fans, 4),
        "in_sasrec_i2i": in_sasrec,
        "in_lightfm_i2i": in_lightfm,
    }


def main():
    print("Loading tracks...")
    tracks = load_tracks(TRACKS_PATH)
    print(f"Loaded {len(tracks)} tracks")

    print("Loading SasRec I2I...")
    sasrec_map = load_i2i_mapping(SASREC_PATH)
    print(f"  {len(sasrec_map)} items")

    print("Loading LightFM I2I...")
    lightfm_map = load_i2i_mapping(LIGHTFM_PATH)
    print(f"  {len(lightfm_map)} items")

    lookups = build_lookups(tracks)

    all_rows = []
    for u in range(N_USERS):
        user_profile = generate_user_profile(lookups)
        for s in range(SESSIONS_PER_USER):
            for row in generate_session(user_profile, tracks, lookups, sasrec_map, lightfm_map):
                all_rows.append(row)
        if (u + 1) % 500 == 0:
            print(f"  Users: {u + 1}/{N_USERS}")

    header = ",".join(FEATURE_NAMES + ["listen_time"])
    lines = [header]
    for row in all_rows:
        feat_str = ",".join(str(row[f]) for f in FEATURE_NAMES)
        lines.append(f"{feat_str},{row['listen_time']}")

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"\nSaved: {OUTPUT_PATH} ({len(all_rows)} rows, {OUTPUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
