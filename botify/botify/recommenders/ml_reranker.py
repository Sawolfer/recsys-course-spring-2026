import json
import math
import pickle
import random
from collections import Counter, defaultdict

import joblib

from .recommender import Recommender


FEATURE_NAMES = [
    "genre_jaccard",
    "same_artist",
    "same_mood",
    "year_diff_norm",
    "log_artist_fans",
    "in_sasrec_i2i",
    "in_lightfm_i2i",
]


class MLReranker(Recommender):
    def __init__(
        self,
        model_path,
        track_features_path,
        listen_history_redis,
        sasrec_i2i_redis,
        lightfm_i2i_redis,
        fallback_recommender,
        artist_index=None,
        genre_index=None,
        popular_tracks=None,
    ):
        self.model = joblib.load(model_path)
        with open(track_features_path, "rb") as f:
            self.track_features = pickle.load(f)
        self.listen_history_redis = listen_history_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.lightfm_i2i_redis = lightfm_i2i_redis
        self.fallback = fallback_recommender

        self.artist_index = artist_index or self._build_artist_index()
        self.genre_index = genre_index or self._build_genre_index()
        self.popular_tracks = popular_tracks or self._build_popular_tracks()
        self.all_track_ids = list(self.track_features.keys())

    def _build_artist_index(self):
        idx = defaultdict(list)
        for tid, feats in self.track_features.items():
            idx[feats["artist"]].append(tid)
        return dict(idx)

    def _build_genre_index(self):
        idx = defaultdict(list)
        for tid, feats in self.track_features.items():
            for g in feats["genres"]:
                idx[g].append(tid)
        return dict(idx)

    def _build_popular_tracks(self):
        tracks = [(tid, feats["artist_fans"]) for tid, feats in self.track_features.items()]
        tracks.sort(key=lambda x: -x[1])
        return [tid for tid, _ in tracks[:1000]]

    def _build_i2i_support(self, history):
        sasrec_set = set()
        lightfm_set = set()
        for track_id, _ in history:
            data = self.sasrec_i2i_redis.get(track_id)
            if data:
                try:
                    for r in pickle.loads(data):
                        sasrec_set.add(int(r))
                except Exception:
                    pass
            data = self.lightfm_i2i_redis.get(track_id)
            if data:
                try:
                    for r in pickle.loads(data):
                        lightfm_set.add(int(r))
                except Exception:
                    pass
        return sasrec_set, lightfm_set

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            return self._recommend_next_impl(user, prev_track, prev_track_time)
        except Exception:
            try:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)
            except Exception:
                return prev_track

    def _recommend_next_impl(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        user_context = self._build_user_context(history)
        sasrec_support, lightfm_support = self._build_i2i_support(history)

        candidates = self._get_candidates(user_context)
        if not candidates:
            return self._safe_fallback(user, prev_track, prev_track_time)

        seen = set(track for track, _ in history)
        best_score = -1.0
        best_track = None

        feature_rows = []
        candidate_list = []

        for track_id in candidates:
            if track_id in seen:
                continue
            feats = self._compute_features(user_context, track_id, sasrec_support, lightfm_support)
            if feats is None:
                continue
            feature_rows.append([feats[name] for name in FEATURE_NAMES])
            candidate_list.append(track_id)

            if len(feature_rows) >= 200:
                break

        if feature_rows:
            scores = self.model.predict(feature_rows)
            for i, score in enumerate(scores):
                if score is not None and score > best_score:
                    best_score = score
                    best_track = candidate_list[i]

        if best_track is not None:
            return best_track
        return self._safe_fallback(user, prev_track, prev_track_time)

    def _safe_fallback(self, user, prev_track, prev_track_time):
        try:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        except Exception:
            return prev_track

    def _get_candidates(self, user_context):
        candidates = set()

        for artist in user_context["artists"]:
            artist_tracks = self.artist_index.get(artist, [])
            sample = artist_tracks if len(artist_tracks) <= 8 else random.sample(artist_tracks, 8)
            candidates.update(sample)

        for genre in list(user_context["genres"])[:5]:
            genre_tracks = self.genre_index.get(genre, [])
            sample = genre_tracks if len(genre_tracks) <= 5 else random.sample(genre_tracks, 5)
            candidates.update(sample)

        popular_sample = random.sample(
            self.popular_tracks, min(30, len(self.popular_tracks))
        )
        candidates.update(popular_sample)

        if len(candidates) < 20:
            extra = random.sample(
                self.all_track_ids, min(50, len(self.all_track_ids))
            )
            candidates.update(extra)

        return list(candidates)

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _build_user_context(self, history):
        if not history:
            return {
                "genres": set(),
                "artists": set(),
                "moods": [],
                "years": [],
                "last_track": None,
                "last_mood": "",
                "last_artist": "",
            }

        genres = set()
        artists = set()
        moods = []
        years = []

        for track_id, _ in history:
            feats = self.track_features.get(track_id)
            if feats is None:
                continue
            genres |= feats["genres"]
            artists.add(feats["artist"])
            if feats["mood"]:
                moods.append(feats["mood"])
            years.append(feats["year"])

        last_id = history[-1][0]
        last_feats = self.track_features.get(last_id, {})

        return {
            "genres": genres,
            "artists": artists,
            "moods": moods,
            "years": years,
            "last_track": last_id,
            "last_mood": last_feats.get("mood", ""),
            "last_artist": last_feats.get("artist", ""),
        }

    def _compute_features(self, user_context, candidate_id, sasrec_support, lightfm_support):
        candidate = self.track_features.get(candidate_id)
        if candidate is None:
            return None

        ctx_genres = user_context["genres"]

        cand_genres = candidate["genres"]
        genre_union = ctx_genres | cand_genres
        genre_jaccard = (
            len(ctx_genres & cand_genres) / len(genre_union) if genre_union else 0.0
        )

        same_artist = 1.0 if candidate["artist"] in user_context["artists"] else 0.0
        same_mood = 1.0 if candidate["mood"] and candidate["mood"] in user_context["moods"] else 0.0

        avg_year = (
            sum(user_context["years"]) / len(user_context["years"])
            if user_context["years"]
            else 2000
        )
        year_diff_norm = min(abs(candidate["year"] - avg_year) / 50.0, 1.0)

        log_fans = math.log10(candidate["artist_fans"] + 1) / 6.0
        log_fans = min(log_fans, 1.0)

        in_sasrec = 1.0 if candidate_id in sasrec_support else 0.0
        in_lightfm = 1.0 if candidate_id in lightfm_support else 0.0

        return {
            "genre_jaccard": round(genre_jaccard, 4),
            "same_artist": same_artist,
            "same_mood": same_mood,
            "year_diff_norm": round(year_diff_norm, 4),
            "log_artist_fans": round(log_fans, 4),
            "in_sasrec_i2i": in_sasrec,
            "in_lightfm_i2i": in_lightfm,
        }
