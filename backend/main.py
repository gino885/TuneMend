import math
import requests
import joblib
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, models
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMOTION_COORDINATES = {
    "Sadness": (-0.82, -0.45),
    "Disappointment": (-0.6, -0.3),
    "Guilt": (-0.6, -0.4),
    "Tension": (-0.55, 0.70),
    "Anger": (-0.7, 0.8),
    "Fear": (-0.64, 0.6),
    "Confusion": (-0.4, 0.5),
    "Tenderness": (0.65, -0.30),
    "Empathy": (0.50, -0.20),
    "Reflection": (0.20, -0.40),
    "Calmness": (0.8, -0.6),
    "Hopefulness": (0.65, 0.30),
    "Joy": (0.85, 0.65),
    "Empowerment": (0.75, 0.75),
    "Excitement": (0.8, 0.8),
    "Gratitude": (0.7, 0.5),
}


class MLEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")

        word_embedding_model = models.Transformer(
            "bert-base-uncased", max_seq_length=256
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        self.encoder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device=self.device,
        )

        xgb_path = os.path.join(models_dir, "xgb_per_label.joblib")
        lb_path = os.path.join(models_dir, "label_binarizer.joblib")

        print("[MLEngine] loading models from:", models_dir)
        self.models_list = joblib.load(xgb_path)
        self.mlb = joblib.load(lb_path)
        self.labels = self.mlb.classes_


try:
    engine = MLEngine()
    print("[MLEngine] loaded on device:", engine.device)
except Exception as e:
    print("[MLEngine] failed to init:", repr(e))
    engine = None


def predict_emotion(text: str):
    text_len = len(text) if text is not None else 0
    if not engine or text_len < 5:
        return 0.0, 0.0

    emb = engine.encoder.encode([text])

    probs = [float(clf.predict_proba(emb)[0][1]) for clf in engine.models_list]

    v_sum, a_sum, weight_sum = 0.0, 0.0, 0.0
    for i, prob in enumerate(probs):
        label_name = engine.labels[i]
        if prob > 0.15 and label_name in EMOTION_COORDINATES:
            v, a = EMOTION_COORDINATES[label_name]
            v_sum += v * prob
            a_sum += a * prob
            weight_sum += prob

    if weight_sum == 0.0:
        return 0.0, 0.0

    v_final = v_sum / weight_sum
    a_final = a_sum / weight_sum
    return v_final, a_final


def closest_emotion(v: float, a: float) -> Optional[str]:
    best_label = None
    best_dist = None
    for label, (ev, ea) in EMOTION_COORDINATES.items():
        d = math.sqrt((v - ev) ** 2 + (a - ea) ** 2)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_label = label
    return best_label


def pick_positive_goal(user_v: float, user_a: float):
    positive_labels = []
    for label, (v, a) in EMOTION_COORDINATES.items():
        if v > 0.3:
            positive_labels.append((label, v, a))

    if not positive_labels:
        return "Joy", EMOTION_COORDINATES["Joy"][0], EMOTION_COORDINATES["Joy"][1]

    candidates = []
    for label, v, a in positive_labels:
        if v >= user_v:
            dist = math.sqrt((v - user_v) ** 2 + (a - user_a) ** 2)
            candidates.append((dist, label, v, a))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, label, v, a = candidates[0]
        return label, v, a

    positive_labels.sort(key=lambda x: x[1], reverse=True)
    label, v, a = positive_labels[0]
    return label, v, a


class SongData(BaseModel):
    title: str
    artist: Optional[str] = None
    lyrics: Optional[str] = None


class SongRequest(BaseModel):
    title: str
    artist: Optional[str] = None


class HealingRequest(BaseModel):
    user_mood: str
    songs: List[SongData]


def fetch_lyrics_ovh(title: str, artist: Optional[str]) -> str:
    if not artist:
        artist = "_"
    url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    print("[lyrics.ovh] GET:", url)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return ""
        data = r.json()
        return data.get("lyrics", "")
    except Exception as e:
        print("[lyrics.ovh] error:", e)
        return ""


@app.post("/add_song")
async def add_song(req: SongRequest):
    lyrics = fetch_lyrics_ovh(req.title, req.artist)
    return {
        "title": req.title,
        "artist": req.artist or "",
        "lyrics": lyrics,
    }


@app.post("/generate_playlist")
async def generate(req: HealingRequest):

    if req.user_mood in EMOTION_COORDINATES:
        user_v, user_a = EMOTION_COORDINATES[req.user_mood]
    else:
        user_v, user_a = predict_emotion(req.user_mood)

    user_v, user_a = float(user_v), float(user_a)
    user_emotion = closest_emotion(user_v, user_a)

    print("==== USER STATE ====")
    print("input:", req.user_mood)
    print("v:", user_v, "a:", user_a, "emotion:", user_emotion)

    goal_label, goal_v, goal_a = pick_positive_goal(user_v, user_a)
    goal_v, goal_a = float(goal_v), float(goal_a)

    print("==== GOAL ====")
    print("goal:", goal_label, "v:", goal_v, "a:", goal_a)

    songs = []
    print("==== SONG EMOTION PREDICTION ====")

    for s in req.songs:
        lyrics = s.lyrics or ""

        if len(lyrics) > 10:
            v, a = predict_emotion(lyrics)
        else:
            v, a = 0.0, 0.0

        emotion = closest_emotion(v, a)

        print(f"SONG: {s.title} - {s.artist}")
        print("lyrics length:", len(lyrics))
        print("v:", v, "a:", a, "emotion:", emotion)
        print("-----")

        songs.append(
            {
                "title": s.title,
                "artist": s.artist,
                "lyrics": lyrics,
                "v": float(v),
                "a": float(a),
            }
        )

    path = []
    NUM_STAGES = 5
    for i in range(NUM_STAGES):
        t = i / float(NUM_STAGES - 1)
        v = float(user_v * (1.0 - t) + goal_v * t)
        a = float(user_a * (1.0 - t) + goal_a * t)
        emotion = closest_emotion(v, a)
        path.append({
            "stage": i + 1,
            "v": v,
            "a": a,
            "emotion": emotion
        })

    print("==== HEALING PATH ====")
    for p in path:
        print(p)

    THRESHOLD = 0.8
    final_playlist = []
    pool = songs.copy()

    print("==== MATCHING SONGS TO PATH ====")

    for step in path:
        print(f"STAGE {step['stage']} TARGET:", step["emotion"], step["v"], step["a"])

        if not pool:
            break

        candidates = []
        for s in pool:
            if s["v"] == 0.0 and s["a"] == 0.0:
                print("SKIP (0,0):", s["title"])
                continue

            dist = float(
                np.sqrt((s["v"] - step["v"]) ** 2 + (s["a"] - step["a"]) ** 2)
            )

            print("COMPARE:", s["title"], "DIST =", dist)

            if dist <= THRESHOLD:
                candidates.append((dist, s))

        if not candidates:
            print("NO MATCHED SONG FOR THIS STAGE")
            continue

        candidates.sort(key=lambda x: x[0])
        selected = candidates[0][1]

        print("SELECTED:", selected["title"], selected["artist"])

        final_playlist.append(
            {
                "stage_number": step["stage"],
                "stage_emotion": step["emotion"],
                "song": {
                    "title": selected["title"],
                    "artist": selected["artist"],
                    "lyrics": selected["lyrics"],
                    "v": float(selected["v"]),
                    "a": float(selected["a"]),
                },
            }
        )

        pool = [s for s in pool if s["title"] != selected["title"]]

    return {
        "user_state": {
            "input": req.user_mood,
            "v": user_v,
            "a": user_a,
            "emotion": user_emotion,
        },
        "goal": {
            "label": goal_label,
            "v": goal_v,
            "a": goal_a,
        },
        "path": path,
        "playlist": final_playlist,
    }
