import requests
import joblib
import torch
import numpy as np
from bs4 import BeautifulSoup
from googlesearch import search
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMOTION_COORDINATES = {
    "Sadness": (-0.82, -0.45), "Disappointment": (-0.6, -0.3), "Guilt": (-0.6, -0.4),
    "Tension": (-0.55, 0.70), "Anger": (-0.7, 0.8), "Fear": (-0.64, 0.6), "Confusion": (-0.4, 0.5),
    "Tenderness": (0.65, -0.30), "Empathy": (0.50, -0.20), "Reflection": (0.20, -0.40), "Calmness": (0.8, -0.6),
    "Hopefulness": (0.65, 0.30), "Joy": (0.85, 0.65), "Empowerment": (0.75, 0.75), "Excitement": (0.8, 0.8), "Gratitude": (0.7, 0.5)
}


def scrape_mojim(title: str) -> str:
    query = f"site:mojim.com {title}"
    target_url = None
    try:
        results = search(query, num_results=3, advanced=True)
        for result in results:
            if "mojim.com" in result.url:
                target_url = result.url
                break
        if not target_url:
            return "(Lyrics not found on Mojim)"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(target_url, headers=headers, timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "lxml")

        content = soup.find(id="fsZx3")
        if not content:
            return "(Mojim structure error)"

        for i in content("ol"):
            i.extract()
        for script in content("script"):
            script.extract()

        lyrics_lines = []
        for line in content.stripped_strings:
            if any(x in line for x in ["更多更詳盡歌詞", "Mojim.com", "魔鏡歌詞網", "：", "[", "]", "--"]):
                continue
            lyrics_lines.append(line)

        lyrics = "\n".join(lyrics_lines)
        return lyrics
    except Exception:
        return ""


class MLEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        word_embedding_model = models.Transformer("bert-base-uncased", max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)
        self.models_list = joblib.load("models/xgb_per_label.joblib")
        self.mlb = joblib.load("models/label_binarizer.joblib")
        self.labels = self.mlb.classes_


try:
    engine = MLEngine()
except Exception:
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
    return float(v_final), float(a_final)


class SongData(BaseModel):
    title: str
class SongRequest(BaseModel):
    title: str

class HealingRequest(BaseModel):
    user_mood: str
    songs: List[SongData]
@app.post("/add_song")
async def add_song(req: SongRequest):
    lyrics = scrape_mojim(req.title)
    
    v, a = predict_emotion(lyrics) if len(lyrics) > 10 else (0,0)
    
    return {
        "title": req.title, 
        "lyrics": lyrics[:100]+"...", 
        "v": v, 
        "a": a
    }

@app.post("/generate_playlist")
async def generate(req: HealingRequest):
    if req.user_mood in EMOTION_COORDINATES:
        user_v, user_a = EMOTION_COORDINATES[req.user_mood]
        user_v, user_a = float(user_v), float(user_a)
    else:
        user_v, user_a = predict_emotion(req.user_mood)

    songs = []
    for s in req.songs:
        lyrics = scrape_mojim(s.title)
        if lyrics and len(lyrics) > 10:
            v, a = predict_emotion(lyrics)
        else:
            v, a = 0.0, 0.0
        songs.append({
            "title": s.title,
            "v": float(v),
            "a": float(a),
        })

    goal_v, goal_a = 0.85, 0.65
    path = []
    for i in range(5):
        t = i / 4.0
        v = float(user_v * (1 - t) + goal_v * t)
        a = float(user_a * (1 - t) + goal_a * t)
        path.append({"stage": i + 1, "v": v, "a": a})

    THRESHOLD = 0.8  # 最大允許距離，超過就不選這首歌

    final_playlist = []
    pool = songs.copy()

    for step in path:
        if not pool:
            break

        candidates = []
        for s in pool:
            if s["v"] == 0.0 and s["a"] == 0.0:
                continue
            dist = float(np.sqrt((s["v"] - step["v"]) ** 2 + (s["a"] - step["a"]) ** 2))
            if dist <= THRESHOLD:
                candidates.append((dist, s))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        selected = candidates[0][1]

        final_playlist.append({
            "stage_number": step["stage"],
            "song": {
                "title": selected["title"],
                "v": float(selected["v"]),
                "a": float(selected["a"]),
            }
        })

        pool = [s for s in pool if s["title"] != selected["title"]]

    return {"path": path, "playlist": final_playlist}
