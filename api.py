from fastapi import FastAPI
from typing import Dict
import os
import json
import random

from src.inference.predict import load_model_and_tokenizer, predict_probabilities
from src.inference.postprocess_input import build_clause_results

app = FastAPI()

model = None
tokenizer = None
THRESHOLD = 0.5


# =========================
# PATH HANDLING (FIXED)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "contrastive_legal_bert.pt")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "contrastive_threshold.json")


# =========================
# LOAD MODEL SAFELY
# =========================
print(f"🔍 Looking for model at: {MODEL_PATH}")

try:
    if os.path.exists(MODEL_PATH):
        print("✅ Model found. Loading...")

        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=MODEL_PATH,
            device="cpu"
        )

        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                THRESHOLD = json.load(f).get("threshold", 0.5)

        print(f"✅ Model loaded. Threshold: {THRESHOLD}")

    else:
        print("⚠️ Model NOT found → fallback mode ENABLED")

except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None


# =========================
# API
# =========================
@app.post("/predict")
def predict(data: Dict):

    clauses = data.get("clauses", [])

    print(f"🔥 API HIT: {len(clauses)} clauses")

    if not clauses:
        return {"results": []}

    # =========================
    # FALLBACK
    # =========================
    if model is None:
        results = []

        for c in clauses:
            label = random.choice(["CRITICAL", "HIGH", "MEDIUM", "SAFE"])

            results.append({
                "id": c.get("id"),
                "text": c.get("text"),
                "severity_band": label,
                "severity_score": random.randint(1, 10),
                "explanation": "Fallback mode (model not loaded)"
            })

        return {"results": results}

    # =========================
    # REAL MODEL
    # =========================
    probs_multi, probs_binary = predict_probabilities(
        clauses,
        model,
        tokenizer,
        device="cpu"
    )

    results = build_clause_results(
        clauses,
        probs_multi,
        threshold=THRESHOLD
    )

    return {"results": results}