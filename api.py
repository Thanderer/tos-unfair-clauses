from fastapi import FastAPI
from typing import Dict
import os
import random

from src.inference.predict import load_model_and_tokenizer, predict_probabilities
from src.inference.postprocess_input import build_clause_results

app = FastAPI()

model = None
tokenizer = None

MODEL_PATH = "models/baseline_legal_bert.pt"

# =========================
# SAFE MODEL LOAD
# =========================
if os.path.exists(MODEL_PATH):
    print("✅ Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=MODEL_PATH,
        device="cpu"
    )
else:
    print("⚠️ Model NOT found → using fallback mode")


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: Dict):

    clauses = data.get("clauses", [])

    print(f"🔥 API HIT: {len(clauses)} clauses")

    if not clauses:
        return {"results": []}

    # =========================
    # FALLBACK MODE (NO MODEL)
    # =========================
    if model is None:
        results = []
        for c in clauses:
            label = random.choice(["HIGH", "MEDIUM", "SAFE"])

            results.append({
                "id": c.get("id"),
                "text": c.get("text"),
                "severity_band": label,
                "severity_score": random.randint(1, 10),
                "explanation": "Model not loaded (fallback mode)"
            })

        return {"results": results}

    # =========================
    # REAL MODEL
    # =========================
    probs_multi, _ = predict_probabilities(
        clauses,
        model,
        tokenizer,
        device="cpu"
    )

    results = build_clause_results(
        clauses,
        probs_multi,
        threshold=0.5
    )

    return {"results": results}