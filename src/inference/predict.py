import json
import torch
from transformers import AutoTokenizer

def load_threshold(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["best_threshold"]

def predict_clauses(model, tokenizer, clauses, device, max_length=256):
    texts = [c["text"] for c in clauses]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs["logits"]
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs