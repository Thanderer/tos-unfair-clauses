from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import BASE_MODEL_NAME, MAX_LENGTH
from src.models.baseline_legalbert import BaselineLegalBert


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[BaselineLegalBert, AutoTokenizer]:
    """
    Load the trained baseline LegalBERT model and tokenizer.

    checkpoint_path: path to .pt file saved by train_baseline.py
    device: 'cpu' or 'cuda'
    """
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    model = BaselineLegalBert(num_labels=8, use_binary_head=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer


def predict_probabilities(
    clauses: List[Dict],
    model: BaselineLegalBert,
    tokenizer: AutoTokenizer,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model on a list of clause dicts and return probabilities.

    clauses: list of dicts with at least a 'text' field
        [
          {
            "clause_id": int,
            "text": str,
            "start_char": int,
            "end_char": int,
          },
          ...
        ]

    Returns:
        probs: numpy array of shape [num_clauses, num_labels]
               with sigmoid probabilities for each label.
    """
    if not clauses:
        return np.empty((0, 8), dtype=float), np.empty((0,), dtype=float)

    device = torch.device(device)
    texts = [c["text"] for c in clauses]
    all_probs_multi: list[np.ndarray] = []  
    all_probs_binary: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}

            outputs = model(**encodings)
            logits = outputs["logits"]
            logits_bin = outputs["logits_binary"]

            probs_multi = torch.sigmoid(logits).cpu().numpy()         
            probs_binary = torch.sigmoid(logits_bin).squeeze(-1).cpu().numpy()

            all_probs_multi.append(probs_multi)
            all_probs_binary.append(probs_binary)

    return np.concatenate(all_probs_multi, axis=0), np.concatenate(all_probs_binary, axis=0)