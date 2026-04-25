# src/inference/preprocess_input.py

import re


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_clauses(text: str):
    # Split by sentence boundaries
    parts = re.split(r'[.\n]+', text)

    clauses = []
    for p in parts:
        p = p.strip()
        if len(p) > 40:  # ignore very small noise
            clauses.append(p)

    return clauses


def load_text_input(text: str):
    """
    MAIN FUNCTION USED BY UI

    Returns:
    [
        {"id": 0, "text": "..."},
        {"id": 1, "text": "..."}
    ]
    """

    if not text:
        return []

    text = clean_text(text)
    raw_clauses = split_into_clauses(text)

    structured = []

    for idx, clause in enumerate(raw_clauses):
        structured.append({
            "id": idx,
            "text": clause
        })

    return structured