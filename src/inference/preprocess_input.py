from pathlib import Path
import re

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_into_clauses(text: str):
    chunks = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    clauses = []
    cursor = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 20:
            continue
        start = text.find(chunk, cursor)
        end = start + len(chunk)
        cursor = end
        clauses.append({
            "clause_id": i,
            "text": chunk,
            "start_char": start,
            "end_char": end,
        })
    return clauses

def load_text_input(text: str):
    text = clean_text(text)
    return split_into_clauses(text)