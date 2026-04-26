import re


# =========================
# CLEAN TEXT
# =========================
def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")

    # Remove headers/footers
    text = re.sub(r'Page \d+.*', '', text)

    # Remove bullets
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)

    # Remove numbering (1. 2. etc.)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# =========================
# SPLIT CLAUSES
# =========================
def split_into_clauses(text: str):
    chunks = re.split(r'(?<=[.!?])\s+|\n{2,}', text)

    clauses = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()

        if len(chunk) < 20:
            continue

        clauses.append({
            "id": i,
            "text": chunk
        })

    return clauses


# =========================
# MAIN
# =========================
def load_text_input(text: str):
    text = clean_text(text)
    return split_into_clauses(text)