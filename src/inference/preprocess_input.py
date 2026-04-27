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

    # Remove numbering
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
    clause_id = 0  # ✅ FIX: sequential IDs

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        # ✅ FIX: do NOT drop short clauses

        # ✅ FIX: split long clauses (>500 chars)
        if len(chunk) > 500:
            sub_chunks = re.split(r'[;,]', chunk)
        else:
            sub_chunks = [chunk]

        for sub in sub_chunks:
            sub = sub.strip()

            if not sub:
                continue

            clauses.append({
                "id": clause_id,
                "text": sub
            })

            clause_id += 1  # increment only on append

    return clauses


# =========================
# MAIN
# =========================
def load_text_input(text: str):
    text = clean_text(text)
    return split_into_clauses(text)