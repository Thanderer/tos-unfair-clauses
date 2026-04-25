# src/frontend/severity_mapping.py

import math


def logits_to_severity(logits) -> int:
    """
    Convert probability/logits → 1–10 severity score
    """

    if 0 <= logits <= 1:
        p = logits
    else:
        p = 1 / (1 + math.exp(-logits))

    score = int(round(p * 10))
    return max(1, min(10, score))


def severity_label(score: int) -> str:
    if score <= 3:
        return "You are good to go"
    if score <= 5:
        return "Needs another look"
    if score <= 7:
        return "This might be trouble"
    return "DO NOT AGREE TO THIS"


def aggregate_severity(results):
    """
    Compute document-level severity
    """
    if not results:
        return 1

    scores = [r["severity_score"] for r in results]

    avg = sum(scores) / len(scores)
    mx = max(scores)

    final = int(round(0.6 * mx + 0.4 * avg))
    return max(1, min(10, final))