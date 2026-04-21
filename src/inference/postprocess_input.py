LABEL_NAMES = [
    "arbitration",
    "unilateral_change",
    "content_removal",
    "jurisdiction",
    "choice_of_law",
    "limitation_of_liability",
    "unilateral_termination",
    "contract_by_using"
]

def prob_to_band(score: float) -> str:
    if score >= 0.61:
        return "HIGH"
    if score >= 0.31:
        return "MEDIUM"
    return "SAFE"

def prob_to_severity(score: float) -> int:
    return max(1, min(10, int(round(score * 10))))

def build_results(clauses, probs, threshold=0.30):
    results = []
    for clause, row in zip(clauses, probs):
        max_prob = float(row.max())
        pred_ids = [i for i, p in enumerate(row) if p >= threshold]
        pred_labels = [LABEL_NAMES[i] for i in pred_ids]

        results.append({
            **clause,
            "probability": max_prob,
            "predicted_labels": pred_labels,
            "severity_score": prob_to_severity(max_prob),
            "severity_band": prob_to_band(max_prob),
        })
    return results