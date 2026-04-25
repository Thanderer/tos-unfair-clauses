# ToS Unfair Clause Detector

This project detects potentially unfair clauses in online Terms of Service (ToS) and assigns a **severity score** to highlight how intrusive each clause is compared to standard practice.

It is the final project for the **Human-Centred Natural Language Processing** course.

---

## Project goals

- Train a baseline **LegalBERT** classifier on the UNFAIR-ToS / LexGLUE dataset.
- Extend it with **contrastive learning** to better separate "standard/fair" vs. "intrusive/unfair" clauses.
- Define a **severity score (1–10)** and simple layman labels, such as:
  - "You are good to go"
  - "Needs another look"
  - "This might be trouble"
  - "DO NOT AGREE TO THIS"
- Build a small **web UI** where a user can upload a ToS (text/PDF) and see:
  - Highlighted problematic clauses
  - Clause-level severity + short explanation
  - An overall document verdict

---

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv

# 2. Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## Repository structure

```text
tos-unfair-clauses/
├── README.md
├── requirements.txt
├── .gitattributes
│
├── notebooks/
│   └── 02_baseline_legalbert.ipynb   # Colab training notebook
│
├── data/
│   ├── raw/         # Original datasets (not committed)
│   ├── interim/     # Cleaned / split CSVs
│   └── processed/   # Model-ready data
│
├── models/
│   ├── baseline_legal_bert.pt        # Trained baseline checkpoint
│   └── baseline_threshold.json       # Tuned multi-label + binary thresholds
│
├── reports/
│   └── baseline_metrics.json         # Test set evaluation results
│
├── scripts/
│   ├── run_baseline.sh
│   ├── run_contrastive.sh
│   └── run_app.sh
│
├── slides/
│
└── src/
    ├── __init__.py
    ├── config.py                      # Paths, model names, hyperparameters
    │
    ├── data/
    │   ├── __init__.py
    │   ├── load_unfair_tos.py         # LexGLUE UNFAIR-ToS loader + tokenization
    │   ├── preprocess_tosdr.py        # ToS;DR preprocessing (optional)
    │   └── utils_pdf_text.py          # PDF → text, sentence splitting, cleaning
    │
    ├── models/
    │   ├── __init__.py
    │   ├── baseline_legalbert.py      # Dual-head classifier (8-label + binary)
    │   └── contrastive_legalbert.py   # Classifier + projection head (TODO)
    │
    ├── training/
    │   ├── __init__.py
    │   ├── train_baseline.py          # Train + threshold tuning + checkpoint save
    │   ├── train_contrastive.py       # Contrastive training pipeline (TODO)
    │   └── evaluate.py                # F1, ROC-AUC, PR-AUC on test split
    │
    ├── inference/
    │   ├── __init__.py
    │   ├── preprocess_input.py        # Clean text + split into clauses
    │   ├── predict.py                 # Load model, run inference, return probs
    │   └── postprocess_input.py       # Probs → severity band + explanations
    │
    └── frontend/
        ├── __init__.py
        ├── severity_mapping.py        # Severity score → layman label (TODO)
        └── app.py                     # Gradio UI (TODO)
```

---

## Current baseline results

The baseline LegalBERT model has a dual-head architecture: an 8-label multi-label head for unfair clause type classification and a dedicated binary head for fair/unfair detection. Both heads are trained jointly with BCEWithLogitsLoss. Separate thresholds are tuned on the validation set for each head and saved to `baseline_threshold.json`.

### Saved outputs

- `models/baseline_legal_bert.pt` — trained baseline checkpoint
- `models/baseline_threshold.json` — contains `threshold` (multi-label, 0.35) and `binary_threshold` (binary head, tuned separately)
- `reports/baseline_metrics.json` — full test metrics

### Test results (UNFAIR-ToS)

| Metric | Score |
|---|---|
| Multi-label macro F1 | 0.5120 |
| Multi-label micro F1 | 0.6333 |
| Binary unfair-vs-fair F1 | 0.7397 |
| Binary ROC-AUC | **0.9674** |
| Binary PR-AUC | **0.8807** |

The strong ROC-AUC (0.9674) confirms the model ranks clauses reliably. The macro F1 gap relative to published LegalBERT baselines (~0.83) is primarily driven by class imbalance across rare unfair clause types.

---

## Why this baseline design

- UNFAIR-ToS provides supervised clause-level labels that directly match the task.
- LegalBERT is pre-trained on legal text, giving a stronger domain baseline than general BERT.
- The dual-head design separates the "what type" (8-label) and "is it unfair" (binary) signals, with separate threshold tuning for each.
- Multi-hot label conversion is required because the dataset stores annotations as variable-length index lists.

---

## Planned next steps

- Implement supervised contrastive loss in `train_contrastive.py` to improve embedding separation between fair and unfair clauses.
- Complete `severity_mapping.py` (`logits_to_severity`) and wire it into the inference pipeline.
- Build the Gradio UI in `app.py` for text/PDF upload with highlighted clauses and severity output.
- Add class-weighted loss and per-label thresholds to improve macro F1.

---

## Run baseline training

```bash
python -m src.training.train_baseline
```

## Evaluate on test set

```bash
python -m src.training.evaluate
```

---

## Notes

- `baseline_threshold.json` stores two keys: `threshold` (used for 8-label multi-label predictions) and `binary_threshold` (used for the binary head). Both are loaded separately in `evaluate.py` and `predict.py`.
- The inference pipeline runs: `preprocess_input.py` → `predict.py` → `postprocess_input.py` and returns per-clause severity bands (`SAFE` / `MEDIUM` / `HIGH`) and explanations.
- Training was run on Google Colab (GPU). CPU training is supported but slow.