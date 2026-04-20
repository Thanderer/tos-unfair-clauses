# ToS Unfair Clause Detector

This project detects potentially unfair clauses in online Terms of Service (ToS) and assigns a **severity score** to highlight how intrusive each clause is compared to standard practice.

It is the final project for the **Human-Centred Natural Language Processing** course.

---

## Project goals

- Train a baseline **LegalBERT** classifier on the Lippi et al. (2019) ToS dataset.
- Extend it with **contrastive learning** to better separate “standard/fair” vs. “intrusive/unfair” clauses.
- Define a **severity score (1–10)** and simple layman labels, such as:
  - “You are good to go”
  - “Needs another look”
  - “This might be trouble”
  - “DO NOT AGREE TO THIS”
- Build a small **web UI** where a user can upload a ToS (text/PDF) and see:
  - Highlighted problematic clauses
  - Clause-level severity + short explanation
  - An overall document verdict

---

## Installation

To set up the project locally and ensure all dependencies are consistent across the team, run the following commands:

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv

# 2. Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\\venv\\Scripts\\activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## Repository structure

```text
tos-unfair-clauses/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 00_colab_tutorials/      # Seminar / example notebooks (read-only)
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_legalbert.ipynb
│   ├── 03_contrastive_learning.ipynb
│   └── 04_frontend_integration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, model names, hyperparameters
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_unfair_tos.py   # Load Lippi et al. dataset / LexGLUE UNFAIR-ToS
│   │   ├── preprocess_tosdr.py  # (Optional) ToS;DR preprocessing
│   │   └── utils_pdf_text.py    # PDF → text, sentence splitting, cleaning
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_legalbert.py      # Baseline classifier
│   │   └── contrastive_legalbert.py   # Classifier + contrastive head
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baseline.py    # Train baseline model
│   │   ├── train_contrastive.py # Train contrastive model
│   │   └── evaluate.py          # F1, AUC, PR-AUC, nDCG, Kendall tau, etc.
│   │
│   └── frontend/
│       ├── __init__.py
│       ├── severity_mapping.py  # Model outputs → [1–10] severity → text labels
│       └── app.py               # Gradio (or similar) UI
│
├── data/
│   ├── raw/         # Original datasets (not committed)
│   ├── interim/     # Cleaned / split CSVs
│   └── processed/   # Model-ready data
│
├── reports/
│   ├── hcnlp_final_report.tex   # 12-page report (ACM 1-column)
│   └── figures/
│       └── ...                  # Plots, diagrams, UI screenshots
│
├── slides/
│   └── presentation.pptx        # ≤ 30 slides
│
└── scripts/
    ├── run_baseline.sh          # (optional) convenience scripts
    ├── run_contrastive.sh
    └── run_app.sh
```

---

## Current baseline

The baseline LegalBERT model is trained and evaluated on UNFAIR-ToS. The model uses multi-label predictions for unfair clause types and a binary unfair-vs-fair head for document-level detection. The best validation threshold was tuned to 0.30 and saved for later use.

### Saved outputs

- `models/baseline_legal_bert.pt` — trained baseline checkpoint.
- `models/baseline_threshold.json` — tuned prediction threshold from validation.

### Test results

- Multi-label macro F1: 0.5570
- Multi-label micro F1: 0.6387
- Binary unfair-vs-fair F1: 0.7517
- Binary ROC-AUC: 0.9649
- Binary PR-AUC: 0.8973

---

## Why this baseline design

- We chose UNFAIR-ToS because it directly matches the unfair-clause detection task and provides supervised clause-level labels.
- We used LegalBERT as a strong baseline to establish a reliable benchmark before adding more advanced methods.
- The labels were converted into fixed multi-hot vectors because the dataset annotations are variable-length lists of unfair clause indices.
- We tuned the prediction threshold on validation data because 0.5 was too conservative and 0.30 gave better micro F1.

---

## Planned next steps

- Implement contrastive learning to improve separation between fair and unfair clauses.
- Add severity scoring from 1–10 with simple layman labels.
- Build a small UI for text/PDF upload, highlighted clauses, severity explanations, and overall verdict.
- Optionally use ToS;DR for qualitative evaluation and demo examples rather than main training.

---

## Run baseline

```bash
python -m src.training.train_baseline
```

## Evaluate baseline

```bash
python -m src.training.evaluate
```

---

## Notes

- The UNFAIR-ToS preprocessing expects label lists to be converted into fixed-length multi-hot vectors before batching.
- The evaluation script should load the saved threshold file so that test metrics use the tuned cutoff instead of a default 0.5 threshold.