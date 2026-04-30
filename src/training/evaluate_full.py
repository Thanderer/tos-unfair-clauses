
from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from itertools import combinations
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.config import MODELS_DIR, BATCH_SIZE
from src.data.load_unfair_tos import prepare_unfair_tos_datasets
from src.models.baseline_legalbert import BaselineLegalBert
from src.models.contrastive_legalbert import ContrastiveLegalBert
from src.training.train_baseline import collate_fn as baseline_collate

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def contrastive_collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "labels":         torch.stack([b["labels"]          for b in batch]).float(),
        "label_binary":   torch.stack([b["label_binary"]    for b in batch]),
    }

def compute_spearman(probs_bin, y_true_bin):
    corr, pvalue = spearmanr(probs_bin, y_true_bin)
    return float(corr), float(pvalue)

def compute_pairwise_accuracy(probs_bin, y_true_bin):
    correct = 0
    total   = 0
    for i, j in combinations(range(len(probs_bin)), 2):
        if y_true_bin[i] != y_true_bin[j]:
            total += 1
            if (probs_bin[i] > probs_bin[j]) == (y_true_bin[i] > y_true_bin[j]):
                correct += 1
    return correct / total if total > 0 else 0.0

def get_baseline_preds(device):
    print("Loading baseline model...")
    model = BaselineLegalBert(num_labels=8, use_binary_head=True)
    state = torch.load(MODELS_DIR / "baseline_legal_bert.pt", map_location=device)
    state = {k: v for k, v in state.items()
             if k not in ["pos_weight", "loss_fct_multi.pos_weight"]}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with open(MODELS_DIR / "baseline_threshold.json") as f:
        data  = json.load(f)
        b_thr = data["threshold"]

    ds, _ = prepare_unfair_tos_datasets(max_length=256)
    loader = DataLoader(ds["test"], batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=baseline_collate)

    all_logits, all_labels, all_binary = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=None, label_binary=None)
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_binary.append(batch["label_binary"].cpu())

    logits    = torch.cat(all_logits).numpy()
    y_true    = torch.cat(all_labels).numpy()
    y_bin     = torch.cat(all_binary).numpy()
    probs     = sigmoid(logits)
    probs_bin = probs.max(axis=1)
    return probs, y_true, y_bin, probs_bin, b_thr

def get_contrastive_preds(device):
    print("Loading contrastive model...")
    model = ContrastiveLegalBert(num_labels=8)
    state = torch.load(MODELS_DIR / "contrastive_legal_bert.pt", map_location=device)
    state = {k: v for k, v in state.items()
             if k not in ["loss_fct.pos_weight"]}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with open(MODELS_DIR / "contrastive_threshold.json") as f:
        data    = json.load(f)
        c_thr   = data["threshold"]
        c_b_thr = data.get("binary_threshold", 0.80)

    ds, _ = prepare_unfair_tos_datasets(max_length=256)
    loader = DataLoader(ds["test"], batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=contrastive_collate)

    all_logits, all_labels, all_binary, all_bin_logits = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs["logits"].cpu())
            all_bin_logits.append(outputs["logits_binary"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_binary.append(batch["label_binary"].cpu())

    logits     = torch.cat(all_logits).numpy()
    y_true     = torch.cat(all_labels).numpy()
    y_bin      = torch.cat(all_binary).numpy()
    bin_logits = torch.cat(all_bin_logits).numpy().squeeze()
    probs      = sigmoid(logits)
    probs_bin  = sigmoid(bin_logits)
    return probs, y_true, y_bin, probs_bin, c_thr

def print_full_report(name, probs, y_true, y_bin, probs_bin, thr):
    y_pred     = (probs >= thr).astype(int)
    y_pred_bin = (probs_bin >= 0.5).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    bin_f1   = f1_score(y_bin, y_pred_bin, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_bin, probs_bin)
    except:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_bin, probs_bin)
    except:
        pr_auc = float("nan")

    spearman_corr, spearman_p = compute_spearman(probs_bin, y_bin)
    pairwise_acc              = compute_pairwise_accuracy(probs_bin, y_bin)

    print(f"\n{'='*55}")
    print(f"  {name} Model — Full Evaluation")
    print(f"{'='*55}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Micro F1:           {micro_f1:.4f}")
    print(f"  Binary F1:          {bin_f1:.4f}")
    print(f"  ROC-AUC:            {roc_auc:.4f}")
    print(f"  PR-AUC:             {pr_auc:.4f}")
    print(f"  Spearman rho:       {spearman_corr:.4f}  (p={spearman_p:.4f})")
    print(f"  Pairwise Accuracy:  {pairwise_acc:.4f}")
    print(f"{'='*55}\n")

    return {
        "macro_f1":          round(macro_f1, 4),
        "micro_f1":          round(micro_f1, 4),
        "binary_f1":         round(bin_f1, 4),
        "roc_auc":           round(roc_auc, 4),
        "pr_auc":            round(pr_auc, 4),
        "spearman_rho":      round(spearman_corr, 4),
        "spearman_p":        round(spearman_p, 4),
        "pairwise_accuracy": round(pairwise_acc, 4),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    Path("reports").mkdir(exist_ok=True)

    # Baseline
    b_probs, b_y_true, b_y_bin, b_probs_bin, b_thr = get_baseline_preds(device)
    b_results = print_full_report(
        "Baseline", b_probs, b_y_true, b_y_bin, b_probs_bin, b_thr)

    # Contrastive
    c_probs, c_y_true, c_y_bin, c_probs_bin, c_thr = get_contrastive_preds(device)
    c_results = print_full_report(
        "Contrastive", c_probs, c_y_true, c_y_bin, c_probs_bin, c_thr)

    # Comparison table
    print("\n" + "="*60)
    print(f"{'Metric':<25} {'Baseline':>12} {'Contrastive':>12}")
    print("="*60)
    for key in b_results:
        b = b_results[key]
        c = c_results[key]
        winner = "✅ C" if c > b else "✅ B"
        print(f"{key:<25} {b:>12.4f} {c:>12.4f}  {winner}")
    print("="*60)

    # Save results
    with open("reports/full_metrics_baseline.json", "w") as f:
        json.dump(b_results, f, indent=2)
    with open("reports/full_metrics_contrastive.json", "w") as f:
        json.dump(c_results, f, indent=2)
    print("\nSaved to reports/full_metrics_baseline.json")
    print("Saved to reports/full_metrics_contrastive.json")

if __name__ == "__main__":
    main()
