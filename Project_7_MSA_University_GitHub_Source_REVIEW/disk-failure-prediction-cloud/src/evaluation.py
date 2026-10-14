"""Operationally relevant, prevalence-preserving failure metrics and plots."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, roc_curve)


def threshold_for_recall(y_true, scores, target_recall: float = 0.8) -> tuple[float, float]:
    """Largest available validation threshold with recall >= desired target."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if np.count_nonzero(y) == 0:
        raise ValueError("No positive cases in validation; threshold cannot be selected.")
    _, rec, thresholds = precision_recall_curve(y, s)
    valid = np.flatnonzero(rec[:-1] >= target_recall)
    if not len(valid):
        raise ValueError("No threshold can satisfy the requested validation recall.")
    idx = valid[-1]
    return float(thresholds[idx]), float(rec[idx])


def metrics(y_true, score, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "observations": int(len(y)), "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "avg_precision_pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "false_alarms_per_1000_drive_days": float(fp / len(y) * 1000) if len(y) else None,
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }


def make_plots(predictions: dict, out_dir):
    """Plot actual test results only, no mock final report graphics."""
    from pathlib import Path
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for kind in ("pr", "roc"):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for name, payload in predictions.items():
            y, score = payload["target"], payload["score"]
            if np.unique(y).size < 2:
                continue
            if kind == "pr":
                prec, rec, _ = precision_recall_curve(y, score)
                ax.plot(rec, prec, label=name)
                ax.set(xlabel="Recall", ylabel="Precision", title="Test-set precision–recall")
            else:
                fpr, tpr, _ = roc_curve(y, score)
                ax.plot(fpr, tpr, label=name)
                ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test-set ROC")
        ax.legend(loc="best")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(folder / f"{kind}_curves.png", dpi=170)
        plt.close(fig)


def by_drive_model(frame: pd.DataFrame, score, threshold: float) -> list[dict]:
    f = frame[["model", "target"]].copy()
    f["score"] = score
    result = []
    for model, part in f.groupby("model"):
        row = metrics(part["target"], part["score"], threshold)
        row["model"] = str(model)
        result.append(row)
    return result
