"""
evaluator.py
-------------
Shared evaluation utilities for Phase 2 (and beyond).

Provides:
    - compute_metrics()       : full metric dict for any model
    - compare_models()        : side-by-side table of all baselines
    - plot_roc_curves()       : ROC overlay for all models
    - plot_calibration()      : reliability diagram
    - early_detection_rate()  : fraction of drops caught N steps early
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ── Core metric computation ────────────────────────────────────────────────────

def compute_metrics(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    thresh:  float = 0.5,
    name:    str   = "model",
) -> dict[str, Any]:
    """
    Compute classification and calibration metrics.

    Returns
    -------
    dict with keys: name, auc, avg_precision, f1, precision,
                    recall, brier, threshold
    """
    y_pred = (y_proba >= thresh).astype(int)

    return {
        "name":          name,
        "auc":           round(float(roc_auc_score(y_true, y_proba)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_proba)), 4),
        "f1":            round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision":     round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":        round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "brier":         round(float(brier_score_loss(y_true, y_proba)), 4),
        "threshold":     thresh,
    }


def early_detection_rate(
    y_true:     np.ndarray,
    y_proba:    np.ndarray,
    lookahead:  int   = 3,
    thresh:     float = 0.5,
) -> float:
    """
    Fraction of actual drops for which the model predicted risk
    at least `lookahead` steps before the drop window.

    Assumes chronological ordering of samples.
    """
    drop_indices = np.where(y_true == 1)[0]
    caught = 0
    for idx in drop_indices:
        start = max(0, idx - lookahead)
        if any(y_proba[start:idx] >= thresh):
            caught += 1
    return caught / max(len(drop_indices), 1)


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_models(
    results: list[dict[str, Any]],
    save_path: str | None = None,
) -> None:
    """
    Print a formatted comparison table and optionally save as JSON.

    Parameters
    ----------
    results : list of metric dicts from compute_metrics()
    """
    header = f"{'Model':<22} {'AUC':>7} {'AP':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Brier':>7}"
    divider = "─" * len(header)
    print(f"\n{divider}")
    print(header)
    print(divider)
    for r in results:
        print(
            f"{r['name']:<22} "
            f"{r['auc']:>7.4f} "
            f"{r['avg_precision']:>7.4f} "
            f"{r['f1']:>7.4f} "
            f"{r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['brier']:>7.4f}"
        )
    print(divider)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Evaluator] Saved comparison → {save_path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

PALETTE = ["#00e5ff", "#39ff93", "#ffb347", "#bf7fff", "#ff6b6b"]


def plot_roc_curves(
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    save_path: str,
) -> None:
    """
    Parameters
    ----------
    curves : list of (y_true, y_proba, label) tuples
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0a0c10")
    ax.set_facecolor("#111318")

    for i, (y_true, y_proba, label) in enumerate(curves):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                lw=2, label=f"{label}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#4a5568", lw=1)
    ax.set_xlabel("False Positive Rate", color="#c8d6e5")
    ax.set_ylabel("True Positive Rate",  color="#c8d6e5")
    ax.set_title("ROC Curves — Baseline Comparison", color="#eaf2ff", fontsize=13)
    ax.legend(facecolor="#1e2330", edgecolor="#4a5568", labelcolor="#c8d6e5")
    ax.tick_params(colors="#4a5568")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2330")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] ROC plot → {save_path}")


def plot_calibration(
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    save_path: str,
    n_bins: int = 10,
) -> None:
    """
    Reliability diagram for probability calibration.

    Parameters
    ----------
    curves : list of (y_true, y_proba, label)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0a0c10")
    ax.set_facecolor("#111318")

    ax.plot([0, 1], [0, 1], "--", color="#4a5568", lw=1, label="Perfect calibration")

    for i, (y_true, y_proba, label) in enumerate(curves):
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, "o-",
                color=PALETTE[i % len(PALETTE)], lw=2, label=label)

    ax.set_xlabel("Mean Predicted Probability", color="#c8d6e5")
    ax.set_ylabel("Fraction of Positives",      color="#c8d6e5")
    ax.set_title("Calibration Curves",           color="#eaf2ff", fontsize=13)
    ax.legend(facecolor="#1e2330", edgecolor="#4a5568", labelcolor="#c8d6e5")
    ax.tick_params(colors="#4a5568")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2330")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Calibration plot → {save_path}")


def plot_training_history(
    history: list[dict[str, float]],
    save_path: str,
    title: str = "Training History",
) -> None:
    """Plot train vs val loss curves from LSTM/Transformer history."""
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0a0c10")
    ax.set_facecolor("#111318")

    ax.plot(epochs, train_loss, color="#00e5ff", lw=2, label="Train Loss")
    ax.plot(epochs, val_loss,   color="#39ff93", lw=2, label="Val Loss")
    ax.set_xlabel("Epoch", color="#c8d6e5")
    ax.set_ylabel("BCE Loss", color="#c8d6e5")
    ax.set_title(title, color="#eaf2ff", fontsize=13)
    ax.legend(facecolor="#1e2330", edgecolor="#4a5568", labelcolor="#c8d6e5")
    ax.tick_params(colors="#4a5568")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2330")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] History plot → {save_path}")