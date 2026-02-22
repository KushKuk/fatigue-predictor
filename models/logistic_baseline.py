"""
logistic_baseline.py
---------------------
Logistic Regression baseline for fatigue drop prediction.

Serves as the AUC floor that the LSTM and Transformer must beat.
Includes:
    - Grid-searched C and solver
    - Calibration via Platt scaling (CalibratedClassifierCV)
    - Full evaluation report
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Suppress sklearn FutureWarning about deprecated penalty param
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ── Hyperparameter grid ────────────────────────────────────────────────────────
PARAM_GRID: dict[str, list[Any]] = {
    "clf__C":      [0.01, 0.1, 1.0, 10.0],
    "clf__solver": ["lbfgs", "saga"],
}


def build_logistic_pipeline(pos_weight: float = 1.0) -> Pipeline:
    """Return a Pipeline with a class-weighted LR classifier."""
    class_weight = {0: 1.0, 1: pos_weight}
    clf = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=42,
    )
    return Pipeline([("clf", clf)])


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    pos_weight: float = 1.0,
    cv_folds: int = 3,
    verbose: bool = True,
) -> CalibratedClassifierCV:
    """
    Grid-search best LR hyperparams on train, calibrate on val.

    Returns a fitted CalibratedClassifierCV (Platt scaling).
    """
    pipe = build_logistic_pipeline(pos_weight)

    gs = GridSearchCV(
        pipe,
        PARAM_GRID,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    if verbose:
        print(f"[LR] Best params : {gs.best_params_}")
        print(f"[LR] CV AUC      : {gs.best_score_:.4f}")

    best_pipe = gs.best_estimator_

    # Platt scaling: calibrate on combined train+val with cv=3
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    calibrated = CalibratedClassifierCV(best_pipe, method="sigmoid", cv=3)
    calibrated.fit(X_all, y_all)

    return calibrated


def evaluate(
    model: CalibratedClassifierCV,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Test",
) -> dict[str, float]:
    """Return evaluation metrics dict and print report."""
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc   = roc_auc_score(y, proba)
    ap    = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)

    print(f"\n── {split_name} Results ({'Logistic Regression'}) ──────────────")
    print(f"  ROC AUC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(f"  Brier    : {brier:.4f}")
    print(classification_report(y, pred, target_names=["No Drop", "Drop"]))

    return {"auc": float(auc), "avg_precision": float(ap), "brier": float(brier)}


def save_model(model: CalibratedClassifierCV, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[LR] Saved → {path}")


def load_model(path: str) -> CalibratedClassifierCV:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.dataset import load_dataset

    parquet = sys.argv[1] if len(sys.argv) > 1 else \
        "../../phase1/outputs/phase1_features_match3773386.parquet"

    ds = load_dataset(parquet)
    model = train_logistic(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                           pos_weight=ds.pos_weight())
    metrics = evaluate(model, ds.X_test, ds.y_test)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    save_model(model, os.path.join(out_dir, "lr_baseline.pkl"))

    with open(os.path.join(out_dir, "lr_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)