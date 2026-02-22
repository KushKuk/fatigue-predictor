"""
pipeline.py
-----------
Phase 2 end-to-end pipeline:

    1. Load Phase 1 feature dataset
    2. Train Logistic Regression baseline
    3. Train LSTM baseline
    4. Evaluate and compare both models
    5. Save models, metrics, and plots

Usage:
    python pipeline.py --parquet ../../phase1/outputs/phase1_features_match3788741.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BASE_DIR)

DEFAULT_PARQUET = os.path.join(ROOT_DIR, "phase1", "outputs", "phase1_features_match3773386.parquet")
DEFAULT_OUT     = os.path.join(BASE_DIR, "outputs")

from data.dataset                 import load_dataset
from models.logistic_baseline     import train_logistic, evaluate as eval_lr, save_model as save_lr
from models.lstm_baseline         import (
    FatigueLSTM, make_loaders, train_lstm,
    evaluate_lstm, save_lstm,
)
from evaluation.evaluator         import (
    compute_metrics, compare_models,
    plot_roc_curves, plot_calibration,
    plot_training_history, early_detection_rate,
)


def run_pipeline(
    parquet_path: str,
    out_dir:      str,
    seq_len:      int   = 10,
    batch_size:   int   = 64,
    lstm_epochs:  int   = 50,
    device_str:   str   = "cpu",
    verbose:      bool  = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.time()-t0:6.1f}s] {msg}")

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    log(f"Device: {device}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    log("Loading Phase 1 dataset …")
    ds = load_dataset(parquet_path, verbose=verbose)

    # ── 2. Logistic Regression ────────────────────────────────────────────────
    log("Training Logistic Regression baseline …")
    lr_model = train_logistic(
        ds.X_train, ds.y_train,
        ds.X_val,   ds.y_val,
        pos_weight=ds.pos_weight(),
        verbose=verbose,
    )
    save_lr(lr_model, os.path.join(out_dir, "lr_baseline.pkl"))

    lr_proba_test = lr_model.predict_proba(ds.X_test)[:, 1]
    lr_metrics    = compute_metrics(ds.y_test, lr_proba_test, name="Logistic Regression")
    lr_edr        = early_detection_rate(ds.y_test, lr_proba_test)
    lr_metrics["early_detection_rate"] = round(lr_edr, 4)
    log(f"LR  AUC={lr_metrics['auc']:.4f}  EDR={lr_edr:.2%}")

    # ── 3. LSTM ───────────────────────────────────────────────────────────────
    log("Building sequence loaders …")
    train_loader, val_loader, test_loader = make_loaders(
        ds.X_train, ds.y_train,
        ds.X_val,   ds.y_val,
        ds.X_test,  ds.y_test,
        seq_len=seq_len, batch_size=batch_size,
    )

    log("Training LSTM baseline …")
    lstm_model = FatigueLSTM(n_features=ds.n_features)
    lstm_history = train_lstm(
        lstm_model, train_loader, val_loader,
        pos_weight=ds.pos_weight(),
        epochs=lstm_epochs,
        device=str(device),
        verbose=verbose,
    )
    save_lstm(lstm_model, os.path.join(out_dir, "lstm_baseline.pt"))
    plot_training_history(lstm_history, os.path.join(out_dir, "lstm_training_history.png"), "LSTM Training History")

    # Collect LSTM probabilities on test set
    lstm_model.eval()
    lstm_probas: list[float] = []
    lstm_labels: list[int]   = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = lstm_model(xb.to(device))
            lstm_probas.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            lstm_labels.extend(yb.numpy().astype(int).tolist())

    lstm_proba_arr = np.array(lstm_probas)
    lstm_label_arr = np.array(lstm_labels)

    lstm_metrics = compute_metrics(lstm_label_arr, lstm_proba_arr, name="LSTM")
    lstm_edr     = early_detection_rate(lstm_label_arr, lstm_proba_arr)
    lstm_metrics["early_detection_rate"] = round(lstm_edr, 4)
    log(f"LSTM AUC={lstm_metrics['auc']:.4f}  EDR={lstm_edr:.2%}")

    # ── 4. Compare ────────────────────────────────────────────────────────────
    all_results = [lr_metrics, lstm_metrics]
    compare_models(all_results, save_path=os.path.join(out_dir, "model_comparison.json"))

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    # ROC — align lengths (LSTM test set may differ due to seq windowing)
    min_len = min(len(lr_proba_test), len(lstm_proba_arr))

    plot_roc_curves(
        [
            (ds.y_test[:min_len],  lr_proba_test[:min_len],  "Logistic Regression"),
            (lstm_label_arr[:min_len], lstm_proba_arr[:min_len], "LSTM"),
        ],
        save_path=os.path.join(out_dir, "roc_curves.png"),
    )

    plot_calibration(
        [
            (ds.y_test[:min_len],  lr_proba_test[:min_len],  "Logistic Regression"),
            (lstm_label_arr[:min_len], lstm_proba_arr[:min_len], "LSTM"),
        ],
        save_path=os.path.join(out_dir, "calibration_curves.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n── Phase 2 Summary ──────────────────────────────────────────────────")
    log(f"  Target AUC (>0.75) : {'✓' if max(lr_metrics['auc'], lstm_metrics['auc']) > 0.75 else '✗'}")
    log(f"  Best AUC           : {max(lr_metrics['auc'], lstm_metrics['auc']):.4f}")
    log(f"  Output dir         : {out_dir}")
    log("─────────────────────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fatigue Predictor — Phase 2 Pipeline")
    parser.add_argument("--parquet",     type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--out_dir",     type=str, default=DEFAULT_OUT)
    parser.add_argument("--seq_len",     type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--lstm_epochs", type=int, default=50)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    run_pipeline(
        parquet_path=args.parquet,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lstm_epochs=args.lstm_epochs,
        device_str=args.device,
        verbose=not args.quiet,
    )