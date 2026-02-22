"""
pipeline.py
-----------
Phase 3 end-to-end pipeline:

    1. Load Phase 1 feature dataset
    2. Optional: Optuna hyperparameter search
    3. Train Spatio-Temporal Transformer
    4. Optional: SHAP feature importance
    5. Compare against Phase 2 baselines
    6. Save model, metrics, plots

Run from fatigue_predictor/ root:
    python phase3/pipeline.py
    python phase3/pipeline.py --tune --n_trials 20
    python phase3/pipeline.py --survival

Notes on data requirements:
    The Transformer benefits significantly from more players.
    With ~32 players you get ~1 000 training sequences — marginal for
    attention-based models.  Aim for 100+ players / 10 000+ sequences
    before expecting AUC > 0.75.  Use --tune once you have more data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BASE_DIR)

from data.dataset         import load_dataset
from models.transformer   import (
    FatigueTransformer, make_transformer_loaders, save_transformer,
)
from models.trainer       import train_transformer, get_probabilities
from models.tuner         import run_tuning
from evaluation.evaluator import (
    compute_metrics, compare_models,
    plot_roc_curves, plot_calibration,
    plot_training_history, early_detection_rate,
)

DEFAULT_PARQUET = os.path.join(
    ROOT_DIR, "phase1", "outputs", "phase1_features_match3773386.parquet"
)
DEFAULT_OUT = os.path.join(BASE_DIR, "outputs")

# ── Default Transformer hyperparams ───────────────────────────────────────────
DEFAULT_HP = {
    "d_model":    64,    # keep small relative to dataset size
    "n_heads":    4,
    "n_layers":   2,
    "ffn_dim":    128,
    "dropout":    0.1,
    "seq_len":    8,     # shorter = more sequences from limited data
    "batch_size": 32,
    "lr":         3e-4,
}


def _loader_len(loader: DataLoader) -> int:
    """Return number of samples in a DataLoader without triggering type errors."""
    dataset = loader.dataset
    if hasattr(dataset, "__len__"):
        return len(dataset)  # type: ignore[arg-type]
    return len(loader) * int(loader.batch_size or 1)


def run_pipeline(
    parquet_path: str,
    out_dir:      str,
    tune:         bool       = False,
    n_trials:     int        = 20,
    survival:     bool       = False,
    use_shap:     bool       = False,
    epochs:       int        = 80,
    device_str:   str        = "cpu",
    verbose:      bool       = True,
    hp_override:  dict | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.time()-t0:6.1f}s] {msg}")

    device = torch.device(
        device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    log(f"Device: {device}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    log("Loading dataset …")
    ds = load_dataset(parquet_path, verbose=verbose)

    if verbose:
        print()
        ds.summary()
        print()

    # Warn if dataset is likely too small for a Transformer to shine
    n_train_players = np.unique(ds.player_ids_train).size
    if n_train_players < 50:
        log(f"⚠  Only {n_train_players} training players — Transformer may underperform "
            f"vs simpler baselines.  Aim for 100+ players for best results.")

    # ── 2. Hyperparameter search (optional) ───────────────────────────────────
    hp = {**DEFAULT_HP, **(hp_override or {})}

    if tune:
        log(f"Running Optuna search ({n_trials} trials) …")
        best_params = run_tuning(
            ds.X_train, ds.y_train,
            ds.X_val,   ds.y_val,
            n_features=ds.n_features,
            pos_weight=ds.pos_weight(),
            n_trials=n_trials,
            device=str(device),
            verbose=verbose,
        )
        hp.update(best_params)
        log(f"Best HP: {hp}")
        with open(os.path.join(out_dir, "best_hp.json"), "w") as f:
            json.dump(hp, f, indent=2)

    # ── 3. Build data loaders ─────────────────────────────────────────────────
    seq_len    = int(hp["seq_len"])
    batch_size = int(hp["batch_size"])

    train_loader, val_loader, test_loader = make_transformer_loaders(
        ds.X_train, ds.y_train,
        ds.X_val,   ds.y_val,
        ds.X_test,  ds.y_test,
        seq_len=seq_len,
        batch_size=batch_size,
        # Pass player IDs so windows never cross player boundaries
        player_ids_train=ds.player_ids_train,
        player_ids_val=ds.player_ids_val,
        player_ids_test=ds.player_ids_test,
    )

    log(f"Sequences — train: {_loader_len(train_loader)} | "
        f"val: {_loader_len(val_loader)} | test: {_loader_len(test_loader)}")

    if _loader_len(train_loader) < batch_size:
        raise RuntimeError(
            f"Training set has fewer sequences ({_loader_len(train_loader)}) "
            f"than batch_size ({batch_size}).  Reduce seq_len or batch_size, "
            f"or add more data."
        )

    # ── 4. Build model ────────────────────────────────────────────────────────
    model = FatigueTransformer(
        n_features=ds.n_features,
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ffn_dim=int(hp["ffn_dim"]),
        dropout=float(hp["dropout"]),
        survival=survival,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    # Sanity check: parameters vs sequences ratio
    n_seqs = _loader_len(train_loader)
    ratio  = n_params / max(n_seqs, 1)
    if ratio > 100:
        log(f"⚠  {n_params:,} params / {n_seqs} sequences = {ratio:.0f}× ratio. "
            f"High overfitting risk — consider smaller d_model/n_layers or more data.")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    log("Training Spatio-Temporal Transformer …")
    history = train_transformer(
        model, train_loader, val_loader,
        pos_weight=ds.pos_weight(),
        lr=float(hp["lr"]),
        epochs=epochs,
        patience=20,        # generous patience to survive LR warmup
        warmup_epochs=3,
        device=str(device),
        verbose=verbose,
    )

    save_transformer(model, os.path.join(out_dir, "transformer.pt"))
    plot_training_history(
        history,
        os.path.join(out_dir, "transformer_training.png"),
        title="Transformer Training History",
    )

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    log("Evaluating on test set …")
    proba, y_true = get_probabilities(model, test_loader, str(device))

    tf_metrics = compute_metrics(y_true, proba, name="Transformer")
    tf_edr     = early_detection_rate(y_true, proba)
    tf_metrics["early_detection_rate"] = round(tf_edr, 4)

    log(f"Transformer AUC={tf_metrics['auc']:.4f} | "
        f"F1={tf_metrics['f1']:.4f} | "
        f"EDR={tf_edr:.2%}")

    # ── 7. Load Phase 2 results for comparison ────────────────────────────────
    all_results   = [tf_metrics]
    p2_comparison = os.path.join(ROOT_DIR, "phase2", "outputs", "model_comparison.json")
    if os.path.exists(p2_comparison):
        with open(p2_comparison) as f:
            p2_results = json.load(f)
        all_results = p2_results + [tf_metrics]

    compare_models(
        all_results,
        save_path=os.path.join(out_dir, "model_comparison.json"),
    )

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_roc_curves(
        [(y_true, proba, "Transformer")],
        save_path=os.path.join(out_dir, "roc_transformer.png"),
    )
    plot_calibration(
        [(y_true, proba, "Transformer")],
        save_path=os.path.join(out_dir, "calibration_transformer.png"),
    )

    # ── 9. SHAP (optional) ────────────────────────────────────────────────────
    if use_shap:
        log("Computing SHAP feature importance …")
        try:
            from models.explainability import (
                compute_shap_values, plot_feature_importance, print_top_features,
            )
            mean_shap = compute_shap_values(
                model, ds.X_train, ds.X_test,
                seq_len=seq_len, device=str(device),
            )
            print_top_features(mean_shap, ds.feature_names)
            plot_feature_importance(
                mean_shap, ds.feature_names,
                save_path=os.path.join(out_dir, "shap_importance.png"),
            )
            np.save(os.path.join(out_dir, "shap_values.npy"), mean_shap)
        except Exception as e:
            log(f"SHAP skipped: {e}")

    # ── 10. Save metrics ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "transformer_metrics.json"), "w") as f:
        json.dump(tf_metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    target_met = tf_metrics["auc"] > 0.75
    log("\n── Phase 3 Summary ──────────────────────────────────────────────────")
    log(f"  AUC              : {tf_metrics['auc']:.4f}  "
        f"{'✓ > 0.75 target' if target_met else '✗ below 0.75 target'}")
    log(f"  F1               : {tf_metrics['f1']:.4f}")
    log(f"  Early Detection  : {tf_edr:.2%}")
    log(f"  Brier Score      : {tf_metrics['brier']:.4f}")
    log(f"  Training players : {n_train_players}")
    log(f"  Output dir       : {out_dir}")
    if not target_met and n_train_players < 50:
        log("  → Adding more match data is likely the highest-leverage next step.")
    log("─────────────────────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 — Transformer Pipeline")
    parser.add_argument("--parquet",    type=str,  default=DEFAULT_PARQUET)
    parser.add_argument("--out_dir",    type=str,  default=DEFAULT_OUT)
    parser.add_argument("--tune",       action="store_true", help="Run Optuna HP search")
    parser.add_argument("--n_trials",   type=int,  default=20)
    parser.add_argument("--survival",   action="store_true", help="Add survival head")
    parser.add_argument("--shap",       action="store_true", help="Compute SHAP values")
    parser.add_argument("--epochs",     type=int,  default=80)
    parser.add_argument("--device",     type=str,  default="cpu")
    parser.add_argument("--d_model",    type=int,  default=DEFAULT_HP["d_model"])
    parser.add_argument("--n_heads",    type=int,  default=DEFAULT_HP["n_heads"])
    parser.add_argument("--n_layers",   type=int,  default=DEFAULT_HP["n_layers"])
    parser.add_argument("--ffn_dim",    type=int,  default=DEFAULT_HP["ffn_dim"])
    parser.add_argument("--dropout",    type=float,default=DEFAULT_HP["dropout"])
    parser.add_argument("--seq_len",    type=int,  default=DEFAULT_HP["seq_len"])
    parser.add_argument("--batch_size", type=int,  default=DEFAULT_HP["batch_size"])
    parser.add_argument("--lr",         type=float,default=DEFAULT_HP["lr"])
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    hp_override = {
        "d_model":    args.d_model,
        "n_heads":    args.n_heads,
        "n_layers":   args.n_layers,
        "ffn_dim":    args.ffn_dim,
        "dropout":    args.dropout,
        "seq_len":    args.seq_len,
        "batch_size": args.batch_size,
        "lr":         args.lr,
    }

    run_pipeline(
        parquet_path = args.parquet,
        out_dir      = args.out_dir,
        tune         = args.tune,
        n_trials     = args.n_trials,
        survival     = args.survival,
        use_shap     = args.shap,
        epochs       = args.epochs,
        device_str   = args.device,
        verbose      = not args.quiet,
        hp_override  = hp_override,
    )