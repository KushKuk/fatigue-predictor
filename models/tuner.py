"""
tuner.py
--------
Optuna-based hyperparameter search for FatigueTransformer.

Search space:
    d_model      : [64, 128, 256]
    n_heads      : [2, 4, 8]  (must divide d_model)
    n_layers     : [2, 3, 4]
    ffn_dim      : [128, 256, 512]
    dropout      : [0.1, 0.4]
    lr           : [1e-5, 1e-3]  (log scale)
    seq_len      : [8, 16, 32]
    batch_size   : [32, 64, 128]

Objective: maximise validation AUC.
"""

from __future__ import annotations

import os
import sys
import optuna
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from models.transformer import FatigueTransformer, TransformerDataset
from models.trainer     import train_transformer, get_probabilities


def objective(
    trial:       "optuna.Trial",
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    n_features:  int,
    pos_weight:  float,
    device:      str = "cpu",
    max_epochs:  int = 30,
) -> float:
    """Single Optuna trial — returns validation AUC."""
    d_model    = trial.suggest_categorical("d_model",    [64, 128, 256])
    n_heads_choices = [h for h in [2, 4, 8] if d_model % h == 0]
    n_heads    = trial.suggest_categorical("n_heads",    n_heads_choices)
    n_layers   = trial.suggest_int("n_layers",           2, 4)
    ffn_dim    = trial.suggest_categorical("ffn_dim",    [128, 256, 512])
    dropout    = trial.suggest_float("dropout",          0.1, 0.4)
    lr         = trial.suggest_float("lr",               1e-5, 1e-3, log=True)
    seq_len    = trial.suggest_categorical("seq_len",    [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_ds = TransformerDataset(X_train, y_train, seq_len)
    val_ds   = TransformerDataset(X_val,   y_val,   seq_len)

    if len(train_ds) < batch_size or len(val_ds) == 0:
        return 0.5   # invalid config

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = FatigueTransformer(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )

    train_transformer(
        model, train_loader, val_loader,
        pos_weight=pos_weight,
        lr=lr,
        epochs=max_epochs,
        patience=5,
        device=device,
        verbose=False,
    )

    proba, y_true = get_probabilities(model, val_loader, device)

    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, proba))


def run_tuning(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    n_features: int,
    pos_weight: float,
    n_trials:   int  = 20,
    device:     str  = "cpu",
    max_epochs: int  = 30,
    verbose:    bool = True,
) -> dict:
    """
    Run Optuna hyperparameter search.

    Returns best_params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val,
            n_features, pos_weight, device, max_epochs,
        ),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    best = study.best_params
    if verbose:
        print(f"\n[Tuner] Best AUC  : {study.best_value:.4f}")
        print(f"[Tuner] Best params: {best}")

    return best