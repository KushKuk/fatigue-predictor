"""
dataset.py
----------
Loads the Phase 1 feature parquet, cleans it, and produces
train / validation / test splits ready for model training.

Split strategy:
    - Group by player_id → keep player timelines intact
    - 70% train / 15% val / 15% test  (no temporal leakage)
    - StandardScaler fit on train only
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ── Columns to drop before modelling ──────────────────────────────────────────
NON_FEATURE_COLS = {
    "player_id", "window_end_ts", "window_seconds",
    "drop_label", "future_drop", "risk_score",
    # z-score intermediates (derived, not raw features)
    "z_pass_acc", "z_error_rate", "z_sprint_rate", "composite_z",
}

TARGET_COL    = "future_drop"   # binary: drop in next N windows
RISK_COL      = "risk_score"    # soft target for calibration


@dataclass
class FatigueDataset:
    X_train: np.ndarray
    X_val:   np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_val:   np.ndarray
    y_test:  np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    # soft risk scores (for calibration loss)
    r_train: np.ndarray = field(default_factory=lambda: np.array([]))
    r_val:   np.ndarray = field(default_factory=lambda: np.array([]))
    r_test:  np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def class_weights(self) -> dict[int, float]:
        """Inverse-frequency weights to handle class imbalance."""
        n_pos = int(self.y_train.sum())
        n_neg = len(self.y_train) - n_pos
        total = len(self.y_train)
        return {0: total / (2 * max(n_neg, 1)),
                1: total / (2 * max(n_pos, 1))}

    def pos_weight(self) -> float:
        """BCE pos_weight scalar for PyTorch: n_neg / n_pos."""
        n_pos = int(self.y_train.sum())
        n_neg = len(self.y_train) - n_pos
        return n_neg / max(n_pos, 1)


def load_dataset(
    parquet_path: str,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> FatigueDataset:
    """
    Load Phase 1 parquet → clean → split → scale.

    Parameters
    ----------
    parquet_path : path to phase1_features_match*.parquet
    val_size     : fraction of data for validation
    test_size    : fraction of data for test
    random_state : RNG seed

    Returns
    -------
    FatigueDataset dataclass
    """
    df = pd.read_parquet(parquet_path)

    if verbose:
        print(f"[Dataset] Loaded {len(df)} rows, {df.shape[1]} columns")
        print(f"          Players : {df['player_id'].nunique()}")
        print(f"          Drop rate (future): {df[TARGET_COL].mean():.2%}")

    # ── Drop rows with too many NaNs ──────────────────────────────────────────
    df = df.dropna(thresh=int(df.shape[1] * 0.6)).reset_index(drop=True)

    # ── Feature matrix ────────────────────────────────────────────────────────
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].copy()

    # Drop columns that are entirely NaN (no signal at all)
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols and verbose:
        print(f"[Dataset] Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)
    feat_cols = X.columns.tolist()

    # Fill remaining NaNs: first with column median, then 0 for any still-NaN columns
    col_medians = X.median()
    X = X.fillna(col_medians)
    X = X.fillna(0.0)   # catches columns where median itself was NaN

    # Final safety check
    assert not X.isna().any().any(), "NaNs remain after cleaning — check feature pipeline"

    y      = np.array(df[TARGET_COL].values, dtype=np.int64)
    r      = np.array(df[RISK_COL].values,   dtype=np.float32)
    groups = np.array(df["player_id"].values, dtype=np.int64)

    if verbose:
        print(f"[Dataset] Feature columns : {len(feat_cols)}")
        print(f"[Dataset] Class balance   : {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Group-aware train / temp split ───────────────────────────────────────
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size + val_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(gss_test.split(X, y, groups))

    # Split temp → val / test
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
    )
    val_idx, test_idx = next(
        gss_val.split(X.iloc[temp_idx], y[temp_idx], groups[temp_idx])
    )
    # Map back to global indices
    val_idx  = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    X_arr = X.values.astype(np.float32)

    X_train, y_train, r_train = X_arr[train_idx], y[train_idx], r[train_idx]
    X_val,   y_val,   r_val   = X_arr[val_idx],   y[val_idx],   r[val_idx]
    X_test,  y_test,  r_test  = X_arr[test_idx],  y[test_idx],  r[test_idx]

    # ── Scale (fit on train only) ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    if verbose:
        print(f"[Dataset] Train : {X_train.shape} | pos={y_train.sum()}")
        print(f"[Dataset] Val   : {X_val.shape}   | pos={y_val.sum()}")
        print(f"[Dataset] Test  : {X_test.shape}  | pos={y_test.sum()}")

    return FatigueDataset(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        r_train=r_train, r_val=r_val, r_test=r_test,
        feature_names=feat_cols,
        scaler=scaler,
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # dataset.py lives at: fatigue_predictor/data/dataset.py
    # parquet lives at:    fatigue_predictor/phase1/outputs/...
    _root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default = os.path.join(_root, "phase1", "outputs", "phase1_features_match3773386.parquet")
    path = sys.argv[1] if len(sys.argv) > 1 else _default
    ds = load_dataset(path)
    print("\nPos weight for BCE:", ds.pos_weight())
    print("Class weights     :", ds.class_weights())
    print("n_features        :", ds.n_features)