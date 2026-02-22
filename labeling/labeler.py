"""
labeler.py
-----------
Labels "performance drop" windows for each player.

A drop is flagged when any of the following cross their thresholds
within a rolling window compared to the player's own recent baseline:

    1. Pass accuracy z-score drops below  -ZSCORE_THRESH
    2. Error rate z-score rises above     +ZSCORE_THRESH
    3. Sprint rate z-score drops below    -ZSCORE_THRESH
    4. Combined composite score < -COMPOSITE_THRESH

The label is BINARY:
    1 = performance drop occurred in this window
    0 = no drop

Additionally a soft RISK_SCORE (0-1) is produced for regression targets.
"""

import numpy as np
import pandas as pd


# ─── Thresholds ───────────────────────────────────────────────────────────────
ZSCORE_THRESH     = 1.5    # standard deviations from player's rolling mean
COMPOSITE_THRESH  = 1.0    # composite z-score threshold
BASELINE_WINDOW   = 5      # number of prior windows used to build baseline


def _rolling_zscore(series: pd.Series, window: int = BASELINE_WINDOW) -> pd.Series:
    """Compute z-score relative to rolling mean and std."""
    roll_mean = series.rolling(window, min_periods=2).mean()
    roll_std  = series.rolling(window, min_periods=2).std().replace(0, np.nan)
    return (series - roll_mean) / roll_std


def label_performance_drops(
    cognitive_df: pd.DataFrame,
    physical_df:  pd.DataFrame,
    zscore_thresh:    float = ZSCORE_THRESH,
    composite_thresh: float = COMPOSITE_THRESH,
    baseline_window:  int   = BASELINE_WINDOW,
) -> pd.DataFrame:
    """
    Merge cognitive and physical feature DataFrames and produce drop labels.

    Parameters
    ----------
    cognitive_df : output of cognitive_features.compute_dual_window_cognitive
    physical_df  : output of physical_features.compute_dual_window_features
    zscore_thresh    : z-score threshold for individual signals
    composite_thresh : threshold on composite fatigue score
    baseline_window  : rolling baseline window (number of time steps)

    Returns
    -------
    DataFrame with all features + columns:
        drop_label   (int  0/1)
        risk_score   (float 0-1)
        z_pass_acc   z_error_rate   z_sprint_rate   composite_z
    """

    # ── Merge on player + nearest timestamp ──────────────────────────────────
    cog = cognitive_df.sort_values(["player_id", "window_end_ts"]).copy()
    phy = physical_df.sort_values(["player_id", "window_end_ts"]).copy()

    merged = pd.merge_asof(
        cog,
        phy,
        on="window_end_ts",
        by="player_id",
        direction="nearest",
        tolerance=60,    # allow up to 60s mismatch between windows
    )

    # ── Per-player rolling z-scores ───────────────────────────────────────────
    merged = merged.sort_values(["player_id", "window_end_ts"])

    for player_id, grp_idx in merged.groupby("player_id").groups.items():
        grp = merged.loc[grp_idx]

        # Pass accuracy (lower is worse → negative z = drop)
        if "pass_accuracy_1m" in merged.columns:
            merged.loc[grp_idx, "z_pass_acc"] = _rolling_zscore(
                grp["pass_accuracy_1m"], baseline_window
            ).values

        # Error rate (higher is worse → positive z = drop)
        if "error_rate_1m" in merged.columns:
            merged.loc[grp_idx, "z_error_rate"] = _rolling_zscore(
                grp["error_rate_1m"], baseline_window
            ).values

        # Sprint rate (lower is worse → negative z = fatigue)
        if "sprint_rate_1m" in merged.columns:
            merged.loc[grp_idx, "z_sprint_rate"] = _rolling_zscore(
                grp["sprint_rate_1m"], baseline_window
            ).values

    # Fill NaN z-scores with 0 (no signal yet)
    for col in ["z_pass_acc", "z_error_rate", "z_sprint_rate"]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    # ── Composite z-score ─────────────────────────────────────────────────────
    # Sign convention: positive composite = evidence of drop
    merged["composite_z"] = (
        -merged["z_pass_acc"]      # declining pass acc → positive
        + merged["z_error_rate"]   # rising error rate  → positive
        - merged["z_sprint_rate"]  # declining sprint   → positive
    ) / 3.0

    # ── Binary label ─────────────────────────────────────────────────────────
    drop_pass    = merged["z_pass_acc"]   < -zscore_thresh
    drop_errors  = merged["z_error_rate"] >  zscore_thresh
    drop_sprint  = merged["z_sprint_rate"]< -zscore_thresh
    drop_composite = merged["composite_z"] > composite_thresh

    merged["drop_label"] = (
        (drop_pass | drop_errors | drop_sprint | drop_composite)
    ).astype(int)

    # ── Soft risk score (sigmoid of composite_z) ─────────────────────────────
    merged["risk_score"] = 1 / (1 + np.exp(-merged["composite_z"]))

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_total = len(merged)
    n_drops = merged["drop_label"].sum()
    print(f"[Labeler] Total windows: {n_total} | Drops: {n_drops} "
          f"({100*n_drops/max(n_total,1):.1f}%)")

    return merged.reset_index(drop=True)


def forward_label(
    labeled_df: pd.DataFrame,
    lookahead_steps: int = 3,
) -> pd.DataFrame:
    """
    Create a FUTURE drop label: 1 if a drop occurs within the next
    `lookahead_steps` windows (used for predictive modelling).

    This shifts drop_label backwards so that earlier windows can be
    trained to predict upcoming drops.
    """
    df = labeled_df.sort_values(["player_id", "window_end_ts"]).copy()

    df["future_drop"] = (
        df.groupby("player_id")["drop_label"]
          .transform(lambda s: s.rolling(lookahead_steps, min_periods=1)
                                .max()
                                .shift(-lookahead_steps + 1))
    ).fillna(0).astype(int)

    return df


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.data_loader import load_events, generate_synthetic_tracking
    from features.cognitive_features import compute_dual_window_cognitive
    from features.physical_features  import compute_dual_window_features

    MATCH_ID = 3788741   # StatsBomb open — Women's World Cup
    events   = load_events(MATCH_ID)
    player_ids = events["player_id"].dropna().unique().tolist()
    tracking = generate_synthetic_tracking(MATCH_ID, player_ids, duration_seconds=600)

    cog = compute_dual_window_cognitive(events)
    phy = compute_dual_window_features(tracking)

    labeled = label_performance_drops(cog, phy)
    labeled = forward_label(labeled)

    print("\nLabeled sample:")
    print(labeled[["player_id", "window_end_ts", "composite_z",
                   "drop_label", "future_drop", "risk_score"]].head(10).to_string())
    print("\nFuture drop rate:", labeled["future_drop"].mean())