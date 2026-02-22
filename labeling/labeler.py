"""
labeler.py
-----------
Labels "performance drop" windows for each player using
real StatsBomb event data features only (no tracking required).

Signals used:
    From cognitive_features : pass_accuracy_1m, error_rate_1m, decision_latency_mean_1m
    From physical_features  : hi_intensity_rate_1m, fatigue_index_1m, event_rate_1m

A drop is flagged when per-player rolling z-scores cross thresholds,
or when the composite fatigue score exceeds COMPOSITE_THRESH.

Labels:
    drop_label  (int  0/1) - drop occurring in this window
    future_drop (int  0/1) - drop occurring in next N windows
    risk_score  (float 0-1) - sigmoid of composite z-score
"""

import numpy as np
import pandas as pd


ZSCORE_THRESH    = 1.5
COMPOSITE_THRESH = 1.0
BASELINE_WINDOW  = 5


def _rolling_zscore(series: pd.Series, window: int = BASELINE_WINDOW) -> pd.Series:
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
    Merge cognitive + physical feature DataFrames and produce drop labels.
    Uses only event-derived features (no tracking/sprint data).
    """

    # merge_asof requires global sort on the key column
    cog = cognitive_df.sort_values("window_end_ts").copy()
    phy = physical_df.sort_values("window_end_ts").copy()

    merged = pd.merge_asof(
        cog,
        phy,
        on="window_end_ts",
        by="player_id",
        direction="nearest",
        tolerance=60,
    )

    merged = merged.sort_values(["player_id", "window_end_ts"])

    # ── Per-player rolling z-scores ───────────────────────────────────────────
    # Map each signal to its column name and drop direction
    # (True = higher value = worse performance)
    signal_map = {
        "z_pass_acc":      ("pass_accuracy_1m",         False),  # lower = worse
        "z_error_rate":    ("error_rate_1m",             True),   # higher = worse
        "z_latency":       ("decision_latency_mean_1m",  True),   # higher = worse
        "z_hi_intensity":  ("hi_intensity_rate_1m",      False),  # lower = worse
        "z_fatigue_idx":   ("fatigue_index_1m",          True),   # higher = worse
    }

    for z_col, (src_col, higher_is_worse) in signal_map.items():
        if src_col not in merged.columns:
            merged[z_col] = 0.0
            continue
        for player_id, grp_idx in merged.groupby("player_id").groups.items():
            grp = merged.loc[grp_idx]
            z   = _rolling_zscore(grp[src_col], baseline_window)
            merged.loc[grp_idx, z_col] = z.values

    for z_col in signal_map:
        if z_col not in merged.columns:
            merged[z_col] = 0.0
        merged[z_col] = merged[z_col].fillna(0.0)

    # ── Composite z-score ─────────────────────────────────────────────────────
    # Positive composite = evidence of performance drop
    merged["composite_z"] = (
        -merged["z_pass_acc"]       # declining pass acc  -> positive
        + merged["z_error_rate"]    # rising error rate   -> positive
        + merged["z_latency"]       # rising latency      -> positive
        - merged["z_hi_intensity"]  # declining intensity -> positive
        + merged["z_fatigue_idx"]   # rising fatigue idx  -> positive
    ) / 5.0

    # ── Binary label ──────────────────────────────────────────────────────────
    drop_pass      = merged["z_pass_acc"]      < -zscore_thresh
    drop_errors    = merged["z_error_rate"]    >  zscore_thresh
    drop_latency   = merged["z_latency"]       >  zscore_thresh
    drop_intensity = merged["z_hi_intensity"]  < -zscore_thresh
    drop_composite = merged["composite_z"]     >  composite_thresh

    merged["drop_label"] = (
        drop_pass | drop_errors | drop_latency | drop_intensity | drop_composite
    ).astype(int)

    # ── Soft risk score ───────────────────────────────────────────────────────
    merged["risk_score"] = 1 / (1 + np.exp(-merged["composite_z"]))

    n_total = len(merged)
    n_drops = int(merged["drop_label"].sum())
    print(f"[Labeler] Total windows: {n_total} | Drops: {n_drops} "
          f"({100 * n_drops / max(n_total, 1):.1f}%)")

    return merged.reset_index(drop=True)


def forward_label(
    labeled_df:      pd.DataFrame,
    lookahead_steps: int = 3,
) -> pd.DataFrame:
    """
    Shift drop_label back by lookahead_steps so earlier windows
    can be trained to predict upcoming drops.
    """
    df = labeled_df.sort_values(["player_id", "window_end_ts"]).copy()

    df["future_drop"] = (
        df.groupby("player_id")["drop_label"]
          .transform(lambda s: s.rolling(lookahead_steps, min_periods=1)
                                .max()
                                .shift(-lookahead_steps + 1))
    ).fillna(0).astype(int)

    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.data_loader            import load_events
    from features.physical_features  import compute_dual_window_features
    from features.cognitive_features import compute_dual_window_cognitive

    events  = load_events(3773386)
    phy     = compute_dual_window_features(events)
    cog     = compute_dual_window_cognitive(events)
    labeled = label_performance_drops(cog, phy)
    labeled = forward_label(labeled)

    print(labeled[["player_id", "window_end_ts", "composite_z",
                   "drop_label", "future_drop", "risk_score"]].head(10).to_string())
    print("Future drop rate:", labeled["future_drop"].mean())