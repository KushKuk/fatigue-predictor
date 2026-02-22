"""
cognitive_features.py
----------------------
Extracts cognitive fatigue signals from StatsBomb event data.

Features per player per rolling window:
    - decision_latency_mean   : avg seconds from ball receipt to action
    - decision_latency_std    : variance (high = inconsistent decisions)
    - pass_hesitation_index   : passes with duration > median (slow decisions)
    - action_entropy          : Shannon entropy of event-type distribution
    - error_rate              : fraction of events with negative outcome
    - error_rate_trend        : slope of rolling error rate (+ = worsening)
    - pass_accuracy           : pass completion rate in window
    - turnovers               : ball-loss events per window
    - xq_deviation            : placeholder for xThreat deviation (if available)
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ─── Window sizes (seconds) ───────────────────────────────────────────────────
SHORT_WINDOW_S = 60    # 1 minute
LONG_WINDOW_S  = 300   # 5 minutes

# Event types treated as "errors" / negative outcomes
ERROR_EVENT_TYPES = {
    "Miscontrol",
    "Dispossessed",
    "Error",
    "Block",
    "Interception",   # when the intercepting player is the opponent
}

BALL_RECEIPT_TYPE = "Ball Receipt*"
PASS_TYPE         = "Pass"


def _window_cognitive_features(window_df: pd.DataFrame) -> dict:
    """Compute cognitive features for a single player time window."""
    if window_df.empty:
        return {}

    total = len(window_df)

    # ── Decision Latency ────────────────────────────────────────────────────
    receipts = window_df[window_df["event_type"] == BALL_RECEIPT_TYPE].copy()
    receipts = receipts.sort_values("time_seconds")

    latencies = []
    for _, receipt in receipts.iterrows():
        # Find next action by same player after receipt
        subsequent = window_df[
            (window_df["time_seconds"] > receipt["time_seconds"]) &
            (window_df["event_type"] != BALL_RECEIPT_TYPE)
        ]
        if not subsequent.empty:
            next_action = subsequent.iloc[0]
            latency = next_action["time_seconds"] - receipt["time_seconds"]
            if 0 < latency < 10:   # cap to 10s to filter out unrelated events
                latencies.append(latency)

    decision_latency_mean = float(np.mean(latencies)) if latencies else np.nan
    decision_latency_std  = float(np.std(latencies))  if latencies else np.nan

    # ── Pass Hesitation Index ────────────────────────────────────────────────
    passes = window_df[window_df["event_type"] == PASS_TYPE]
    if "duration" in passes.columns and not passes.empty:
        median_dur = passes["duration"].median()
        hesitations = int((passes["duration"] > median_dur * 1.5).sum())
        pass_hesitation_index = hesitations / max(len(passes), 1)
    else:
        pass_hesitation_index = np.nan

    # ── Action Entropy ───────────────────────────────────────────────────────
    type_counts = window_df["event_type"].value_counts(normalize=True)
    action_entropy = float(scipy_entropy(np.array(type_counts.values, dtype=np.float64), base=2)) \
        if len(type_counts) > 1 else 0.0

    # ── Error Rate ───────────────────────────────────────────────────────────
    error_mask = window_df["event_type"].isin(ERROR_EVENT_TYPES)
    error_rate = float(error_mask.sum() / total)

    # ── Pass Accuracy ────────────────────────────────────────────────────────
    if "pass_success" in window_df.columns and not passes.empty:
        pass_accuracy = float(passes["pass_success"].mean())
    else:
        pass_accuracy = np.nan

    # ── Turnovers ────────────────────────────────────────────────────────────
    turnover_types = {"Dispossessed", "Miscontrol"}
    turnovers = int(window_df["event_type"].isin(turnover_types).sum())

    return {
        "decision_latency_mean":  round(decision_latency_mean, 4) if not np.isnan(decision_latency_mean) else np.nan,
        "decision_latency_std":   round(decision_latency_std,  4) if not np.isnan(decision_latency_std)  else np.nan,
        "pass_hesitation_index":  round(pass_hesitation_index, 4) if not np.isnan(pass_hesitation_index) else np.nan,
        "action_entropy":         round(action_entropy, 4),
        "error_rate":             round(error_rate, 4),
        "pass_accuracy":          round(pass_accuracy, 4) if not np.isnan(pass_accuracy) else np.nan,
        "turnovers":              turnovers,
    }


def extract_cognitive_features(
    events_df: pd.DataFrame,
    window_seconds: int = SHORT_WINDOW_S,
    step_seconds: int   = 30,
) -> pd.DataFrame:
    """
    Slide a time window over each player's event stream and compute
    cognitive fatigue features.

    Parameters
    ----------
    events_df      : StatsBomb events DataFrame (from data_loader.load_events)
    window_seconds : rolling window width in seconds
    step_seconds   : slide step in seconds

    Returns
    -------
    DataFrame with (player_id, window_end_ts, feature...) rows.
    """
    records = []

    max_time = events_df["time_seconds"].max()
    players  = events_df["player_id"].dropna().unique()

    for player_id in players:
        player_events = events_df[events_df["player_id"] == player_id].sort_values("time_seconds")

        for window_end in np.arange(window_seconds, max_time + step_seconds, step_seconds):
            window_start = window_end - window_seconds
            window_df = player_events[
                (player_events["time_seconds"] >= window_start) &
                (player_events["time_seconds"] <  window_end)
            ]

            feats = _window_cognitive_features(window_df)
            if feats:
                row = {
                    "player_id":     player_id,
                    "window_end_ts": float(window_end),
                    "window_seconds": window_seconds,
                    **feats,
                }
                records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # ── Error Rate Trend (within player, rolling slope) ──────────────────────
    df = df.sort_values(["player_id", "window_end_ts"])
    df["error_rate_trend"] = (
        df.groupby("player_id")["error_rate"]
          .transform(lambda s: s.rolling(3, min_periods=2).apply(
              lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
          ))
    )

    return df.reset_index(drop=True)


def compute_dual_window_cognitive(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute short (1-min) and long (5-min) cognitive features and merge.
    """
    short = extract_cognitive_features(events_df, SHORT_WINDOW_S, step_seconds=30)
    long_ = extract_cognitive_features(events_df, LONG_WINDOW_S,  step_seconds=60)

    short = short.drop(columns=["window_seconds"])
    long_ = long_.drop(columns=["window_seconds"])

    suffix_cols = [c for c in short.columns if c not in ("player_id", "window_end_ts")]
    short = short.rename(columns={c: f"{c}_1m" for c in suffix_cols})

    suffix_cols = [c for c in long_.columns if c not in ("player_id", "window_end_ts")]
    long_ = long_.rename(columns={c: f"{c}_5m" for c in suffix_cols})

    merged = pd.merge_asof(
        short.sort_values("window_end_ts"),
        long_.sort_values("window_end_ts"),
        on="window_end_ts",
        by="player_id",
        direction="backward",
    )
    return merged


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from data.data_loader import load_events

    # Use match 3773386 (StatsBomb open data — Women's World Cup)
    events = load_events(3773386)
    print("Events loaded:", events.shape)
    print(events["event_type"].value_counts().head(10))

    cog = compute_dual_window_cognitive(events)
    print("\nCognitive features shape:", cog.shape)
    print(cog.head(5).to_string())