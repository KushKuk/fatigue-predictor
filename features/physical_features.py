"""
physical_features.py
---------------------
Derives physical fatigue proxies from StatsBomb EVENT data only.
(No tracking data required.)

Since StatsBomb open data has no raw tracking, we approximate physical
load from event-level signals available in the data:

Features per player per rolling window:
    - event_rate          : total events per minute (activity proxy)
    - carry_rate          : carries per minute (physical load proxy)
    - duel_rate           : duels per minute (contact/exertion proxy)
    - press_rate          : pressures per minute (high-intensity proxy)
    - distance_proxy      : sum of pass lengths in window (if available)
    - action_density      : events in window / window duration
    - position_spread_x   : std dev of x locations (spatial range)
    - position_spread_y   : std dev of y locations
    - avg_x               : mean x position (pitch zone)
    - avg_y               : mean y position
    - fatigue_index       : decline in event rate vs player's own 1st-half avg
"""

import numpy as np
import pandas as pd


SHORT_WINDOW_S = 60     # 1 minute
LONG_WINDOW_S  = 300    # 5 minutes

HIGH_INTENSITY_TYPES = {
    "Pressure", "Duel", "Carry", "Ball Recovery",
    "Clearance", "Block", "Interception",
}


def _window_physical(window_df: pd.DataFrame, window_seconds: int) -> dict:
    """Compute physical proxy features for one player time window."""
    if window_df.empty:
        return {}

    total        = len(window_df)
    window_mins  = window_seconds / 60.0

    event_rate   = total / window_mins

    carry_rate   = window_df["event_type"].eq("Carry").sum()       / window_mins
    duel_rate    = window_df["event_type"].eq("Duel").sum()        / window_mins
    press_rate   = window_df["event_type"].eq("Pressure").sum()    / window_mins
    hi_rate      = window_df["event_type"].isin(HIGH_INTENSITY_TYPES).sum() / window_mins

    # Spatial spread from location data
    locs = window_df.dropna(subset=["x", "y"])
    position_spread_x = float(locs["x"].std()) if len(locs) > 1 else 0.0
    position_spread_y = float(locs["y"].std()) if len(locs) > 1 else 0.0
    avg_x = float(locs["x"].mean()) if not locs.empty else np.nan
    avg_y = float(locs["y"].mean()) if not locs.empty else np.nan

    # Pass length sum as distance proxy
    if "pass_length" in window_df.columns:
        distance_proxy = float(window_df["pass_length"].sum())
    else:
        distance_proxy = np.nan

    return {
        "event_rate":         round(event_rate,         4),
        "carry_rate":         round(carry_rate,         4),
        "duel_rate":          round(duel_rate,          4),
        "press_rate":         round(press_rate,         4),
        "hi_intensity_rate":  round(hi_rate,            4),
        "distance_proxy":     round(distance_proxy, 2) if not np.isnan(distance_proxy) else np.nan,
        "position_spread_x":  round(position_spread_x, 4),
        "position_spread_y":  round(position_spread_y, 4),
        "avg_x":              round(avg_x, 3) if not np.isnan(avg_x) else np.nan,
        "avg_y":              round(avg_y, 3) if not np.isnan(avg_y) else np.nan,
    }


def extract_physical_features(
    events_df:      pd.DataFrame,
    window_seconds: int = SHORT_WINDOW_S,
    step_seconds:   int = 30,
) -> pd.DataFrame:
    """
    Slide a time window over each player's event stream and compute
    physical proxy features.

    Parameters
    ----------
    events_df      : cleaned events DataFrame from data_loader.load_events
    window_seconds : rolling window width in seconds
    step_seconds   : slide step in seconds

    Returns
    -------
    DataFrame with (player_id, window_end_ts, feature...) rows.
    """
    records  = []
    max_time = float(events_df["time_seconds"].max())
    players  = events_df["player_id"].dropna().unique()

    # Per-player first-half event rate (baseline for fatigue_index)
    first_half = events_df[events_df["period"] == 1]
    first_half_duration = float(first_half["time_seconds"].max()) or 2700.0
    baseline_rates: dict[float, float] = {}
    for pid in players:
        p_fh = first_half[first_half["player_id"] == pid]
        baseline_rates[float(pid)] = len(p_fh) / (first_half_duration / 60.0)

    for player_id in players:
        pid_f    = float(player_id)
        p_events = events_df[events_df["player_id"] == player_id].sort_values("time_seconds")
        baseline = baseline_rates.get(pid_f, 1.0)

        for window_end in np.arange(window_seconds, max_time + step_seconds, step_seconds):
            window_start = window_end - window_seconds
            window_df    = p_events[
                (p_events["time_seconds"] >= window_start) &
                (p_events["time_seconds"] <  window_end)
            ]

            feats = _window_physical(window_df, window_seconds)
            if not feats:
                continue

            # Fatigue index: how much has event rate dropped vs first-half baseline
            fatigue_index = float(baseline - feats["event_rate"]) / max(baseline, 1.0)

            records.append({
                "player_id":      float(player_id),
                "window_end_ts":  float(window_end),
                "window_seconds": window_seconds,
                "fatigue_index":  round(fatigue_index, 4),
                **feats,
            })

    return pd.DataFrame(records)


def compute_dual_window_features(
    events_df:      pd.DataFrame,
    sample_rate_hz: int = 10,   # kept for API compat, unused
) -> pd.DataFrame:
    """
    Compute short (1-min) and long (5-min) physical features and merge.
    """
    short = extract_physical_features(events_df, SHORT_WINDOW_S, step_seconds=30)
    long_ = extract_physical_features(events_df, LONG_WINDOW_S,  step_seconds=60)

    short = short.drop(columns=["window_seconds"])
    long_ = long_.drop(columns=["window_seconds"])

    suffix_cols_s = [c for c in short.columns if c not in ("player_id", "window_end_ts")]
    suffix_cols_l = [c for c in long_.columns  if c not in ("player_id", "window_end_ts")]

    short = short.rename(columns={c: f"{c}_1m" for c in suffix_cols_s})
    long_ = long_.rename(columns={c: f"{c}_5m" for c in suffix_cols_l})

    merged = pd.merge_asof(
        short.sort_values("window_end_ts"),
        long_.sort_values("window_end_ts"),
        on="window_end_ts",
        by="player_id",
        direction="backward",
    )
    return merged


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from data.data_loader import load_events

    events = load_events(3773386)
    print("Events loaded:", events.shape)

    features = compute_dual_window_features(events)
    print("Physical features shape:", features.shape)
    print(features.dropna().head(5).to_string())