"""
physical_features.py
---------------------
Extracts physical fatigue signals from player tracking data.

Features computed per player per rolling window:
    - sprint_count          : # velocity bursts above threshold
    - sprint_rate           : sprints per minute
    - mean_velocity         : avg speed in window
    - max_velocity          : peak speed in window
    - distance_covered      : total metres in window
    - hi_effort_ratio       : fraction of frames above high-intensity threshold
    - accel_load            : RMS acceleration (neuromuscular load proxy)
    - cod_count             : direction changes above angle threshold
    - heatmap_drift         : std dev of position from player's mean position
    - fatigue_index         : rolling velocity decline slope (+ = slowing down)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ─── Thresholds (m/s) ────────────────────────────────────────────────────────
SPRINT_THRESHOLD_MS      = 7.0   # ~25 km/h
HIGH_INTENSITY_THRESHOLD = 5.0   # ~18 km/h
COD_ANGLE_THRESHOLD_DEG  = 45    # minimum bearing change for a direction-change event

# ─── Window sizes (seconds) ───────────────────────────────────────────────────
SHORT_WINDOW_S  = 60    # 1 minute
LONG_WINDOW_S   = 300   # 5 minutes


def _compute_bearing(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Direction of travel in degrees [0, 360)."""
    return np.degrees(np.arctan2(vy, vx)) % 360


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angle between two bearing arrays."""
    diff = np.abs(a - b) % 360
    return np.minimum(diff, 360 - diff)


def extract_physical_features(
    tracking_df: pd.DataFrame,
    window_seconds: int = SHORT_WINDOW_S,
    sample_rate_hz: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling physical fatigue features for every player.

    Parameters
    ----------
    tracking_df  : DataFrame with columns [player_id, timestamp, x, y,
                   velocity, acceleration, vx, vy]
    window_seconds : rolling window size in seconds
    sample_rate_hz : tracking sample rate

    Returns
    -------
    DataFrame indexed by (player_id, window_end_timestamp) with feature columns.
    """
    window_frames = window_seconds * sample_rate_hz
    records = []

    for player_id, group in tracking_df.groupby("player_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)

        vel   = np.array(group["velocity"].values,     dtype=np.float64)
        accel = np.array(group["acceleration"].values, dtype=np.float64)
        x     = np.array(group["x"].values,            dtype=np.float64)
        y     = np.array(group["y"].values,            dtype=np.float64)
        vx    = np.array(group["vx"].values,           dtype=np.float64)
        vy    = np.array(group["vy"].values,           dtype=np.float64)
        ts    = np.array(group["timestamp"].values,    dtype=np.float64)

        # Smooth velocity for slope estimation
        vel_smooth = savgol_filter(vel, window_length=min(51, n if n % 2 == 1 else n - 1), polyorder=2) \
            if n > 51 else vel.copy()

        bearing = _compute_bearing(vx, vy)

        step = max(1, window_frames // 10)   # slide every 10% of window

        for end in range(window_frames, n, step):
            start = end - window_frames
            w_vel   = vel[start:end]
            w_accel = accel[start:end]
            w_x     = x[start:end]
            w_y     = y[start:end]
            w_bear  = bearing[start:end]
            w_smooth= vel_smooth[start:end]

            # Sprint detection
            is_sprint = w_vel >= SPRINT_THRESHOLD_MS
            sprint_transitions = np.diff(is_sprint.astype(int))
            sprint_count = int(np.sum(sprint_transitions == 1))

            # Hi-intensity ratio
            hi_effort_ratio = float(np.mean(w_vel >= HIGH_INTENSITY_THRESHOLD))

            # Distance (velocity × dt)
            dt = 1.0 / sample_rate_hz
            distance_covered = float(np.sum(w_vel) * dt)

            # Acceleration load (RMS)
            accel_load = float(np.sqrt(np.mean(w_accel ** 2)))

            # Change-of-direction count
            bear_diff = _angular_diff(w_bear[:-1], w_bear[1:])
            cod_count = int(np.sum(bear_diff >= COD_ANGLE_THRESHOLD_DEG))

            # Heatmap drift (spatial spread)
            heatmap_drift = float(np.std(w_x) + np.std(w_y))

            # Fatigue index: slope of smoothed velocity (positive = declining)
            slope = np.polyfit(np.arange(len(w_smooth)), w_smooth, 1)[0]
            fatigue_index = float(-slope)  # positive when slowing

            window_minutes = window_seconds / 60.0
            sprint_rate = sprint_count / window_minutes

            records.append({
                "player_id":        player_id,
                "window_end_ts":    float(ts[end - 1]),
                "window_seconds":   window_seconds,
                "sprint_count":     sprint_count,
                "sprint_rate":      round(sprint_rate, 4),
                "mean_velocity":    round(float(np.mean(w_vel)), 4),
                "max_velocity":     round(float(w_vel.max()), 4),
                "distance_covered": round(distance_covered, 2),
                "hi_effort_ratio":  round(hi_effort_ratio, 4),
                "accel_load":       round(accel_load, 4),
                "cod_count":        cod_count,
                "heatmap_drift":    round(heatmap_drift, 4),
                "fatigue_index":    round(fatigue_index, 6),
            })

    return pd.DataFrame(records)


def compute_dual_window_features(
    tracking_df: pd.DataFrame,
    sample_rate_hz: int = 10,
) -> pd.DataFrame:
    """
    Compute both short (1-min) and long (5-min) window features
    and merge them into a single wide DataFrame.
    """
    short = extract_physical_features(tracking_df, SHORT_WINDOW_S, sample_rate_hz)
    long_ = extract_physical_features(tracking_df, LONG_WINDOW_S, sample_rate_hz)

    short = short.drop(columns=["window_seconds"])
    long_ = long_.drop(columns=["window_seconds"])

    short = short.rename(columns={c: f"{c}_1m" for c in short.columns
                                   if c not in ("player_id", "window_end_ts")})
    long_ = long_.rename(columns={c: f"{c}_5m" for c in long_.columns
                                   if c not in ("player_id", "window_end_ts")})

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
    from data.data_loader import generate_synthetic_tracking

    tracking = generate_synthetic_tracking(
        match_id=1, player_ids=[101, 102, 103], duration_seconds=600
    )
    print("Tracking sample:")
    print(tracking.head(3))

    features = compute_dual_window_features(tracking)
    print("\nPhysical features sample:")
    print(features.head(3))
    print("Shape:", features.shape)
    print("Columns:", features.columns.tolist())