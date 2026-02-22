"""
data_loader.py
--------------
Loads StatsBomb open event data and synthetic tracking data.
Requires: statsbombpy, pandas, numpy
"""

import pandas as pd
import numpy as np
from statsbombpy import sb


# ─────────────────────────────────────────────
# 1. StatsBomb Event Data
# ─────────────────────────────────────────────

def list_available_competitions() -> pd.DataFrame:
    """Return all free StatsBomb competitions."""
    return pd.DataFrame(sb.competitions(fmt="dataframe"))


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return match list for a competition/season."""
    return pd.DataFrame(sb.matches(competition_id=competition_id, season_id=season_id, fmt="dataframe"))


def load_events(match_id: int) -> pd.DataFrame:
    """
    Load and normalise event data for a single match.

    Returns a flat DataFrame with key columns:
        match_id, period, timestamp, minute, second,
        player_id, player_name, team_name,
        type, outcome, x, y
    """
    raw: pd.DataFrame = pd.DataFrame(sb.events(match_id=match_id, fmt="dataframe"))

    # Flatten location
    if "location" in raw.columns:
        raw[["x", "y"]] = pd.DataFrame(
            raw["location"].apply(
                lambda loc: loc if isinstance(loc, list) else [np.nan, np.nan]
            ).tolist(),
            index=raw.index,
        )

    # Flatten pass outcome
    if "pass_outcome" in raw.columns:
        raw["pass_success"] = raw["pass_outcome"].isna()  # NaN outcome = complete

    cols = [
        "id", "index", "period", "timestamp", "minute", "second",
        "type", "player", "team",
        "x", "y",
        "pass_success",
        "duration",
    ]
    cols = [c for c in cols if c in raw.columns]
    df = raw[cols].copy()

    # Normalise player / team to scalar strings
    df["player_name"] = df["player"].apply(
        lambda v: v.get("name") if isinstance(v, dict) else np.nan
    )
    df["player_id"] = df["player"].apply(
        lambda v: v.get("id") if isinstance(v, dict) else np.nan
    )
    df["team_name"] = df["team"].apply(
        lambda v: v.get("name") if isinstance(v, dict) else np.nan
    )
    df["event_type"] = df["type"].apply(
        lambda v: v.get("name") if isinstance(v, dict) else np.nan
    )
    df["match_id"] = match_id

    df.drop(columns=["player", "team", "type"], inplace=True, errors="ignore")

    # Convert timestamp → seconds elapsed
    df["time_seconds"] = df["minute"] * 60 + df["second"]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. Synthetic Tracking Data Generator
# ─────────────────────────────────────────────

def generate_synthetic_tracking(
    match_id: int,
    player_ids: list,
    duration_seconds: int = 5400,   # 90 minutes
    sample_rate_hz: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate plausible synthetic player tracking data.

    Simulates:
    - Base position drift (players roam their zones)
    - Random sprints (velocity spikes)
    - Fatigue decay (speed decreases over time)

    Returns DataFrame:
        match_id, player_id, timestamp, x, y, velocity, acceleration
    """
    rng = np.random.default_rng(seed)
    records = []

    # Pitch dimensions: 105m x 68m (StatsBomb scale: 120 x 80)
    PITCH_X, PITCH_Y = 120.0, 80.0

    # Assign each player a base zone
    zones = {
        pid: (rng.uniform(10, 110), rng.uniform(5, 75))
        for pid in player_ids
    }

    total_frames = duration_seconds * sample_rate_hz
    dt = 1.0 / sample_rate_hz  # seconds per frame

    for pid in player_ids:
        base_x, base_y = zones[pid]
        x, y = base_x, base_y
        vx, vy = 0.0, 0.0

        for frame in range(total_frames):
            t = frame * dt  # elapsed seconds
            fatigue_factor = max(0.4, 1.0 - (t / duration_seconds) * 0.6)

            # Random walk with zone attraction
            ax = (base_x - x) * 0.01 + rng.normal(0, 2.0) * fatigue_factor
            ay = (base_y - y) * 0.01 + rng.normal(0, 2.0) * fatigue_factor

            # Occasional sprint burst
            if rng.random() < 0.005 * fatigue_factor:
                ax += rng.choice([-1, 1]) * rng.uniform(15, 30)
                ay += rng.choice([-1, 1]) * rng.uniform(5, 15)

            # Damping
            vx = vx * 0.85 + ax * dt
            vy = vy * 0.85 + ay * dt

            x = float(np.clip(x + vx * dt, 0, PITCH_X))
            y = float(np.clip(y + vy * dt, 0, PITCH_Y))

            velocity = float(np.sqrt(vx**2 + vy**2))
            acceleration = float(np.sqrt(ax**2 + ay**2))

            records.append({
                "match_id": match_id,
                "player_id": pid,
                "frame": frame,
                "timestamp": round(t, 3),
                "x": round(x, 3),
                "y": round(y, 3),
                "vx": round(vx, 3),
                "vy": round(vy, 3),
                "velocity": round(velocity, 3),
                "acceleration": round(acceleration, 3),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 3. Convenience: Load a full match bundle
# ─────────────────────────────────────────────

def load_match_bundle(
    match_id: int,
    use_synthetic_tracking: bool = True,
    tracking_sample_rate: int = 10,
) -> dict:
    """
    Returns a dict with keys:
        'events'   → event DataFrame
        'tracking' → tracking DataFrame
    """
    events = load_events(match_id)

    player_ids = events["player_id"].dropna().unique().tolist()

    if use_synthetic_tracking:
        tracking = generate_synthetic_tracking(
            match_id=match_id,
            player_ids=player_ids,
            sample_rate_hz=tracking_sample_rate,
        )
    else:
        raise NotImplementedError(
            "Real tracking ingestion not yet implemented. "
            "Set use_synthetic_tracking=True or supply a tracking DataFrame directly."
        )

    return {"events": events, "tracking": tracking}


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    comps = list_available_competitions()
    print(comps[["competition_id", "season_id", "competition_name"]].head(10))

    # Example: La Liga 2020/21
    COMPETITION_ID = 11
    SEASON_ID = 90
    matches = load_matches(COMPETITION_ID, SEASON_ID)
    MATCH_ID = int(matches["match_id"].iloc[0])
    print(f"\nLoading match {MATCH_ID} …")

    bundle = load_match_bundle(MATCH_ID)
    print("Events shape:", bundle["events"].shape)
    print("Tracking shape:", bundle["tracking"].shape)
    print(bundle["events"].head(3))
    print(bundle["tracking"].head(3))