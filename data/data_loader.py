"""
data_loader.py
--------------
Loads StatsBomb open event data.
Requires: statsbombpy>=1.16.0, pandas, numpy

Column reality in statsbombpy 1.16.0 (fmt="dataframe"):
  - 'player_id'  : int   (NaN for non-player events like Starting XI)
  - 'player'     : str   (player name, NaN for non-player events)
  - 'team_id'    : int
  - 'team'       : str
  - 'type'       : str   (event type name, already a string)
  - 'location'   : list  [x, y] or NaN
  - 'pass_outcome': str  (NaN = successful pass)
  - 'duration'   : float
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

    Returns a flat DataFrame with columns:
        match_id, period, timestamp, time_seconds, minute, second,
        player_id, player_name, team_id, team_name,
        event_type, x, y, pass_success, duration
    """
    raw: pd.DataFrame = pd.DataFrame(sb.events(match_id=match_id, fmt="dataframe"))

    df = pd.DataFrame()

    # ── IDs & timing ──────────────────────────────────────────────────────────
    df["match_id"]     = match_id
    df["event_id"]     = raw["id"]
    df["index"]        = raw["index"]
    df["period"]       = raw["period"]
    df["timestamp"]    = raw["timestamp"]
    df["minute"]       = raw["minute"]
    df["second"]       = raw["second"]
    df["time_seconds"] = raw["minute"] * 60 + raw["second"]

    # ── Player & team ─────────────────────────────────────────────────────────
    # statsbombpy 1.16 already flattens these to scalar columns
    df["player_id"]   = pd.to_numeric(raw["player_id"],  errors="coerce")
    df["player_name"] = raw["player"].astype(str).where(raw["player"].notna(), other=np.nan)
    df["team_id"]     = pd.to_numeric(raw["team_id"],    errors="coerce")
    df["team_name"]   = raw["team"].astype(str).where(raw["team"].notna(),     other=np.nan)

    # ── Event type (already a plain string in 1.16) ───────────────────────────
    df["event_type"]  = raw["type"].astype(str).where(raw["type"].notna(), other=np.nan)

    # ── Location ──────────────────────────────────────────────────────────────
    locs = raw["location"].apply(
        lambda v: v if isinstance(v, list) and len(v) >= 2 else [np.nan, np.nan]
    ).tolist()
    loc_df = pd.DataFrame(locs, columns=["x", "y"], index=raw.index)
    df["x"] = loc_df["x"].values
    df["y"] = loc_df["y"].values

    # ── Pass success (NaN outcome = completed) ────────────────────────────────
    if "pass_outcome" in raw.columns:
        df["pass_success"] = raw["pass_outcome"].isna()
    else:
        df["pass_success"] = np.nan

    # ── Duration ──────────────────────────────────────────────────────────────
    if "duration" in raw.columns:
        df["duration"] = pd.to_numeric(raw["duration"], errors="coerce")
    else:
        df["duration"] = np.nan

    # ── Duel outcome for error detection ──────────────────────────────────────
    if "duel_outcome" in raw.columns:
        df["duel_outcome"] = raw["duel_outcome"]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. Convenience: Load a full match bundle
# ─────────────────────────────────────────────

def load_match_bundle(match_id: int) -> dict:
    """
    Returns a dict with key:
        'events' -> cleaned event DataFrame

    Uses real StatsBomb event data only — no synthetic tracking.
    Physical signals are derived from event-level data.
    """
    events = load_events(match_id)

    player_ids = events["player_id"].dropna().unique()
    n_players  = len(player_ids)
    n_events   = len(events)

    print(f"  Players found : {n_players}")
    print(f"  Events loaded : {n_events}")
    print(f"  Event types   : {events['event_type'].value_counts().head(5).to_dict()}")

    if n_players == 0:
        raise ValueError(
            f"No player IDs found for match {match_id}. "
            "Ensure statsbombpy has access to this match's data."
        )

    return {"events": events}


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    comps = list_available_competitions()
    print(comps[["competition_id", "season_id", "competition_name"]].head(10))

    COMPETITION_ID = 11   # La Liga
    SEASON_ID      = 90
    matches = load_matches(COMPETITION_ID, SEASON_ID)
    MATCH_ID = int(matches["match_id"].iloc[0])
    print(f"\nLoading match {MATCH_ID} ...")

    bundle = load_match_bundle(MATCH_ID)
    ev = bundle["events"]
    print("\nEvents shape:", ev.shape)
    print("Columns:", ev.columns.tolist())
    print("\nPlayer sample (non-null):")
    print(ev.dropna(subset=["player_id"])[
        ["player_id", "player_name", "team_name", "event_type", "x", "y", "time_seconds"]
    ].head(8).to_string())