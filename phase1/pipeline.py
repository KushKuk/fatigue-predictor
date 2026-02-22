"""
pipeline.py
-----------
Phase 1 end-to-end pipeline:

    1. Load StatsBomb events + synthetic tracking
    2. Extract physical fatigue features
    3. Extract cognitive fatigue features
    4. Label performance drops
    5. Save clean feature dataset to disk

Usage:
    python pipeline.py --match_id 3788741 --out_dir ../outputs
"""

import argparse
import os
import sys
import time

import pandas as pd

# ─── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.data_loader              import load_events, generate_synthetic_tracking, load_matches
from features.physical_features    import compute_dual_window_features
from features.cognitive_features   import compute_dual_window_cognitive
from labeling.labeler              import label_performance_drops, forward_label


# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MATCH_ID       = 3788741   # StatsBomb open data
DEFAULT_OUT_DIR        = os.path.join(BASE_DIR, "outputs")
DEFAULT_TRACKING_RATE  = 10        # Hz
DEFAULT_LOOKAHEAD      = 3         # future windows for forward label


def run_pipeline(
    match_id:      int,
    out_dir:       str,
    tracking_rate: int  = DEFAULT_TRACKING_RATE,
    lookahead:     int  = DEFAULT_LOOKAHEAD,
    verbose:       bool = True,
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    def log(msg):
        if verbose:
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s] {msg}")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    log(f"Loading events for match {match_id} …")
    events = load_events(match_id)
    log(f"  Events loaded: {len(events)} rows, "
        f"{events['player_id'].nunique()} players")

    player_ids = events["player_id"].dropna().unique().tolist()
    log(f"Generating synthetic tracking for {len(player_ids)} players …")
    tracking = generate_synthetic_tracking(
        match_id=match_id,
        player_ids=player_ids,
        sample_rate_hz=tracking_rate,
    )
    log(f"  Tracking generated: {len(tracking)} frames")

    # ── Step 2: Physical features ────────────────────────────────────────────
    log("Extracting physical fatigue features …")
    phy = compute_dual_window_features(tracking, sample_rate_hz=tracking_rate)
    log(f"  Physical features: {phy.shape}")

    # ── Step 3: Cognitive features ────────────────────────────────────────────
    log("Extracting cognitive fatigue features …")
    cog = compute_dual_window_cognitive(events)
    log(f"  Cognitive features: {cog.shape}")

    # ── Step 4: Labeling ──────────────────────────────────────────────────────
    log("Labeling performance drops …")
    labeled = label_performance_drops(cog, phy)
    labeled = forward_label(labeled, lookahead_steps=lookahead)
    log(f"  Labeled dataset: {labeled.shape} | "
        f"Drop rate: {labeled['drop_label'].mean():.2%} | "
        f"Future drop rate: {labeled['future_drop'].mean():.2%}")

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"phase1_features_match{match_id}.parquet")
    labeled.to_parquet(out_path, index=False)
    log(f"Saved → {out_path}")

    csv_path = os.path.join(out_dir, f"phase1_features_match{match_id}.csv")
    labeled.to_csv(csv_path, index=False)
    log(f"Saved → {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n── Phase 1 Summary ─────────────────────────────────────────────────")
    log(f"  Match ID         : {match_id}")
    log(f"  Players          : {labeled['player_id'].nunique()}")
    log(f"  Feature windows  : {len(labeled)}")
    log(f"  Feature columns  : {labeled.shape[1]}")
    log(f"  Drop label dist  : {labeled['drop_label'].value_counts().to_dict()}")
    log(f"  Future drop dist : {labeled['future_drop'].value_counts().to_dict()}")
    log(f"  Output           : {out_dir}")
    log("─────────────────────────────────────────────────────────────────────")

    return labeled


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fatigue Predictor — Phase 1 Pipeline")
    parser.add_argument("--match_id",  type=int, default=DEFAULT_MATCH_ID)
    parser.add_argument("--out_dir",   type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rate",      type=int, default=DEFAULT_TRACKING_RATE,
                        help="Tracking sample rate in Hz")
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD,
                        help="Forward-label lookahead in windows")
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    df = run_pipeline(
        match_id      = args.match_id,
        out_dir       = args.out_dir,
        tracking_rate = args.rate,
        lookahead     = args.lookahead,
        verbose       = not args.quiet,
    )

    print("\nSample output (first 5 rows):")
    preview_cols = [
        "player_id", "window_end_ts",
        "sprint_rate_1m", "hi_effort_ratio_1m", "fatigue_index_1m",
        "decision_latency_mean_1m", "error_rate_1m", "pass_accuracy_1m",
        "composite_z", "drop_label", "future_drop", "risk_score",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(5).to_string())