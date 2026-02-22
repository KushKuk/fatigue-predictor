"""
pipeline.py
-----------
Phase 1 end-to-end pipeline.

Run from the project root (fatigue_predictor/):
    python phase1/pipeline.py
    python phase1/pipeline.py --match_id 3773386 --out_dir phase1/outputs
"""

import argparse
import os
import sys
import time

import pandas as pd

# ── Make root-level packages (data/, features/, labeling/) importable ─────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from data.data_loader import load_events
from features.physical_features import compute_dual_window_features
from features.cognitive_features import compute_dual_window_cognitive
from labeling.labeler import label_performance_drops, forward_label

DEFAULT_MATCH_ID  = 3773386
DEFAULT_OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DEFAULT_LOOKAHEAD = 3


def run_pipeline(
    match_id:  int,
    out_dir:   str,
    lookahead: int  = DEFAULT_LOOKAHEAD,
    verbose:   bool = True,
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.time()-t0:6.1f}s] {msg}")

    log(f"Loading events for match {match_id} ...")
    events = load_events(match_id)
    log(f"  Events: {len(events)} rows | "
        f"Players: {events['player_id'].nunique()} | "
        f"Types: {events['event_type'].nunique()} unique")

    log("Extracting physical features from events ...")
    phy = compute_dual_window_features(events)
    log(f"  Physical features: {phy.shape}")

    log("Extracting cognitive features from events ...")
    cog = compute_dual_window_cognitive(events)
    log(f"  Cognitive features: {cog.shape}")

    log("Labeling performance drops ...")
    labeled = label_performance_drops(cog, phy)
    labeled = forward_label(labeled, lookahead_steps=lookahead)
    log(f"  Labeled: {labeled.shape} | "
        f"Drop rate: {labeled['drop_label'].mean():.2%} | "
        f"Future drop rate: {labeled['future_drop'].mean():.2%}")

    parquet_path = os.path.join(out_dir, f"phase1_features_match{match_id}.parquet")
    csv_path     = os.path.join(out_dir, f"phase1_features_match{match_id}.csv")
    labeled.to_parquet(parquet_path, index=False)
    labeled.to_csv(csv_path, index=False)
    log(f"Saved -> {parquet_path}")
    log(f"Saved -> {csv_path}")

    log("\n── Phase 1 Summary ──────────────────────────────────────────────")
    log(f"  Match ID        : {match_id}")
    log(f"  Players         : {labeled['player_id'].nunique()}")
    log(f"  Feature windows : {len(labeled)}")
    log(f"  Feature columns : {labeled.shape[1]}")
    log(f"  Drop dist       : {labeled['drop_label'].value_counts().to_dict()}")
    log(f"  Future drop dist: {labeled['future_drop'].value_counts().to_dict()}")
    log("─────────────────────────────────────────────────────────────────")

    return labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Pipeline")
    parser.add_argument("--match_id",  type=int, default=DEFAULT_MATCH_ID)
    parser.add_argument("--out_dir",   type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD)
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    df = run_pipeline(
        match_id  = args.match_id,
        out_dir   = args.out_dir,
        lookahead = args.lookahead,
        verbose   = not args.quiet,
    )

    preview_cols = [
        "player_id", "window_end_ts",
        "event_rate_1m", "hi_intensity_rate_1m", "fatigue_index_1m",
        "decision_latency_mean_1m", "error_rate_1m", "pass_accuracy_1m",
        "composite_z", "drop_label", "future_drop", "risk_score",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print("\nSample output:")
    print(df[preview_cols].dropna().head(5).to_string())