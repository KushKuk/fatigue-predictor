"""
pipeline.py
-----------
Phase 4 end-to-end pipeline:

    1. Load all trained models (Phase 2 + 3)
    2. Load Phase 1 feature dataset
    3. Build InferenceEngine + AlertManager
    4. Run MatchSimulator (fast replay)
    5. Save simulation results + alert log
    6. Print summary report

Run from fatigue_predictor/ root:
    python phase4/pipeline.py
    python phase4/pipeline.py --speed 120 --max_time 2700
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BASE_DIR)

from data.dataset               import load_dataset, FatigueDataset, NON_FEATURE_COLS
from inference.inference_engine import InferenceEngine, _LRWrapper, _LSTMWrapper, _TransformerWrapper
from inference.alert_manager    import AlertManager, file_handler
from inference.match_simulator  import MatchSimulator

DEFAULT_PARQUET   = os.path.join(ROOT_DIR, "phase1", "outputs", "phase1_features_match3773386.parquet")
DEFAULT_OUT       = os.path.join(BASE_DIR, "outputs")
DEFAULT_LR_PATH   = os.path.join(ROOT_DIR, "phase2", "outputs", "lr_baseline.pkl")
DEFAULT_LSTM_PATH = os.path.join(ROOT_DIR, "phase2", "outputs", "lstm_baseline.pt")
DEFAULT_TF_PATH   = os.path.join(ROOT_DIR, "phase3", "outputs", "transformer.pt")


def _load_engine(
    ds:           FatigueDataset,
    lr_path:      str,
    lstm_path:    str,
    tf_path:      str,
    lstm_seq_len: int,
    tf_seq_len:   int,
    verbose:      bool,
) -> InferenceEngine:
    """Load all three model components into the InferenceEngine."""
    import pickle
    import torch
    from models.lstm_baseline import FatigueLSTM
    from models.transformer   import FatigueTransformer

    if verbose:
        print("[Engine] Loading Logistic Regression …")
    with open(lr_path, "rb") as f:
        lr_model = pickle.load(f)
    lr_w = _LRWrapper(lr_model)

    if verbose:
        print("[Engine] Loading LSTM …")
    lstm_sd    = torch.load(lstm_path, map_location="cpu")
    lstm_model = FatigueLSTM(n_features=ds.n_features)
    lstm_model.load_state_dict(lstm_sd)
    lstm_w = _LSTMWrapper(lstm_model, seq_len=lstm_seq_len)

    if verbose:
        print("[Engine] Loading Transformer …")
    ckpt                 = torch.load(tf_path, map_location="cpu")
    tf_cfg               = ckpt.get("config", {})
    tf_cfg["n_features"] = ds.n_features
    tf_model             = FatigueTransformer(**tf_cfg)
    tf_model.load_state_dict(ckpt["state_dict"])
    tf_w = _TransformerWrapper(tf_model, seq_len=tf_seq_len)

    return InferenceEngine(
        scaler       = ds.scaler,
        lr_wrapper   = lr_w,
        lstm_wrapper = lstm_w,
        tf_wrapper   = tf_w,
        lstm_seq_len = lstm_seq_len,
        tf_seq_len   = tf_seq_len,
    )


def run_pipeline(
    parquet_path: str,
    out_dir:      str,
    lr_path:      str,
    lstm_path:    str,
    tf_path:      str,
    speed_factor: float        = 300.0,
    max_time_s:   float | None = None,
    lstm_seq_len: int          = 10,
    tf_seq_len:   int          = 16,
    cooldown_s:   float        = 60.0,
    verbose:      bool         = True,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.time()-t0:5.1f}s] {msg}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    log("Loading Phase 1 dataset …")
    ds: FatigueDataset = load_dataset(parquet_path, verbose=verbose)

    # ── 2. Check model files ──────────────────────────────────────────────────
    for label, path in [("LR", lr_path), ("LSTM", lstm_path), ("Transformer", tf_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} model not found at {path}.\n"
                f"Run the Phase 2/3 pipeline first."
            )

    # ── 3. Load inference engine ──────────────────────────────────────────────
    log("Loading models into InferenceEngine …")
    engine = _load_engine(
        ds, lr_path, lstm_path, tf_path, lstm_seq_len, tf_seq_len, verbose
    )

    # ── 4. Build alert manager ────────────────────────────────────────────────
    alert_log_path = os.path.join(out_dir, "alerts.jsonl")
    alert_mgr      = AlertManager(cooldown_s=cooldown_s, escalation_windows=2)
    alert_mgr.register_handler(file_handler(alert_log_path))
    log(f"Alert log → {alert_log_path}")

    # ── 5. Load raw features ──────────────────────────────────────────────────
    features_df = pd.read_parquet(parquet_path)
    feat_cols   = [c for c in features_df.columns if c not in NON_FEATURE_COLS]

    # Build player name map from events if available
    player_names: dict[float, str] = {}
    try:
        from data.data_loader import load_events
        match_id = (
            int(features_df["match_id"].iloc[0])
            if "match_id" in features_df.columns
            else 3773386
        )
        events = load_events(match_id)
        pmap   = (
            events.dropna(subset=["player_id", "player_name"])
                  .groupby("player_id")["player_name"]
                  .first()
        )
        # Cast key through str before float: pmap.items() yields (Hashable, str)
        # and float() does not accept Hashable directly per Pyright's type stubs.
        player_names = {float(str(k)): str(v) for k, v in pmap.items()}
    except Exception:
        pass

    if "player_name" not in features_df.columns:
        features_df["player_name"] = features_df["player_id"].map(
            # Same str() cast for the lambda argument — pid is typed Hashable
            # by pandas .map(), so float(str(pid)) is required for type safety.
            lambda pid: player_names.get(float(str(pid)), f"Player {pid:.0f}")
        )

    # ── 6. Run simulation ─────────────────────────────────────────────────────
    log(f"Running match simulation (speed={speed_factor}×) …")

    from inference.match_simulator import SimulationTick

    def _on_tick(tick: SimulationTick) -> None:
        if tick.result.alert_level == "RED" and verbose:
            ts_m = int(tick.match_time_s // 60)
            ts_s = int(tick.match_time_s % 60)
            print(
                f"  🔴 [{ts_m:02d}:{ts_s:02d}] {tick.player_name:<28s} "
                f"risk={tick.result.risk_score:.3f}"
            )

    sim = MatchSimulator(
        engine        = engine,
        alert_manager = alert_mgr,
        features_df   = features_df,
        feature_cols  = feat_cols,
        player_names  = player_names,
        on_tick       = _on_tick,
    )

    ticks = sim.run(speed_factor=speed_factor, max_time_s=max_time_s, verbose=verbose)

    # ── 7. Save results ───────────────────────────────────────────────────────
    results_df = sim.summary_dataframe()
    csv_path   = os.path.join(out_dir, "simulation_results.csv")
    results_df.to_csv(csv_path, index=False)
    log(f"Results → {csv_path}")

    # Save feature names for dashboard SHAP panel
    feat_names_path = os.path.join(ROOT_DIR, "phase3", "outputs", "feature_names.json")
    if not os.path.exists(feat_names_path):
        with open(feat_names_path, "w") as f:
            json.dump(ds.feature_names, f)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    n_alerts = results_df["alert_fired"].sum()
    n_red    = (results_df["alert_level"] == "RED").sum()
    avg_lat  = results_df["latency_ms"].mean()

    log("\n── Phase 4 Summary ──────────────────────────────────────────────────")
    log(f"  Players tracked  : {results_df['player_id'].nunique()}")
    log(f"  Ticks processed  : {len(results_df)}")
    log(f"  Total alerts     : {int(n_alerts)}")
    log(f"  RED alerts       : {int(n_red)}")
    log(f"  Avg latency      : {avg_lat:.2f} ms")
    log(f"  Output dir       : {out_dir}")
    log("\n  Launch dashboard:")
    log(f"  streamlit run phase4/dashboard/dashboard.py")
    log("─────────────────────────────────────────────────────────────────────")

    return results_df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 — Real-time Inference Pipeline")
    parser.add_argument("--parquet",      type=str,   default=DEFAULT_PARQUET)
    parser.add_argument("--out_dir",      type=str,   default=DEFAULT_OUT)
    parser.add_argument("--lr_path",      type=str,   default=DEFAULT_LR_PATH)
    parser.add_argument("--lstm_path",    type=str,   default=DEFAULT_LSTM_PATH)
    parser.add_argument("--tf_path",      type=str,   default=DEFAULT_TF_PATH)
    parser.add_argument("--speed",        type=float, default=300.0,
                        help="Simulation speed multiplier (300 = 5 min replay per sec)")
    parser.add_argument("--max_time",     type=float, default=None,
                        help="Stop simulation after N match seconds")
    parser.add_argument("--lstm_seq_len", type=int,   default=10)
    parser.add_argument("--tf_seq_len",   type=int,   default=16)
    parser.add_argument("--cooldown",     type=float, default=60.0)
    parser.add_argument("--quiet",        action="store_true")
    args = parser.parse_args()

    run_pipeline(
        parquet_path = args.parquet,
        out_dir      = args.out_dir,
        lr_path      = args.lr_path,
        lstm_path    = args.lstm_path,
        tf_path      = args.tf_path,
        speed_factor = args.speed,
        max_time_s   = args.max_time,
        lstm_seq_len = args.lstm_seq_len,
        tf_seq_len   = args.tf_seq_len,
        cooldown_s   = args.cooldown,
        verbose      = not args.quiet,
    )