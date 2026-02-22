"""
match_simulator.py
-------------------
Replays a StatsBomb match as a real-time event stream, extracting
feature windows on the fly and feeding them to the InferenceEngine.

Simulates:
    - Rolling 60-second feature windows advancing every 30s
    - Per-player feature extraction from event data
    - Real-time scoring and alert dispatch

Usage:
    sim = MatchSimulator(engine, alert_manager, events_df, features_df)
    sim.run(speed_factor=60.0)   # 60x faster than real time
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from inference.inference_engine import InferenceEngine, PredictionResult
from inference.alert_manager    import AlertManager


@dataclass
class SimulationTick:
    match_time_s:  float
    player_id:     float
    player_name:   str
    result:        PredictionResult
    alert_fired:   bool


class MatchSimulator:
    """
    Replays a pre-processed features DataFrame in chronological order,
    scoring each player window through the InferenceEngine.

    Parameters
    ----------
    engine        : InferenceEngine with all models loaded
    alert_manager : AlertManager for alert dispatch
    features_df   : output of Phase 1 pipeline (must have window_end_ts,
                    player_id, player_name columns + feature columns)
    feature_cols  : list of feature column names to use
    player_names  : dict mapping player_id → name
    on_tick       : optional callback called for each tick
    """

    def __init__(
        self,
        engine:        InferenceEngine,
        alert_manager: AlertManager,
        features_df:   pd.DataFrame,
        feature_cols:  list[str],
        player_names:  dict[float, str] | None = None,
        on_tick:       Callable[[SimulationTick], None] | None = None,
    ) -> None:
        self.engine        = engine
        self.alert_manager = alert_manager
        self.feature_cols  = feature_cols
        self.player_names  = player_names or {}
        self.on_tick       = on_tick

        # Sort by time then player for deterministic replay
        self.df = features_df.sort_values(
            ["window_end_ts", "player_id"]
        ).reset_index(drop=True)

        self._tick_log: list[SimulationTick] = []

    def run(
        self,
        speed_factor: float = 60.0,
        max_time_s:   float | None = None,
        verbose:      bool  = True,
    ) -> list[SimulationTick]:
        """
        Replay the match.

        Parameters
        ----------
        speed_factor : 1.0 = real time, 60.0 = 60× faster
        max_time_s   : stop after this match time (None = full match)
        """
        self.engine.reset_all()
        self.alert_manager.__init__(
            self.alert_manager.cooldown_s,
            self.alert_manager.escalation_windows,
        )
        self._tick_log.clear()

        times = sorted(self.df["window_end_ts"].unique())
        if max_time_s:
            times = [t for t in times if t <= max_time_s]

        prev_wall = time.time()
        prev_sim  = float(times[0]) if times else 0.0

        if verbose:
            total_min = int(times[-1] // 60) if times else 0
            print(f"\n[Simulator] Starting match replay — "
                  f"{len(times)} windows | ~{total_min} min match\n")

        for sim_time in times:
            # Wall-clock delay to simulate real time
            sim_delta  = sim_time - prev_sim
            wall_delay = sim_delta / speed_factor
            sleep_time = (prev_wall + wall_delay) - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            prev_wall = time.time()
            prev_sim  = sim_time

            # Get all player windows at this time
            window_rows = self.df[self.df["window_end_ts"] == sim_time]

            for _, row in window_rows.iterrows():
                pid  = float(row["player_id"])
                name = self.player_names.get(pid, f"Player {pid:.0f}")

                # Extract feature vector
                fvec = row[self.feature_cols].values.astype(np.float32)
                if np.isnan(fvec).any():
                    fvec = np.nan_to_num(fvec, nan=0.0)

                # Score
                result = self.engine.score(
                    player_id=pid,
                    feature_vector=fvec,
                    timestamp_s=sim_time,
                    smooth=True,
                )

                # Alert
                alert = self.alert_manager.process(result, player_name=name)

                tick = SimulationTick(
                    match_time_s=sim_time,
                    player_id=pid,
                    player_name=name,
                    result=result,
                    alert_fired=alert is not None,
                )
                self._tick_log.append(tick)

                if self.on_tick:
                    self.on_tick(tick)

        if verbose:
            n_alerts = sum(1 for t in self._tick_log if t.alert_fired)
            n_red    = sum(
                1 for t in self._tick_log
                if t.alert_fired and t.result.alert_level == "RED"
            )
            print(f"\n[Simulator] Done. "
                  f"Ticks: {len(self._tick_log)} | "
                  f"Alerts: {n_alerts} | RED: {n_red}")

        return self._tick_log

    def get_player_timeline(self, player_id: float) -> list[SimulationTick]:
        """Return all ticks for a specific player."""
        return [t for t in self._tick_log if t.player_id == player_id]

    def summary_dataframe(self) -> pd.DataFrame:
        """Return tick log as a DataFrame for analysis."""
        rows = []
        for t in self._tick_log:
            rows.append({
                "match_time_s":     t.match_time_s,
                "match_time_min":   t.match_time_s / 60.0,
                "player_id":        t.player_id,
                "player_name":      t.player_name,
                "risk_score":       t.result.risk_score,
                "lr_score":         t.result.lr_score,
                "lstm_score":       t.result.lstm_score,
                "transformer_score":t.result.transformer_score,
                "alert_level":      t.result.alert_level,
                "confidence_low":   t.result.confidence_low,
                "confidence_high":  t.result.confidence_high,
                "alert_fired":      t.alert_fired,
                "latency_ms":       t.result.latency_ms,
            })
        return pd.DataFrame(rows)