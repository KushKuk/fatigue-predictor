"""
inference_engine.py
--------------------
Real-time inference engine for fatigue drop prediction.

Loads all Phase 2/3 trained models and scores incoming feature
windows as they arrive from the event stream.

Supports:
    - Ensemble scoring (weighted average of LR + LSTM + Transformer)
    - Per-player rolling risk buffer
    - Confidence intervals via MC-Dropout (Transformer only)
    - Sub-second latency on CPU

Usage:
    engine = InferenceEngine.from_outputs(root_dir=".")
    result = engine.score(player_id=123, feature_vector=np.array([...]))
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    player_id:       float
    timestamp_s:     float
    risk_score:      float          # ensemble 0–1
    lr_score:        float
    lstm_score:      float
    transformer_score: float
    alert_level:     str            # "GREEN" | "AMBER" | "RED"
    confidence_low:  float          # MC-Dropout lower bound
    confidence_high: float          # MC-Dropout upper bound
    latency_ms:      float


# ── Risk thresholds ────────────────────────────────────────────────────────────

ALERT_THRESHOLDS = {
    "GREEN": (0.00, 0.40),
    "AMBER": (0.40, 0.65),
    "RED":   (0.65, 1.00),
}

ENSEMBLE_WEIGHTS = {
    "lr":          0.20,
    "lstm":        0.35,
    "transformer": 0.45,
}

MC_DROPOUT_SAMPLES = 20
RISK_BUFFER_LEN    = 10   # smooth over last N windows


def _alert_level(score: float) -> str:
    for level, (lo, hi) in ALERT_THRESHOLDS.items():
        if lo <= score <= hi:
            return level
    return "RED"


# ── Thin model wrappers ────────────────────────────────────────────────────────

class _LRWrapper:
    """Wraps sklearn CalibratedClassifierCV for single-vector scoring."""
    def __init__(self, model: CalibratedClassifierCV) -> None:
        self.model: CalibratedClassifierCV = model

    def predict_proba(self, x: np.ndarray) -> float:
        return float(self.model.predict_proba(x.reshape(1, -1))[0, 1])


class _LSTMWrapper:
    """Wraps FatigueLSTM for single-sequence scoring."""
    def __init__(self, model: torch.nn.Module, seq_len: int = 10) -> None:
        self.model   = model
        self.seq_len = seq_len
        self.model.eval()

    def predict_proba(self, seq: np.ndarray) -> float:
        """seq: (seq_len, F)"""
        x = torch.tensor(seq[None], dtype=torch.float32)
        with torch.no_grad():
            logit = self.model(x)
            logit = logit[0] if isinstance(logit, tuple) else logit
        return float(torch.sigmoid(logit).item())


class _TransformerWrapper:
    """Wraps FatigueTransformer; supports MC-Dropout for uncertainty."""
    def __init__(self, model: torch.nn.Module, seq_len: int = 16) -> None:
        self.model   = model
        self.seq_len = seq_len

    def predict_proba(self, seq: np.ndarray) -> float:
        self.model.eval()
        x = torch.tensor(seq[None], dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)
            logit = out[0] if isinstance(out, tuple) else out
        return float(torch.sigmoid(logit).item())

    def predict_with_uncertainty(
        self, seq: np.ndarray, n_samples: int = MC_DROPOUT_SAMPLES
    ) -> tuple[float, float, float]:
        """Enable dropout at inference time for MC-Dropout uncertainty."""
        def _enable_dropout(m: torch.nn.Module) -> None:
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.model.eval()
        self.model.apply(_enable_dropout)

        x       = torch.tensor(seq[None], dtype=torch.float32)
        probas: list[float] = []
        with torch.no_grad():
            for _ in range(n_samples):
                out   = self.model(x)
                logit = out[0] if isinstance(out, tuple) else out
                probas.append(float(torch.sigmoid(logit).item()))

        self.model.eval()   # restore eval mode
        arr = np.array(probas)
        return float(arr.mean()), float(np.percentile(arr, 5)), float(np.percentile(arr, 95))


# ── Inference Engine ───────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Loads all trained models and scores feature windows in real time.

    Parameters
    ----------
    scaler      : fitted StandardScaler from Phase 2 dataset
    lr_wrapper  : _LRWrapper
    lstm_wrapper: _LSTMWrapper
    tf_wrapper  : _TransformerWrapper
    """

    def __init__(
        self,
        scaler:       StandardScaler,
        lr_wrapper:   _LRWrapper,
        lstm_wrapper: _LSTMWrapper,
        tf_wrapper:   _TransformerWrapper,
        lstm_seq_len: int = 10,
        tf_seq_len:   int = 16,
        weights:      dict[str, float] = ENSEMBLE_WEIGHTS,
    ) -> None:
        self.scaler       = scaler
        self.lr           = lr_wrapper
        self.lstm         = lstm_wrapper
        self.tf           = tf_wrapper
        self.lstm_seq_len = lstm_seq_len
        self.tf_seq_len   = tf_seq_len
        self.weights      = weights

        # Per-player feature history buffers (for sequence models)
        max_buf = max(lstm_seq_len, tf_seq_len) + 5
        self._buffers: dict[float, deque] = {}
        self._max_buf = max_buf

        # Per-player smoothed risk buffer
        self._risk_buffers: dict[float, deque] = {}

    @classmethod
    def from_outputs(
        cls,
        root_dir:     str = ".",
        lstm_seq_len: int = 10,
        tf_seq_len:   int = 16,
    ) -> "InferenceEngine":
        """
        Load all models from standard output paths.

        Expects:
            {root}/phase2/outputs/lr_baseline.pkl
            {root}/phase2/outputs/lstm_baseline.pt
            {root}/phase3/outputs/transformer.pt
            (dataset scaler embedded in lr_baseline.pkl via pipeline)
        """
        # Scaler
        from data.dataset import load_dataset
        import glob
        parquets = sorted(glob.glob(
            os.path.join(root_dir, "phase1", "outputs", "*.parquet")
        ))
        if not parquets:
            raise FileNotFoundError("No phase1 parquet found. Run phase1 pipeline first.")
        ds = load_dataset(parquets[0], verbose=False)
        scaler = ds.scaler

        # LR
        lr_path = os.path.join(root_dir, "phase2", "outputs", "lr_baseline.pkl")
        with open(lr_path, "rb") as f:
            lr_model = pickle.load(f)
        lr_w = _LRWrapper(lr_model)

        # LSTM
        from models.lstm_baseline import FatigueLSTM, load_lstm
        lstm_path = os.path.join(root_dir, "phase2", "outputs", "lstm_baseline.pt")
        lstm_model = load_lstm(lstm_path, n_features=ds.n_features)
        lstm_w = _LSTMWrapper(lstm_model, seq_len=lstm_seq_len)

        # Transformer
        from models.transformer import load_transformer
        tf_path = os.path.join(root_dir, "phase3", "outputs", "transformer.pt")
        tf_model = load_transformer(tf_path, n_features=ds.n_features)
        tf_w = _TransformerWrapper(tf_model, seq_len=tf_seq_len)

        return cls(scaler, lr_w, lstm_w, tf_w, lstm_seq_len, tf_seq_len)

    def _get_buffer(self, player_id: float) -> deque:
        if player_id not in self._buffers:
            self._buffers[player_id] = deque(maxlen=self._max_buf)
        return self._buffers[player_id]

    def _smooth_risk(self, player_id: float, score: float) -> float:
        if player_id not in self._risk_buffers:
            self._risk_buffers[player_id] = deque(maxlen=RISK_BUFFER_LEN)
        self._risk_buffers[player_id].append(score)
        return float(np.mean(self._risk_buffers[player_id]))

    def score(
        self,
        player_id:      float,
        feature_vector: np.ndarray,
        timestamp_s:    float | None = None,
        smooth:         bool  = True,
    ) -> PredictionResult:
        """
        Score a single feature window for one player.

        Parameters
        ----------
        player_id      : player identifier
        feature_vector : (F,) scaled feature vector
        timestamp_s    : match time in seconds (defaults to wall clock)
        smooth         : apply rolling average over risk scores

        Returns
        -------
        PredictionResult
        """
        t_start = time.perf_counter()
        ts      = timestamp_s or time.time()

        # Scale
        x_scaled = self.scaler.transform(feature_vector.reshape(1, -1))[0]

        # Update buffer
        buf = self._get_buffer(player_id)
        buf.append(x_scaled)

        # ── LR score (flat features) ──────────────────────────────────────────
        lr_score = self.lr.predict_proba(x_scaled)

        # ── LSTM score (sequence) ─────────────────────────────────────────────
        if len(buf) >= self.lstm_seq_len:
            lstm_seq  = np.array(list(buf))[-self.lstm_seq_len:]
            lstm_score = self.lstm.predict_proba(lstm_seq)
        else:
            lstm_score = lr_score   # fallback until buffer fills

        # ── Transformer score + uncertainty ───────────────────────────────────
        if len(buf) >= self.tf_seq_len:
            tf_seq = np.array(list(buf))[-self.tf_seq_len:]
            tf_score, ci_lo, ci_hi = self.tf.predict_with_uncertainty(tf_seq)
        else:
            tf_score, ci_lo, ci_hi = lr_score, lr_score * 0.8, min(lr_score * 1.2, 1.0)

        # ── Ensemble ──────────────────────────────────────────────────────────
        w  = self.weights
        ensemble = (
            w["lr"]          * lr_score +
            w["lstm"]        * lstm_score +
            w["transformer"] * tf_score
        )

        if smooth:
            ensemble = self._smooth_risk(player_id, ensemble)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return PredictionResult(
            player_id         = player_id,
            timestamp_s       = ts,
            risk_score        = round(ensemble, 4),
            lr_score          = round(lr_score, 4),
            lstm_score        = round(lstm_score, 4),
            transformer_score = round(tf_score, 4),
            alert_level       = _alert_level(ensemble),
            confidence_low    = round(ci_lo, 4),
            confidence_high   = round(ci_hi, 4),
            latency_ms        = round(latency_ms, 2),
        )

    def reset_player(self, player_id: float) -> None:
        """Clear history buffers for a player (e.g. after substitution)."""
        self._buffers.pop(player_id, None)
        self._risk_buffers.pop(player_id, None)

    def reset_all(self) -> None:
        self._buffers.clear()
        self._risk_buffers.clear()