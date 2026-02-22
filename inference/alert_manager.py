"""
alert_manager.py
-----------------
Alert management system for fatigue drop predictions.

Responsibilities:
    - Debounce repeated alerts (cooldown per player)
    - Escalate: GREEN → AMBER → RED based on consecutive windows
    - Route alerts to registered handlers (console, file, webhook)
    - Maintain alert history per player
    - Produce structured alert payloads for the dashboard
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Callable

from inference.inference_engine import PredictionResult


# ── Alert payload ──────────────────────────────────────────────────────────────

@dataclass
class Alert:
    alert_id:      str
    player_id:     float
    player_name:   str
    timestamp_s:   float
    alert_level:   str            # GREEN | AMBER | RED
    risk_score:    float
    confidence_lo: float
    confidence_hi: float
    message:       str
    consecutive:   int            # consecutive windows at this level
    action:        str            # recommended coaching action


LEVEL_PRIORITY = {"GREEN": 0, "AMBER": 1, "RED": 2}

ACTION_MAP = {
    "GREEN": "Monitor — player within normal parameters.",
    "AMBER": "Consider rotation in next natural stoppage.",
    "RED":   "Immediate substitution recommended.",
}

MESSAGE_MAP = {
    "GREEN": "Performance nominal.",
    "AMBER": "Early fatigue indicators detected — elevated risk.",
    "RED":   "CRITICAL: High probability of imminent performance drop.",
}


# ── Alert Manager ──────────────────────────────────────────────────────────────

class AlertManager:
    """
    Stateful alert manager that tracks per-player alert history
    and routes escalation events to registered handlers.

    Parameters
    ----------
    cooldown_s         : minimum seconds between alerts for same player+level
    escalation_windows : consecutive windows at level before escalating
    max_history        : max alerts kept per player
    """

    def __init__(
        self,
        cooldown_s:          float = 60.0,
        escalation_windows:  int   = 2,
        max_history:         int   = 100,
    ) -> None:
        self.cooldown_s         = cooldown_s
        self.escalation_windows = escalation_windows

        # Per-player tracking
        self._last_alert_ts:   dict[float, float]        = defaultdict(float)
        self._consecutive:     dict[float, dict[str, int]]= defaultdict(lambda: defaultdict(int))
        self._history:         dict[float, deque]         = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._handlers:        list[Callable[[Alert], None]] = []

        # Register default console handler
        self.register_handler(_console_handler)

    def register_handler(self, fn: Callable[[Alert], None]) -> None:
        """Register a callback that receives Alert objects."""
        self._handlers.append(fn)

    def process(
        self,
        result:      PredictionResult,
        player_name: str = "",
    ) -> Alert | None:
        """
        Process a PredictionResult and fire an alert if warranted.

        Returns the Alert if one was fired, else None.
        """
        pid   = result.player_id
        level = result.alert_level
        now   = result.timestamp_s or time.time()

        # Update consecutive counter
        for other_level in LEVEL_PRIORITY:
            if other_level != level:
                self._consecutive[pid][other_level] = 0
        self._consecutive[pid][level] += 1
        consecutive = self._consecutive[pid][level]

        # Only alert if:
        # 1. Level is AMBER or RED
        # 2. Consecutive windows at this level >= threshold
        # 3. Outside cooldown window
        if LEVEL_PRIORITY[level] == 0:
            return None

        if consecutive < self.escalation_windows:
            return None

        if now - self._last_alert_ts[pid] < self.cooldown_s:
            return None

        # Build alert
        alert = Alert(
            alert_id     = f"{pid:.0f}_{level}_{now:.0f}",
            player_id    = pid,
            player_name  = player_name or f"Player {pid:.0f}",
            timestamp_s  = now,
            alert_level  = level,
            risk_score   = result.risk_score,
            confidence_lo= result.confidence_low,
            confidence_hi= result.confidence_high,
            message      = MESSAGE_MAP[level],
            consecutive  = consecutive,
            action       = ACTION_MAP[level],
        )

        # Record and dispatch
        self._last_alert_ts[pid] = now
        self._history[pid].append(alert)
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"[AlertManager] Handler error: {e}")

        return alert

    def get_history(self, player_id: float) -> list[Alert]:
        return list(self._history[player_id])

    def get_all_active(self) -> dict[float, Alert | None]:
        """Return most recent alert per player."""
        return {
            pid: hist[-1] if hist else None
            for pid, hist in self._history.items()
        }

    def reset_player(self, player_id: float) -> None:
        """Reset state for a player (e.g. after substitution)."""
        self._consecutive.pop(player_id, None)
        self._last_alert_ts.pop(player_id, None)
        self._history.pop(player_id, None)


# ── Default handlers ───────────────────────────────────────────────────────────

_LEVEL_COLOURS = {"GREEN": "\033[92m", "AMBER": "\033[93m", "RED": "\033[91m"}
_RESET = "\033[0m"


def _console_handler(alert: Alert) -> None:
    colour = _LEVEL_COLOURS.get(alert.alert_level, "")
    ts_min = int(alert.timestamp_s // 60)
    ts_sec = int(alert.timestamp_s % 60)
    print(
        f"{colour}[{alert.alert_level}] {alert.player_name:30s} "
        f"risk={alert.risk_score:.2f} "
        f"({alert.confidence_lo:.2f}–{alert.confidence_hi:.2f}) "
        f"@ {ts_min:02d}:{ts_sec:02d}  →  {alert.action}{_RESET}"
    )


def file_handler(log_path: str) -> Callable[[Alert], None]:
    """Returns a handler that appends alerts as JSON lines to log_path."""
    def _handler(alert: Alert) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(asdict(alert)) + "\n")
    return _handler


def webhook_handler(url: str, timeout: float = 2.0) -> Callable[[Alert], None]:
    """Returns a handler that POSTs alerts as JSON to a webhook URL."""
    def _handler(alert: Alert) -> None:
        try:
            import urllib.request
            payload = json.dumps(asdict(alert)).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:
            print(f"[Webhook] Failed to POST alert: {e}")
    return _handler