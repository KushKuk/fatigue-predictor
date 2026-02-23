# AI Fatigue Predictor

A real-time athlete performance drop prediction system built on StatsBomb open event data. Uses an ensemble of Logistic Regression, BiLSTM, and a Spatio-Temporal Transformer to detect early signs of physical and cognitive fatigue in football players — before the drop becomes visible on the pitch.

**Transformer AUC: 0.802** on multi-match held-out test set.

---

## The Problem

Coaching staff make substitution decisions based on what they can see — visible fatigue, errors, reduced sprint intensity. By the time these signals are observable, performance has already degraded. There is no systematic, data-driven way to predict *who is about to drop* rather than *who has already dropped*.

This system solves that by:

- Extracting physical and cognitive performance features from raw event data in rolling time windows
- Learning the temporal signature of pre-drop behaviour across hundreds of players and matches
- Scoring every player every 30 seconds during a live match and issuing tiered alerts to coaching staff
- Providing a real-time dashboard showing risk scores, alert history, and model confidence intervals

---

## Architecture Overview

```
StatsBomb Events
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase 1 — Feature Engineering              │
│  Physical features (event rate, carry rate, │
│  duel rate, hi-intensity, fatigue index)     │
│  Cognitive features (decision latency,       │
│  pass accuracy, action entropy, error rate)  │
│  → 39 features × 1-min + 5-min windows      │
│  → Z-score drop labelling (5 signals)        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Phase 2 — Baseline Models                  │
│  Logistic Regression  (AUC 0.63)            │
│  BiLSTM + Attention   (AUC 0.53)            │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Phase 3 — Spatio-Temporal Transformer      │
│  CLS token + PreNorm blocks                 │
│  Learnable positional encoding              │
│  Optional survival head                     │
│  Optuna hyperparameter search               │
│  SHAP feature importance                    │
│  AUC 0.802 on multi-match data              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Phase 4 — Real-time Inference              │
│  Ensemble: LR 20% + LSTM 35% + TF 45%      │
│  MC-Dropout confidence intervals            │
│  Per-player rolling risk buffer             │
│  Alert manager (GREEN / AMBER / RED)        │
│  Match simulator (live replay)              │
│  Streamlit dashboard                        │
└─────────────────────────────────────────────┘
```

---

## Project Structure

```
fatigue_predictor/
│
├── data/
│   ├── data_loader.py          # StatsBomb 1.16 ingestion
│   └── dataset.py              # Train/val/test splits, scaler, FatigueDataset
│
├── features/
│   ├── physical_features.py    # Event-based physical proxies
│   └── cognitive_features.py  # Decision latency, entropy, pass accuracy
│
├── labeling/
│   └── labeler.py              # 5-signal z-score drop labelling
│
├── models/
│   ├── logistic_baseline.py    # GridSearchCV + Platt calibration
│   ├── lstm_baseline.py        # BiLSTM + attention pooling
│   ├── transformer.py          # Spatio-Temporal Transformer
│   ├── trainer.py              # Training loop, warmup cosine LR, AMP
│   ├── tuner.py                # Optuna TPE hyperparameter search
│   └── explainability.py       # SHAP DeepExplainer
│
├── evaluation/
│   └── evaluator.py            # Metrics, ROC, calibration, EDR
│
├── inference/
│   ├── inference_engine.py     # Ensemble scorer, MC-Dropout CI
│   ├── alert_manager.py        # Cooldown, escalation, handlers
│   └── match_simulator.py      # Chronological match replay
│
├── dashboard/
│   └── dashboard.py            # Streamlit live monitoring dashboard
│
├── phase1/
│   ├── pipeline.py             # Phase 1 orchestrator
│   └── outputs/                # Parquet feature files
│
├── phase2/
│   ├── pipeline.py             # Phase 2 orchestrator
│   └── outputs/                # lr_baseline.pkl, lstm_baseline.pt
│
├── phase3/
│   ├── pipeline.py             # Phase 3 orchestrator
│   └── outputs/                # transformer.pt, shap_importance.png
│
└── phase4/
    ├── pipeline.py             # Phase 4 orchestrator
    └── outputs/                # simulation_results.csv, alerts.jsonl
```

---

## How It Works

### Feature Engineering (Phase 1)

Raw StatsBomb events (passes, carries, duels, pressures, etc.) are processed into rolling time windows per player. Two window sizes are used — 1-minute and 5-minute — to capture both acute spikes and sustained trends.

**Physical features** proxy for physical load without GPS or tracking data:
- `event_rate` — actions per minute (drops when a player fatigues)
- `carry_rate`, `duel_rate`, `press_rate` — high-intensity action fractions
- `hi_intensity_rate` — proportion of events classified as high-effort
- `fatigue_index` — decline in event rate relative to the player's first-half baseline
- `position_spread_x/y` — spatial coverage (fatigued players cover less ground)

**Cognitive features** proxy for decision quality:
- `decision_latency` — time between receiving the ball and acting
- `pass_accuracy` — rolling pass completion rate
- `action_entropy` — variety of action types (drops when players simplify under fatigue)
- `error_rate` — turnovers and miscontrol frequency

**Labelling** uses a composite z-score across 5 signals (`z_pass_acc`, `z_error_rate`, `z_latency`, `z_hi_intensity`, `z_fatigue_idx`). A player-window is labelled a drop if any signal crosses ±1.5σ or the composite exceeds 1.0. The target is `future_drop` — whether a drop occurs in the next N windows — making it a genuinely predictive rather than retrospective task.

---

### Baseline Models (Phase 2)

Two baselines establish the performance floor:

**Logistic Regression** — trained with GridSearchCV and Platt-calibrated. Conservative predictor: high precision, low recall. AUC 0.63 on single-match data. Provides interpretable coefficients and fast inference.

**BiLSTM with Attention** — bidirectional LSTM over 10-step sequences with a soft attention pooling layer. AUC 0.53 on single-match data — below LR because LSTMs need substantial sequential data to learn temporal patterns, and a single match provides too few examples.

---

### Spatio-Temporal Transformer (Phase 3)

The Transformer architecture:

- **Input projection** — linear layer mapping 39 features to `d_model` dimensions
- **CLS token** — a learnable classification token prepended to each sequence (BERT-style), pooled at the output for classification
- **Learnable positional encoding** — better than sinusoidal for the short sequences (8–32 steps) used here
- **PreNorm Transformer blocks** — LayerNorm applied before attention (more stable training than post-norm), multi-head self-attention, GELU feedforward network
- **Classification head** — two-layer MLP on the CLS token representation → sigmoid probability
- **Optional survival head** — Softplus output estimating time-to-drop in window units alongside the binary label

Training uses a linear warmup into cosine annealing LR schedule, weighted BCE loss for class imbalance, gradient clipping, and early stopping with best-weight restoration. Mixed precision (AMP) is enabled when CUDA is available.

Hyperparameters are tuned with Optuna TPE search over `d_model`, `n_heads`, `n_layers`, `ffn_dim`, `dropout`, `lr`, `seq_len`, and `batch_size`. The objective is validation AUC.

**SHAP explainability** uses `DeepExplainer` to compute per-feature mean |SHAP| values, collapsed from the `(M, seq_len, F)` attribution tensor.

**On multi-match data: AUC 0.802.**

---

### Real-time Inference (Phase 4)

The inference layer wraps all three trained models into a unified scoring engine:

**Ensemble scoring** — weighted average of all three models (LR 20%, LSTM 35%, Transformer 45%), with a 10-window rolling average to smooth transient noise.

**MC-Dropout confidence intervals** — at inference time, the Transformer's dropout layers are re-enabled and 20 stochastic forward passes are run. The 5th and 95th percentiles form a confidence interval displayed on every alert.

**Alert tiers:**
| Level | Risk Score | Action |
|-------|-----------|--------|
| GREEN | < 0.40 | Monitor normally |
| AMBER | 0.40 – 0.65 | Consider rotation at next stoppage |
| RED | > 0.65 | Immediate substitution recommended |

Alerts require 2 consecutive windows at the same level before firing, and a 60-second cooldown per player to prevent spam. Handlers route alerts to the console, a JSONL log file, or any webhook URL.

The **match simulator** replays a processed features DataFrame chronologically at configurable speed (default 300× real time). Results are written to CSV on every tick so the dashboard updates live.

---

### Dashboard (Phase 4)

A Streamlit application providing:

- **KPI row** — live RED/AMBER counts, total alerts, average inference latency
- **Player risk leaderboard** — colour-coded risk bars for all tracked players, sorted by current risk score
- **Alert feed** — scrollable history of fired alerts with timestamps and recommended actions
- **Risk timeline** — per-player rolling risk score chart over the match
- **Model agreement panel** — separate time series for each model's output to diagnose disagreement
- **SHAP importance** — top-15 feature importance chart from the trained Transformer
- **Model comparison table** — AUC, F1, Brier score across all three architectures

---

## Quickstart

```bash
# Install dependencies
pip install -r phase4/requirements.txt

# Phase 1 — extract features from a StatsBomb match
python phase1/pipeline.py

# Phase 2 — train baseline models
python phase2/pipeline.py

# Phase 3 — train Transformer (with optional tuning)
python phase3/pipeline.py --tune --n_trials 20

# Phase 4 — run live simulation
python phase4/pipeline.py

# Launch dashboard (in a separate terminal)
python -m streamlit run dashboard/dashboard.py
```

---

## Results

| Model | AUC | F1 | Precision | Recall | Brier |
|-------|-----|----|-----------|--------|-------|
| Logistic Regression | 0.63 | 0.07 | 0.71 | 0.04 | 0.22 |
| BiLSTM | 0.53 | 0.48 | 0.35 | 0.77 | 0.26 |
| **Transformer (multi-match)** | **0.802** | — | — | — | — |

The Transformer's improvement from 0.53 to 0.802 is driven almost entirely by data volume. Attention-based models require sufficient examples of varied fatigue trajectories to learn temporal patterns — a single match with 32 players is insufficient. The multi-match merged dataset provides that volume.

---

## Data

Uses [StatsBomb open data](https://github.com/statsbomb/open-data) via the `statsbombpy` library. No credentials required for open data access. Events data includes passes, carries, duels, pressures, ball receipts, and all other on-ball actions with timestamps and player identifiers.

---

## Key Design Decisions

**Event data only, no tracking data** — GPS and physical tracking data is proprietary and unavailable for most teams. This system deliberately works with event data alone, making it deployable with any StatsBomb feed.

**Player-boundary-aware windowing** — sequences never cross player boundaries. Each player's timeline is windowed independently to prevent the model from learning spurious cross-player patterns.

**Predictive not retrospective labelling** — the target is `future_drop` (will this player drop in the next N windows?) rather than `drop_label` (has this player already dropped?). This is what makes the system actionable.

**Ensemble over single model** — the three models have different failure modes. LR is conservative and rarely fires; the LSTM has high recall but noisy precision; the Transformer is the strongest overall. The weighted ensemble balances these properties at inference time.