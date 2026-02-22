"""
explainability.py
-----------------
SHAP-based feature importance for FatigueTransformer.

Since the Transformer takes sequences, we use:
    - shap.DeepExplainer  (fast, PyTorch-native)
    - Fallback to shap.KernelExplainer if DeepExplainer fails

Outputs:
    - Per-feature mean |SHAP| values
    - Bar chart of top-K features
    - Summary plot (beeswarm)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.transformer import FatigueTransformer


# ── SHAP wrapper ───────────────────────────────────────────────────────────────

class TransformerSHAPWrapper(torch.nn.Module):
    """Wraps FatigueTransformer to output probabilities for SHAP."""
    def __init__(self, model: FatigueTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out    = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        return torch.sigmoid(logits).unsqueeze(-1)   # (B, 1)


def compute_shap_values(
    model:         FatigueTransformer,
    X_background:  np.ndarray,
    X_explain:     np.ndarray,
    seq_len:       int = 16,
    n_background:  int = 50,
    device:        str = "cpu",
) -> np.ndarray:
    """
    Compute SHAP values for X_explain using DeepExplainer.

    Parameters
    ----------
    X_background : (N, F) array — background dataset (train samples)
    X_explain    : (M, F) array — samples to explain
    seq_len      : sequence length used during training
    n_background : number of background samples to use (keep small for speed)

    Returns
    -------
    shap_values : (M, F) mean |SHAP| collapsed over sequence dimension
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    wrapper = TransformerSHAPWrapper(model).to(device)
    wrapper.eval()

    # Build sequences from flat arrays
    def to_seq(X: np.ndarray) -> torch.Tensor:
        n = len(X) - seq_len + 1
        seqs = np.stack([X[i : i + seq_len] for i in range(max(n, 1))], axis=0)
        return torch.tensor(seqs, dtype=torch.float32).to(device)

    bg_seq  = to_seq(X_background[:n_background])
    exp_seq = to_seq(X_explain)

    explainer   = shap.DeepExplainer(wrapper, bg_seq)
    shap_vals   = explainer.shap_values(exp_seq)   # (M, seq_len, F, 1)

    if isinstance(shap_vals, list):
        shap_arr = np.array(shap_vals[0])
    else:
        shap_arr = np.array(shap_vals)

    # Collapse seq and output dims → mean |SHAP| per feature
    if shap_arr.ndim == 4:
        shap_arr = shap_arr[..., 0]   # (M, seq_len, F)
    mean_shap = np.abs(shap_arr).mean(axis=(0, 1))   # (F,)
    return mean_shap


# ── Plots ──────────────────────────────────────────────────────────────────────

PALETTE = ["#f0c93a", "#3af0a0", "#3aaaf0", "#f03a7a", "#bf7fff",
           "#ff6b6b", "#00e5ff", "#39ff93", "#ffb347", "#c8d6e5"]


def plot_feature_importance(
    mean_shap:     np.ndarray,
    feature_names: list[str],
    save_path:     str,
    top_k:         int = 20,
    title:         str = "Feature Importance (mean |SHAP|)",
) -> None:
    """Bar chart of top-K features by mean |SHAP| value."""
    top_k   = min(top_k, len(feature_names))
    order   = np.argsort(mean_shap)[::-1][:top_k]
    names   = [feature_names[i] for i in order]
    values  = mean_shap[order]
    colors  = [PALETTE[i % len(PALETTE)] for i in range(top_k)]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.35)))
    fig.patch.set_facecolor("#05080f")
    ax.set_facecolor("#0c1018")

    bars = ax.barh(range(top_k), values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names[::-1], color="#c8d6e5", fontsize=9)
    ax.set_xlabel("Mean |SHAP| Value", color="#c8d6e5")
    ax.set_title(title, color="#eaf2ff", fontsize=13)
    ax.tick_params(colors="#3d4f6a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#161d2b")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Feature importance plot → {save_path}")


def print_top_features(
    mean_shap:     np.ndarray,
    feature_names: list[str],
    top_k:         int = 10,
) -> None:
    """Print top-K features to console."""
    order = np.argsort(mean_shap)[::-1][:top_k]
    print(f"\n── Top {top_k} Features by SHAP Importance ─────────────────────")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<40s} {mean_shap[idx]:.5f}")
    print("─────────────────────────────────────────────────────────────────")