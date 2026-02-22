"""
transformer.py
--------------
Spatio-Temporal Transformer for fatigue drop prediction.

Architecture:
    Input (B, T, F)
        → Linear projection → Positional Encoding
        → N x TransformerEncoderLayer (multi-head attention + FFN)
        → CLS token pooling
        → Dropout → FC head → sigmoid

Design choices:
    - CLS token prepended (BERT-style) for sequence classification
    - Learnable positional encoding (works better for short sequences)
    - Pre-norm (LayerNorm before attention) for stable training
    - Survival head optional: estimates time-to-drop in minutes

IMPORTANT — player-aware sequences:
    TransformerDataset accepts an optional player_ids array.
    When provided, sliding windows are built *within* each player's
    rows only — never crossing player boundaries.  Always pass
    player_ids to avoid mixing timelines from different players.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Positional Encoding ────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable position embeddings — better than sinusoidal for short seqs."""
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


# ── Transformer Block ──────────────────────────────────────────────────────────

class PreNormTransformerBlock(nn.Module):
    """Single Transformer block with pre-LayerNorm (more stable training)."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ── Main Model ─────────────────────────────────────────────────────────────────

class FatigueTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for per-player fatigue drop prediction.

    Parameters
    ----------
    n_features   : number of input features (F)
    d_model      : transformer hidden dimension
    n_heads      : number of attention heads
    n_layers     : number of transformer blocks
    ffn_dim      : feed-forward network hidden size
    dropout      : dropout rate
    max_seq_len  : maximum sequence length (for positional encoding)
    survival     : if True, add a survival head for time-to-drop estimation
    """
    def __init__(
        self,
        n_features:  int,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 3,
        ffn_dim:     int   = 256,
        dropout:     float = 0.2,
        max_seq_len: int   = 64,
        survival:    bool  = False,
    ) -> None:
        super().__init__()
        self.d_model  = d_model
        self.survival = survival

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (seq_len + 1 for CLS)
        self.pos_enc = LearnablePositionalEncoding(max_seq_len + 1, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PreNormTransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Optional survival head (time-to-drop in window units)
        if survival:
            self.surv_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus(),   # positive output only
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, F)

        Returns
        -------
        logits      : (B,)          if survival=False
        (logits, t) : (B,), (B,)   if survival=True
        """
        B = x.size(0)

        x = self.input_proj(x)                       # (B, T, D)

        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)             # (B, T+1, D)

        x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_repr = self.dropout(x[:, 0, :])          # (B, D)
        logits   = self.cls_head(cls_repr).squeeze(-1)  # (B,)

        if self.survival:
            t_pred = self.surv_head(cls_repr).squeeze(-1)  # (B,)
            return logits, t_pred

        return logits


# ── Sequence Dataset ───────────────────────────────────────────────────────────

class TransformerDataset(torch.utils.data.Dataset):
    """
    Sliding-window sequence dataset for the Transformer.

    When player_ids is provided (strongly recommended), windows are
    built within each player's contiguous block of rows only.
    This prevents sequences from spanning across player boundaries,
    which would mix unrelated timelines and corrupt temporal signal.

    Parameters
    ----------
    X          : (N, F) feature array
    y          : (N,)   label array
    seq_len    : number of timesteps per sequence
    stride     : step between consecutive windows (default 1)
    player_ids : (N,) integer player identifier per row.
                 Pass ds.player_ids_train / _val / _test from FatigueDataset.
    """
    def __init__(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        seq_len:    int = 16,
        stride:     int = 1,
        player_ids: np.ndarray | None = None,
    ) -> None:
        self.seqs:   list[np.ndarray] = []
        self.labels: list[int]        = []

        if player_ids is None:
            # Fallback: treat entire array as one block (legacy behaviour).
            # WARNING: sequences will cross player boundaries if X contains
            # rows from multiple players. Always pass player_ids.
            groups = [np.arange(len(X))]
        else:
            unique_players = np.unique(player_ids)
            groups = [np.where(player_ids == pid)[0] for pid in unique_players]

        n_cross_boundary = 0
        for idx in groups:
            X_p = X[idx]
            y_p = y[idx]
            n   = len(X_p)
            if n < seq_len:
                # Player has fewer rows than seq_len — skip entirely.
                n_cross_boundary += 1
                continue
            for start in range(0, n - seq_len + 1, stride):
                self.seqs.append(X_p[start : start + seq_len])
                self.labels.append(int(y_p[start + seq_len - 1]))

        if n_cross_boundary > 0 and player_ids is not None:
            print(f"[TransformerDataset] Skipped {n_cross_boundary} player(s) "
                  f"with fewer than {seq_len} rows.")

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.seqs[idx],   dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def make_transformer_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    seq_len:           int                = 16,
    batch_size:        int                = 64,
    stride:            int                = 1,
    player_ids_train:  np.ndarray | None  = None,
    player_ids_val:    np.ndarray | None  = None,
    player_ids_test:   np.ndarray | None  = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders with player-aware sequence windows.

    Always pass player_ids_* from FatigueDataset to avoid cross-player
    sequence contamination.
    """
    train_ds = TransformerDataset(X_train, y_train, seq_len, stride, player_ids_train)
    val_ds   = TransformerDataset(X_val,   y_val,   seq_len, stride, player_ids_val)
    test_ds  = TransformerDataset(X_test,  y_test,  seq_len, stride, player_ids_test)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )


# ── Save / Load ────────────────────────────────────────────────────────────────

def save_transformer(model: FatigueTransformer, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "n_features": model.input_proj[0].in_features,
            "d_model":    model.d_model,
            "survival":   model.survival,
        },
    }, path)
    print(f"[Transformer] Saved → {path}")


def load_transformer(
    path: str,
    n_features: int,
    **kwargs: int | float | bool,
) -> FatigueTransformer:
    ckpt  = torch.load(path, map_location="cpu")
    cfg   = ckpt.get("config", {})
    cfg.update(kwargs)
    cfg["n_features"] = n_features
    model = FatigueTransformer(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model