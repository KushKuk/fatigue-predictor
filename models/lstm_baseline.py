"""
lstm_baseline.py
-----------------
LSTM sequence model for fatigue drop prediction.

Architecture:
    Input  → LayerNorm → BiLSTM (2 layers) → Dropout
           → Attention pooling → FC head → sigmoid

The sequence dimension is a sliding window of T consecutive
feature vectors per player (chunked from the flat feature matrix).
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Dataset ────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Wraps flat (N, F) feature arrays into overlapping sequences of length seq_len.
    Each sample is (seq_len, F) → label scalar.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 10,
        stride:  int = 1,
    ) -> None:
        self.seq_len = seq_len
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []

        for start in range(0, len(X) - seq_len + 1, stride):
            self.sequences.append(X[start : start + seq_len])
            self.labels.append(int(y[start + seq_len - 1]))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],    dtype=torch.float32)
        return x, y


def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    seq_len:    int = 10,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SequenceDataset(X_train, y_train, seq_len)
    val_ds   = SequenceDataset(X_val,   y_val,   seq_len)
    test_ds  = SequenceDataset(X_test,  y_test,  seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ── Model ──────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Soft attention over LSTM output timesteps → single context vector."""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        weights = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * weights).sum(dim=1)                # (B, H)


class FatigueLSTM(nn.Module):
    """
    Bidirectional LSTM with attention pooling for binary drop prediction.

    Parameters
    ----------
    n_features  : number of input features (F)
    hidden_dim  : LSTM hidden size per direction
    num_layers  : number of stacked LSTM layers
    dropout     : dropout probability
    """
    def __init__(
        self,
        n_features: int,
        hidden_dim: int  = 128,
        num_layers: int  = 2,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2   # bidirectional

        self.attn_pool = AttentionPool(lstm_out_dim)
        self.dropout   = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.norm(x)
        out, _ = self.lstm(x)       # (B, T, 2*H)
        ctx = self.attn_pool(out)   # (B, 2*H)
        ctx = self.dropout(ctx)
        return self.head(ctx).squeeze(-1)   # (B,) logits


# ── Trainer ────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_lstm(
    model:        FatigueLSTM,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    pos_weight:   float = 1.0,
    lr:           float = 3e-4,
    epochs:       int   = 50,
    patience:     int   = 7,
    device:       str   = "cpu",
    verbose:      bool  = True,
) -> list[dict[str, float]]:
    """
    Train FatigueLSTM with BCE + pos_weight. Returns history list.
    Restores best checkpoint on early stop.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper   = EarlyStopping(patience=patience)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()

        train_loss /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader),   1)
        scheduler.step()

        rec = {"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)}
        history.append(rec)

        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if stopper.step(val_loss, model):
            if verbose:
                print(f"  Early stop at epoch {epoch}. Best val: {stopper.best_loss:.4f}")
            break

    # Restore best weights
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return history


def evaluate_lstm(
    model:       FatigueLSTM,
    loader:      DataLoader,
    device:      str = "cpu",
    split_name:  str = "Test",
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    model.eval()
    all_proba: list[float] = []
    all_y:     list[int]   = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            proba  = torch.sigmoid(logits).cpu().numpy()
            all_proba.extend(proba.tolist())
            all_y.extend(yb.numpy().astype(int).tolist())

    y_arr = np.array(all_y)
    p_arr = np.array(all_proba)

    auc   = roc_auc_score(y_arr, p_arr)
    ap    = average_precision_score(y_arr, p_arr)
    brier = brier_score_loss(y_arr, p_arr)

    print(f"\n── {split_name} Results (LSTM) ──────────────────────────────────")
    print(f"  ROC AUC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(f"  Brier    : {brier:.4f}")

    return {"auc": float(auc), "avg_precision": float(ap), "brier": float(brier)}


def save_lstm(model: FatigueLSTM, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[LSTM] Saved → {path}")


def load_lstm(
    path:       str,
    n_features: int,
    hidden_dim: int   = 128,
    num_layers: int   = 2,
    dropout:    float = 0.3,
) -> FatigueLSTM:
    model = FatigueLSTM(
        n_features=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.dataset import load_dataset

    parquet = sys.argv[1] if len(sys.argv) > 1 else \
        "../../phase1/outputs/phase1_features_match3773386.parquet"

    ds = load_dataset(parquet)
    train_loader, val_loader, test_loader = make_loaders(
        ds.X_train, ds.y_train,
        ds.X_val,   ds.y_val,
        ds.X_test,  ds.y_test,
        seq_len=10, batch_size=64,
    )

    model = FatigueLSTM(n_features=ds.n_features)
    print(model)

    history = train_lstm(model, train_loader, val_loader,
                         pos_weight=ds.pos_weight(), epochs=20)
    metrics = evaluate_lstm(model, test_loader)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    save_lstm(model, os.path.join(out_dir, "lstm_baseline.pt"))