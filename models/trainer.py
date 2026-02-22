"""
trainer.py
----------
Training loop for FatigueTransformer.

Features:
    - Weighted BCE loss for class imbalance
    - Optional survival loss (MSE on time-to-drop)
    - Cosine annealing with warm restarts
    - Early stopping with best-checkpoint restore
    - Mixed precision (torch.cuda.amp) when CUDA available
    - Full training history returned for plotting
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.transformer import FatigueTransformer


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best       = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Combined Loss ──────────────────────────────────────────────────────────────

class FatigueLoss(nn.Module):
    """
    BCE classification loss + optional MSE survival loss.

    total_loss = bce_loss + survival_weight * mse_loss
    """
    def __init__(
        self,
        pos_weight:      float = 1.0,
        survival_weight: float = 0.1,
        device:          str   = "cpu",
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        self.mse             = nn.MSELoss()
        self.survival_weight = survival_weight

    def forward(
        self,
        logits:    torch.Tensor,
        targets:   torch.Tensor,
        t_pred:    Optional[torch.Tensor] = None,
        t_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.bce(logits, targets)
        if t_pred is not None and t_targets is not None:
            loss = loss + self.survival_weight * self.mse(t_pred, t_targets)
        return loss


# ── Trainer ────────────────────────────────────────────────────────────────────

def train_transformer(
    model:           FatigueTransformer,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    pos_weight:      float = 1.0,
    lr:              float = 1e-4,
    weight_decay:    float = 1e-4,
    epochs:          int   = 80,
    patience:        int   = 10,
    warmup_epochs:   int   = 5,
    survival_weight: float = 0.0,
    device:          str   = "cpu",
    verbose:         bool  = True,
) -> list[dict[str, float]]:
    """
    Train FatigueTransformer. Returns history list of dicts.

    Parameters
    ----------
    warmup_epochs   : linear LR warmup before cosine annealing
    survival_weight : weight on survival MSE loss (0 = disabled)
    """
    model = model.to(device)
    criterion = FatigueLoss(pos_weight, survival_weight, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Linear warmup then cosine annealing
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    stopper   = EarlyStopping(patience=patience)
    scaler    = GradScaler() if device == "cuda" else None

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    out    = model(xb)
                    logits = out[0] if isinstance(out, tuple) else out
                    t_pred = out[1] if isinstance(out, tuple) and model.survival else None
                    loss   = criterion(logits, yb, t_pred, None)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out    = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                t_pred = out[1] if isinstance(out, tuple) and model.survival else None
                loss   = criterion(logits, yb, t_pred, None)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += float(loss.item())

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out    = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                val_loss += float(criterion(logits, yb).item())

        train_loss /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader),   1)
        scheduler.step()

        elapsed = time.time() - t0
        rec: dict[str, float] = {
            "epoch":      float(epoch),
            "train_loss": float(train_loss),
            "val_loss":   float(val_loss),
            "lr":         float(optimizer.param_groups[0]["lr"]),
            "elapsed_s":  float(elapsed),
        }
        history.append(rec)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"lr {rec['lr']:.2e} | {elapsed:.1f}s")

        if stopper.step(val_loss, model):
            if verbose:
                print(f"  ✓ Early stop at epoch {epoch}. "
                      f"Best val loss: {stopper.best:.4f}")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        if verbose:
            print("  ✓ Best weights restored.")

    return history


# ── Inference ──────────────────────────────────────────────────────────────────

def get_probabilities(
    model:  FatigueTransformer,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.

    Returns
    -------
    proba : np.ndarray shape (N,)  — predicted probabilities
    y     : np.ndarray shape (N,)  — true labels
    """
    model.eval()
    all_proba: list[float] = []
    all_y:     list[int]   = []

    with torch.no_grad():
        for xb, yb in loader:
            out    = model(xb.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            proba  = torch.sigmoid(logits).cpu().numpy()
            all_proba.extend(proba.tolist())
            all_y.extend(yb.numpy().astype(int).tolist())

    return np.array(all_proba, dtype=np.float32), np.array(all_y, dtype=np.int64)