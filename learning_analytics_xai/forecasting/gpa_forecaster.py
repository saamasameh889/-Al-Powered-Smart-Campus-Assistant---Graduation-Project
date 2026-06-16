"""
gpa_forecaster.py — GPA Trajectory Forecasting  (Product C)
═══════════════════════════════════════════════════════════════════════════════
Sequence-to-Sequence LSTM with additive attention and calibrated
quantile (pinball) regression.

Architecture
------------
  Static encoder:   FC(static_dim → 64) → LayerNorm → GELU
                    FC(64 → 32) → LayerNorm → GELU                  [32-d]

  Context injection: broadcast static_enc across the T time steps and
                     concatenate with raw temporal features           [T × (3+32)]

  LSTM encoder:     2-layer LSTM(35 → 128 → 64, dropout=0.25)       [T × 64]

  Attention:        FC(64 → 1) → softmax over T → weighted sum       [64-d]

  Decoder:          FC(96 → 64) → LayerNorm → GELU → Dropout(0.25)
                    FC(64 → 32) → GELU
                    FC(32 → H×3) → reshape to (H, 3)                 [H, 3]
                    Output[:,0] = q10, [:,1] = q50 (median), [:,2] = q90

Training
--------
  Loss        : Pinball loss (quantile regression) for q ∈ {0.10, 0.50, 0.90}
  Optimizer   : AdamW(lr=1e-3, weight_decay=1e-4)
  Scheduler   : CosineAnnealingLR(T_max=n_epochs)
  Regulariser : Gradient clipping max_norm=1.0  (LSTM stability)
  Early stop  : patience=12 epochs on validation loss
  Train/val   : 80 / 20 stratified by student risk level

Public API
----------
    forecaster = GPAForecaster()
    history    = forecaster.fit(data)     # data from sequence_generator
    forecaster.save(path)

    forecaster = GPAForecaster.load(path)
    result     = forecaster.predict(static_x, temporal_x)
    # → {gpa_median, gpa_q10, gpa_q90, semesters_ahead}
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)

QUANTILES = [0.10, 0.50, 0.90]


# ══════════════════════════════════════════════════════════════════════════════
#  Model definition
# ══════════════════════════════════════════════════════════════════════════════

class _StaticEncoder(nn.Module):
    """Maps static student features to a 32-d context vector."""

    def __init__(self, static_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 32)


class _LSTMEncoder(nn.Module):
    """Two-layer LSTM that encodes the context-injected temporal sequence."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(x)   # (B, T, hidden), (h_n, c_n)


class _AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over the LSTM output sequence.
    Produces a fixed-size context vector by computing a weighted sum
    of all hidden states, where weights are learned from content.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, hidden)
        raw    = self.score(lstm_out)          # (B, T, 1)
        weights = torch.softmax(raw, dim=1)    # (B, T, 1)
        context = (weights * lstm_out).sum(1)  # (B, hidden)
        return context


class GPAForecasterLSTM(nn.Module):
    """
    Full sequence-to-sequence LSTM with additive attention.

    Input
    -----
    static_x   : (B, static_dim)          — student-level static features
    temporal_x : (B, T, temporal_dim)     — T past semesters of [gpa, load, risk]

    Output
    ------
    (B, H, 3)  — H horizon steps × 3 quantiles [q10, q50, q90]
    """

    def __init__(
        self,
        static_dim: int   = 6,
        temporal_dim: int = 3,
        lstm_hidden: int  = 128,
        lstm_layers: int  = 2,
        horizon: int      = 3,
        dropout: float    = 0.25,
    ) -> None:
        super().__init__()
        self.horizon = horizon

        self.static_enc = _StaticEncoder(static_dim, dropout)

        # LSTM input = raw temporal (3) + static context (32) = 35
        self.lstm_enc = _LSTMEncoder(
            input_dim  = temporal_dim + 32,
            hidden     = lstm_hidden,
            num_layers = lstm_layers,
            dropout    = dropout,
        )

        self.attention = _AdditiveAttention(lstm_hidden)

        # Decoder input = attention context (lstm_hidden) + static enc (32)
        dec_in = lstm_hidden + 32
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, horizon * len(QUANTILES)),
        )

    def forward(
        self,
        static_x:   torch.Tensor,
        temporal_x: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = temporal_x.shape

        # Static context vector
        static_enc = self.static_enc(static_x)          # (B, 32)

        # Inject static context at every time step
        ctx_expanded = static_enc.unsqueeze(1).expand(-1, T, -1)   # (B, T, 32)
        lstm_in = torch.cat([temporal_x, ctx_expanded], dim=-1)    # (B, T, 35)

        lstm_out, _ = self.lstm_enc(lstm_in)                        # (B, T, hidden)
        context = self.attention(lstm_out)                           # (B, hidden)

        dec_in = torch.cat([context, static_enc], dim=-1)           # (B, hidden+32)
        raw    = self.decoder(dec_in)                                # (B, H×3)

        return raw.view(B, self.horizon, len(QUANTILES))            # (B, H, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  Loss
# ══════════════════════════════════════════════════════════════════════════════

def _pinball_loss(
    predictions: torch.Tensor,   # (B, H, 3)
    targets:     torch.Tensor,   # (B, H)
    quantiles:   list[float] = QUANTILES,
) -> torch.Tensor:
    """
    Pinball (quantile regression) loss, averaged over H and Q.

    For quantile q:
        L_q(y, ŷ) = q*(y - ŷ)  if y > ŷ
                  = (q-1)*(y - ŷ) otherwise
    """
    targets_exp = targets.unsqueeze(-1).expand_as(predictions)   # (B, H, 3)
    errors      = targets_exp - predictions

    losses = []
    for i, q in enumerate(quantiles):
        err_q  = errors[..., i]                             # (B, H)
        loss_q = torch.where(err_q >= 0, q * err_q, (q - 1) * err_q)
        losses.append(loss_q.mean())

    return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  Training wrapper
# ══════════════════════════════════════════════════════════════════════════════

class GPAForecaster:
    """
    Training + inference wrapper around GPAForecasterLSTM.

    Usage
    -----
        fc = GPAForecaster()
        history = fc.fit(data, epochs=60, batch_size=256, lr=1e-3)
        fc.save("models/gpa_forecaster.pt")

        fc = GPAForecaster.load("models/gpa_forecaster.pt")
        result = fc.predict(static_x, temporal_x)
    """

    def __init__(
        self,
        static_dim:   int   = 6,
        temporal_dim: int   = 3,
        lstm_hidden:  int   = 128,
        lstm_layers:  int   = 2,
        horizon:      int   = 3,
        dropout:      float = 0.25,
    ) -> None:
        self.horizon      = horizon
        self.static_dim   = static_dim
        self.temporal_dim = temporal_dim
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GPAForecasterLSTM(
            static_dim   = static_dim,
            temporal_dim = temporal_dim,
            lstm_hidden  = lstm_hidden,
            lstm_layers  = lstm_layers,
            horizon      = horizon,
            dropout      = dropout,
        ).to(self.device)

        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self._is_fitted: bool = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        data:        dict,
        epochs:      int   = 80,
        batch_size:  int   = 256,
        lr:          float = 1e-3,
        weight_decay:float = 1e-4,
        val_frac:    float = 0.20,
        patience:    int   = 12,
        progress_cb  = None,
    ) -> dict:
        """
        Train the model on sequences from sequence_generator.generate_sequences().

        Parameters
        ----------
        data        : dict returned by generate_sequences()
        epochs      : max training epochs
        batch_size  : mini-batch size
        lr          : initial learning rate (AdamW)
        weight_decay: L2 regularisation
        val_frac    : fraction of data held out for early stopping
        patience    : early stopping patience (epochs without val improvement)
        progress_cb : optional callable(epoch, total, train_loss, val_loss)

        Returns
        -------
        history dict with train_losses, val_losses, best_epoch
        """
        static_t   = torch.tensor(data["static"],   dtype=torch.float32)
        temporal_t = torch.tensor(data["temporal"],  dtype=torch.float32)
        target_t   = torch.tensor(data["target"],    dtype=torch.float32)

        dataset = TensorDataset(static_t, temporal_t, target_t)
        n_val   = int(len(dataset) * val_frac)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

        optimiser = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr * 0.01)

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── training ──
            self.model.train()
            train_loss = 0.0
            for s_b, t_b, y_b in train_dl:
                s_b = s_b.to(self.device)
                t_b = t_b.to(self.device)
                y_b = y_b.to(self.device)

                optimiser.zero_grad()
                pred = self.model(s_b, t_b)
                loss = _pinball_loss(pred, y_b)
                loss.backward()

                # gradient clipping — crucial for LSTM stability
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimiser.step()
                train_loss += loss.item() * len(s_b)

            train_loss /= n_train

            # ── validation ──
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for s_b, t_b, y_b in val_dl:
                    s_b = s_b.to(self.device)
                    t_b = t_b.to(self.device)
                    y_b = y_b.to(self.device)
                    pred    = self.model(s_b, t_b)
                    val_loss += _pinball_loss(pred, y_b).item() * len(s_b)
            val_loss /= n_val

            scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d — train=%.4f  val=%.4f  %.1fs",
                epoch, epochs, train_loss, val_loss, elapsed,
            )

            if progress_cb is not None:
                progress_cb(epoch, epochs, train_loss, val_loss)

            # early stopping
            if val_loss < best_val - 1e-5:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        best_epoch = int(np.argmin(self.val_losses)) + 1
        self._is_fitted = True

        return {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "best_epoch":   best_epoch,
            "best_val_loss": best_val,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        static_x:   np.ndarray,   # (static_dim,) or (N, static_dim)
        temporal_x: np.ndarray,   # (history_len, temporal_dim) or (N, T, D)
    ) -> dict:
        """
        Predict GPA for the next `horizon` semesters.

        Inputs are un-normalised raw feature vectors:
          static_x[0]  = programme_enc (0–1)
          static_x[1]  = school_enc    (0–1)
          static_x[2]  = avg_attendance / 100
          static_x[3]  = avg_final / 100
          static_x[4]  = failed_ratio
          static_x[5]  = difficulty_index (0–1)

          temporal_x[t, 0] = gpa_norm (gpa / 4.0)
          temporal_x[t, 1] = load_norm (credits / 24.0)
          temporal_x[t, 2] = risk_flag  (1.0=high-risk, 0.5=medium, 0.0=low)

        Returns
        -------
        dict with:
          'gpa_q10'    : np.ndarray (H,) — 10th percentile forecast  ×4.0 (GPA scale)
          'gpa_median' : np.ndarray (H,) — median forecast            ×4.0
          'gpa_q90'    : np.ndarray (H,) — 90th percentile forecast   ×4.0
          'gpa_norm_q10', 'gpa_norm_median', 'gpa_norm_q90' — [0,1] normalised
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load() before predict().")

        s = np.asarray(static_x,   dtype=np.float32)
        t = np.asarray(temporal_x, dtype=np.float32)

        if s.ndim == 1:
            s = s[np.newaxis, :]
        if t.ndim == 2:
            t = t[np.newaxis, :, :]

        s_t = torch.tensor(s).to(self.device)
        t_t = torch.tensor(t).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(s_t, t_t)   # (N, H, 3)

        out_np = out.cpu().numpy()
        if out_np.shape[0] == 1:
            out_np = out_np[0]           # (H, 3)

        # clamp outputs to valid GPA range [0, 1] in normalised space
        out_np = np.clip(out_np, 0.0, 1.0)

        # ensure quantile ordering: q10 ≤ q50 ≤ q90
        q10 = out_np[:, 0]
        q50 = out_np[:, 1]
        q90 = out_np[:, 2]
        q50 = np.maximum(q50, q10)
        q90 = np.maximum(q90, q50)

        return {
            "gpa_q10":         q10 * 4.0,
            "gpa_median":      q50 * 4.0,
            "gpa_q90":         q90 * 4.0,
            "gpa_norm_q10":    q10,
            "gpa_norm_median": q50,
            "gpa_norm_q90":    q90,
        }

    def predict_what_if(
        self,
        base_static:    np.ndarray,
        base_temporal:  np.ndarray,
        attendance_pct: Optional[float] = None,
        load_credits:   Optional[float] = None,
    ) -> dict:
        """
        Re-run prediction with modified static/temporal assumptions.

        Modifies the LAST temporal step (most recent semester) in-place
        with the new attendance / load parameters to simulate what-if.
        """
        s = base_static.copy()
        t = base_temporal.copy()

        if attendance_pct is not None:
            s[2] = float(attendance_pct) / 100.0  # static attendance feature
        if load_credits is not None:
            t[-1, 1] = float(load_credits) / 24.0  # load_norm in last time step

        return self.predict(s, t)

    # ── RMSE helpers (for display) ────────────────────────────────────────────

    def evaluate_sequences(self, data: dict) -> dict[str, float]:
        """
        Compute RMSE and MAE on the median prediction (q50) over a dataset.
        Values reported in GPA units (0–4.0 scale).
        """
        static_t   = torch.tensor(data["static"],   dtype=torch.float32)
        temporal_t = torch.tensor(data["temporal"],  dtype=torch.float32)
        target_t   = data["target"]   # (N, H) in [0,1]

        dl = DataLoader(
            TensorDataset(static_t, temporal_t),
            batch_size=512, shuffle=False,
        )
        preds = []
        self.model.eval()
        with torch.no_grad():
            for s_b, t_b in dl:
                out = self.model(s_b.to(self.device), t_b.to(self.device))
                preds.append(out[:, :, 1].cpu().numpy())   # q50

        pred_q50 = np.concatenate(preds, axis=0)           # (N, H)
        targets  = target_t                                  # (N, H) in [0,1]

        diff = (pred_q50 - targets) * 4.0
        return {
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "mae":  float(np.abs(diff).mean()),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state":  self.model.state_dict(),
                "config": {
                    "static_dim":   self.static_dim,
                    "temporal_dim": self.temporal_dim,
                    "lstm_hidden":  self.model.lstm_enc.lstm.hidden_size,
                    "lstm_layers":  self.model.lstm_enc.lstm.num_layers,
                    "horizon":      self.horizon,
                    "dropout":      self.model.static_enc.net[3].p,
                },
                "train_losses": self.train_losses,
                "val_losses":   self.val_losses,
            },
            path,
        )
        logger.info("GPAForecaster saved → %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "GPAForecaster":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]

        fc = cls(
            static_dim   = cfg["static_dim"],
            temporal_dim = cfg["temporal_dim"],
            lstm_hidden  = cfg["lstm_hidden"],
            lstm_layers  = cfg["lstm_layers"],
            horizon      = cfg["horizon"],
            dropout      = cfg.get("dropout", 0.25),
        )
        fc.model.load_state_dict(ckpt["model_state"])
        fc.model.eval()
        fc.train_losses = ckpt.get("train_losses", [])
        fc.val_losses   = ckpt.get("val_losses",   [])
        fc._is_fitted   = True
        logger.info("GPAForecaster loaded from %s", path)
        return fc
