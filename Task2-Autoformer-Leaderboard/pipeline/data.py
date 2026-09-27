"""Loading, feature engineering, chronological splits and window sampling for Task 2.

Every statistic (target transform scale, covariate means/stds) is estimated on the
training split only. Window origins are indexed by the first forecast step ``o``:
    encoder input  = y[o - seq_len : o]
    decoder marks  = marks[o - label_len : o + pred_len]
    target         = y[o : o + pred_len]
A window belongs to a split when its whole target lies inside that split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HORIZON = 168
N_HISTORY = 43_656
N_TOTAL = 43_824

# Periods recovered from the periodogram of the data (see scripts/explore.py and
# results/figures/periodogram.pdf): a 24-step cycle in the target and every continuous
# covariate, and a ~8766-step cycle (the dominant peak of features A-C). Phase features are
# built from the integer index only, so they are known for every future step.
PHASE_PERIODS = (24.0, 8766.0)

FEATURES = [f"feature_{c}" for c in "ABCDEFGHIJ"]
CONTINUOUS = FEATURES[:6]
LOG_FEATURES = ["feature_D", "feature_E", "feature_F"]   # non-negative, heavy-tailed
BINARY = FEATURES[6:]                                    # mutually exclusive indicators
FEATURE_GROUPS = {                                       # used by the feature ablation
    "A": ["feature_A"], "B": ["feature_B"], "C": ["feature_C"],
    "D+GHIJ": ["feature_D", *BINARY], "E+F": ["feature_E", "feature_F"],
}
EXOG_MODES = ("none", "past", "future")
# Where the covariates enter the Autoformer: as additive "mark" embeddings (the reference
# code's route for known inputs), or as extra value channels of the encoder and decoder.
EXOG_ENTRIES = ("mark", "value")


@dataclass(frozen=True)
class Split:
    """Target-time boundaries: train [0, train_end), validation [train_end, val_end),
    test [val_end, N_HISTORY)."""
    name: str
    train_end: int
    val_end: int


# Model selection: chronological 80 / 10 / 10 of the released history.
EXPERIMENT_SPLIT = Split("experiment", int(0.8 * N_HISTORY), int(0.9 * N_HISTORY))
# Final fit: train on everything except the last 13 weeks, which drive early stopping only.
PRODUCTION_SPLIT = Split("production", N_HISTORY - 13 * HORIZON, N_HISTORY)


def load_raw(data_dir: Path = DATA_DIR):
    target = pd.read_csv(data_dir / "student_train.csv")
    external = pd.read_csv(data_dir / "optional_external_data.csv")
    test = pd.read_csv(data_dir / "student_test.csv")
    assert len(target) == N_HISTORY and target["value"].notna().all()
    assert len(external) == N_TOTAL
    assert np.array_equal(target.time_idx, np.arange(1, N_HISTORY + 1))
    assert np.array_equal(external.time_idx, np.arange(1, N_TOTAL + 1))
    assert np.array_equal(test.time_idx, np.arange(N_HISTORY + 1, N_TOTAL + 1))
    assert (external[BINARY].sum(axis=1) == 1).all(), "indicators must be one-hot"
    return target["value"].to_numpy(np.float64), external


def phase_features(periods=PHASE_PERIODS, length: int = N_TOTAL) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    columns = [np.stack([np.sin(2 * np.pi * t / p), np.cos(2 * np.pi * t / p)], 1) for p in periods]
    return np.concatenate(columns, axis=1) if columns else np.zeros((length, 0))


class TargetTransform:
    """Optional log1p followed by a z-score fitted on the training split."""

    def __init__(self, kind: str, train_values: np.ndarray):
        if kind not in {"none", "log1p"}:
            raise ValueError(kind)
        self.kind = kind
        base = self._forward_base(train_values)
        self.mean, self.std = float(base.mean()), float(base.std())

    def _forward_base(self, y):
        return np.log1p(y) if self.kind == "log1p" else np.asarray(y, dtype=np.float64)

    def forward(self, y):
        return (self._forward_base(y) - self.mean) / self.std

    def inverse(self, z):
        base = np.asarray(z, dtype=np.float64) * self.std + self.mean
        # The target is non-negative, so the inverse is clipped at zero.
        return np.clip(np.expm1(base) if self.kind == "log1p" else base, 0.0, None)


def covariate_matrix(external: pd.DataFrame, train_end: int, columns) -> np.ndarray:
    """Scaled covariates [N_TOTAL, len(columns)] using training-split statistics only."""
    frame = external[list(columns)].astype(np.float64).copy()
    for column in columns:
        if column in LOG_FEATURES:
            frame[column] = np.log1p(frame[column])
        if column in CONTINUOUS:
            train = frame[column].iloc[:train_end]
            frame[column] = (frame[column] - train.mean()) / train.std()
    return frame.to_numpy(np.float64)


@dataclass
class Prepared:
    """Arrays for one split and one covariate configuration."""
    split: Split
    raw: np.ndarray            # [N_HISTORY] target in physical units
    scaled: np.ndarray         # [N_HISTORY] transformed target
    marks: np.ndarray          # [N_TOTAL, m] phase features (+ covariates)
    n_exog: int                # trailing columns of ``marks`` that are external covariates
    exog_mode: str
    transform: TargetTransform
    exog_entry: str = "mark"

    @property
    def value_channels(self) -> int:
        return 1 + (self.n_exog if self.exog_entry == "value" else 0)

    @property
    def mark_dim(self) -> int:
        return self.marks.shape[1] - (self.n_exog if self.exog_entry == "value" else 0)


def prepare(split: Split, exog_mode: str = "future", transform: str = "log1p",
            exog_columns=FEATURES, phase_periods=PHASE_PERIODS, exog_entry: str = "mark",
            data_dir: Path = DATA_DIR) -> Prepared:
    if exog_mode not in EXOG_MODES or exog_entry not in EXOG_ENTRIES:
        raise ValueError((exog_mode, exog_entry))
    raw, external = load_raw(data_dir)
    target_transform = TargetTransform(transform, raw[:split.train_end])
    marks = [phase_features(phase_periods)]
    columns = list(exog_columns) if exog_mode != "none" else []
    if columns:
        marks.append(covariate_matrix(external, split.train_end, columns))
    return Prepared(split, raw, target_transform.forward(raw), np.concatenate(marks, 1),
                    len(columns), exog_mode, target_transform, exog_entry)


def origins(start: int, end: int, seq_len: int, pred_len: int = HORIZON,
            stride: int = 1) -> np.ndarray:
    """Forecast origins whose whole target lies in [start, end)."""
    return np.arange(max(start, seq_len), end - pred_len + 1, stride, dtype=np.int64)


class WindowSampler:
    """Builds (x_enc, mark_enc, mark_dec, dec_exog, target) batches on the device.

    With ``exog_entry == "value"`` the covariates are appended to the encoder values and
    returned as ``dec_exog`` for the decoder; otherwise they stay in the marks and
    ``dec_exog`` is None.
    """

    def __init__(self, prepared: Prepared, seq_len: int, label_len: int,
                 pred_len: int = HORIZON, device="cpu"):
        self.p, self.seq_len, self.label_len, self.pred_len = prepared, seq_len, label_len, pred_len
        self.device = torch.device(device)
        padded = np.full(N_TOTAL, np.nan)
        padded[:N_HISTORY] = prepared.scaled
        self.series = torch.as_tensor(padded, dtype=torch.float32, device=self.device)
        self.marks = torch.as_tensor(prepared.marks, dtype=torch.float32, device=self.device)
        self.enc_offsets = torch.arange(-seq_len, 0, device=self.device)
        self.dec_offsets = torch.arange(-label_len, pred_len, device=self.device)
        self.target_offsets = torch.arange(pred_len, device=self.device)

    def batch(self, origin: np.ndarray | torch.Tensor):
        o = torch.as_tensor(origin, device=self.device)[:, None]
        n = self.p.n_exog
        x_enc = self.series[o + self.enc_offsets][..., None]              # [B,seq,1]
        mark_enc = self.marks[o + self.enc_offsets]                        # [B,seq,m]
        mark_dec = self.marks[o + self.dec_offsets].clone()                # [B,label+pred,m]
        if self.p.exog_mode == "past" and n:
            # Covariates known only up to the forecast origin: hold the last observed value
            # flat across the horizon instead of revealing the future measurements.
            last = self.marks[o[:, 0] - 1, -n:]                            # [B,n_exog]
            mark_dec[:, self.label_len:, -n:] = last[:, None, :]
        dec_exog = None
        if self.p.exog_entry == "value" and n:
            x_enc = torch.cat([x_enc, mark_enc[..., -n:]], dim=-1)          # [B,seq,1+n]
            dec_exog = mark_dec[..., -n:]                                  # [B,label+pred,n]
            mark_enc, mark_dec = mark_enc[..., :-n], mark_dec[..., :-n]
        target_index = o + self.target_offsets
        target = self.series[target_index.clamp(max=N_TOTAL - 1)]          # NaN beyond history
        return x_enc, mark_enc, mark_dec, dec_exog, target
