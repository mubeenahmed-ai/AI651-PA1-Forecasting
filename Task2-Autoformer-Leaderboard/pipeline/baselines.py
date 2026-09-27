"""Reference forecasts (not submitted) that put the Autoformer's errors in context."""
from __future__ import annotations

import numpy as np

from .data import HORIZON, N_HISTORY, Split, origins
from .metrics import block_metrics


def baseline_table(raw: np.ndarray, split: Split, seq_len: int = 168, stride: int = 24):
    rows = []
    regions = {"val": (split.train_end, split.val_end), "test": (split.val_end, N_HISTORY)}
    train_mean = raw[:split.train_end].mean()
    train_median = np.median(raw[:split.train_end])
    for region, (start, end) in regions.items():
        starts = origins(start, end, seq_len, HORIZON, stride)
        if not len(starts):
            continue
        target = raw[starts[:, None] + np.arange(HORIZON)]
        history = raw[starts[:, None] + np.arange(-seq_len, 0)]
        forecasts = {
            "train mean": np.full_like(target, train_mean),
            "train median": np.full_like(target, train_median),
            "persistence (last value)": np.repeat(history[:, -1:], HORIZON, axis=1),
            "window mean": np.repeat(history.mean(1, keepdims=True), HORIZON, axis=1),
            "seasonal naive (24)": np.tile(history[:, -24:], HORIZON // 24),
        }
        for name, forecast in forecasts.items():
            rows.append({"model": name, "region": region, **block_metrics(forecast, target)})
    rows.extend(covariate_ridge(split))
    return rows


def _ridge_features(z, marks, starts, history=24):
    """Per-horizon-step design: last ``history`` target values (plain and decayed with the
    step), phase + covariates at that step, covariates at the origin, and the step index."""
    steps = np.arange(HORIZON)
    past = z[starts[:, None] + np.arange(-history, 0)]                      # [n,history]
    past = np.repeat(past[:, None, :], HORIZON, axis=1)                    # [n,H,history]
    decayed = past * np.exp(-steps / 24.0)[None, :, None]
    future = marks[starts[:, None] + steps]                                # [n,H,m]
    origin = np.repeat(marks[starts - 1][:, None, :], HORIZON, axis=1)     # [n,H,m]
    step = np.broadcast_to((steps / HORIZON)[None, :, None], (len(starts), HORIZON, 1))
    ones = np.ones_like(step)
    return np.concatenate([past, decayed, future, origin, step, ones], axis=-1)


def covariate_ridge(split: Split, penalty: float = 1.0, seq_len: int = 168):
    """A 78-parameter linear reference using the same future covariates (not submitted)."""
    from .data import prepare  # local import keeps this module importable on its own
    prepared = prepare(split, exog_mode="future", transform="none")
    z, marks = prepared.scaled, prepared.marks
    train = origins(0, split.train_end, seq_len, HORIZON, 6)
    X = _ridge_features(z, marks, train)
    X = X.reshape(-1, X.shape[-1])
    Y = z[train[:, None] + np.arange(HORIZON)].reshape(-1)
    weights = np.linalg.solve(X.T @ X + penalty * np.eye(X.shape[1]), X.T @ Y)
    rows = []
    for region, (start, end) in {"val": (split.train_end, split.val_end),
                                 "test": (split.val_end, N_HISTORY)}.items():
        starts = origins(start, end, seq_len, HORIZON, 24)
        if not len(starts):
            continue
        forecast = prepared.transform.inverse(_ridge_features(z, marks, starts) @ weights)
        target = prepared.raw[starts[:, None] + np.arange(HORIZON)]
        rows.append({"model": f"covariate ridge ({len(weights)} params)", "region": region,
                     **block_metrics(forecast, target)})
    return rows
