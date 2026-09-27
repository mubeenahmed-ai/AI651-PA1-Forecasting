"""The three metrics of Section 2.6, per 168-step block and pooled."""
from __future__ import annotations

import numpy as np


def smape(prediction, target):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    denominator = np.abs(target) + np.abs(prediction)
    ratio = np.where(denominator > 0, 2 * np.abs(prediction - target) / np.where(
        denominator > 0, denominator, 1.0), 0.0)            # 0/0 (both zero) counts as exact
    return 100 * ratio.mean(axis=-1)


def block_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """prediction/target [n_blocks, 168] in physical units.

    ``RMSE`` is the mean over blocks of each block's RMSE: the expected value of the
    leaderboard's single-block score. ``RMSE pooled`` is the RMSE over all points.
    """
    error = prediction - target
    per_block_rmse = np.sqrt(np.mean(error ** 2, axis=-1))
    return {
        "blocks": int(len(target)),
        "RMSE": float(per_block_rmse.mean()),
        "RMSE block std": float(per_block_rmse.std()),
        "RMSE pooled": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "sMAPE": float(np.mean(smape(prediction, target))),
        # Pooled RMSE at selected horizon steps (1-based): shows whether recent history is used.
        **{f"RMSE step {h}": float(np.sqrt(np.mean(error[:, h - 1] ** 2)))
           for h in (1, 6, 24, 168) if h <= error.shape[-1]},
    }
