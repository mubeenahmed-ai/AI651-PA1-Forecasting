"""Named configurations for the validation study. Each one differs from BASE in one choice.

BASE was fixed after a first exploratory round (results/exploratory/): the reference-style
setup (log1p target, covariates as additive marks, lr 1e-3) overfitted within 1-2 epochs and
was unstable across seeds; raw-scale targets and covariates as value channels were better
and far more stable. Every alternative below is re-measured against the revised BASE.
"""
from __future__ import annotations

from dataclasses import replace

from autoformer import AutoformerConfig

from .data import FEATURE_GROUPS, FEATURES, PHASE_PERIODS
from .train import TrainConfig

BASE_MODEL = AutoformerConfig(seq_len=168, label_len=48, pred_len=168, d_model=32, d_ff=64,
                              heads=4, e_layers=1, d_layers=1, kernel=25, factor=1.0,
                              dropout=0.1)
BASE_TRAIN = TrainConfig(learning_rate=3e-4)
BASE_DATA = dict(exog_mode="future", transform="none", exog_entry="value",
                 exog_columns=tuple(FEATURES), phase_periods=PHASE_PERIODS)


def _config(model=None, data=None, train=None):
    return dict(model=replace(BASE_MODEL, **(model or {})), data={**BASE_DATA, **(data or {})},
                train=replace(BASE_TRAIN, **(train or {})))


CONFIGS = {
    "base": _config(),
    # Required ablation: what the optional file does, and which part of it.
    "exog: past only": _config(data=dict(exog_mode="past")),
    "exog: none": _config(data=dict(exog_mode="none")),
    # Where the covariates enter the network.
    "entry: marks": _config(data=dict(exog_entry="mark")),
    # Target transform.
    "target: log1p": _config(data=dict(transform="log1p")),
    # Periodic phase features recovered from the periodogram.
    "phase: daily only": _config(data=dict(phase_periods=(24.0,))),
    "phase: none": _config(data=dict(phase_periods=())),
    # Model size (the leaderboard charges for parameters).
    "d_model 16": _config(model=dict(d_model=16, d_ff=32)),
    "d_model 64": _config(model=dict(d_model=64, d_ff=128)),
    # Decomposition width, input length, number of aggregated delays, learning rate.
    "kernel 13": _config(model=dict(kernel=13)),
    "kernel 49": _config(model=dict(kernel=49)),
    "seq_len 96": _config(model=dict(seq_len=96)),
    "seq_len 336": _config(model=dict(seq_len=336)),
    "factor 3": _config(model=dict(factor=3.0)),
    "lr 1e-3": _config(train=dict(learning_rate=1e-3)),
    # Last-value anchoring (added after the per-horizon check in results/horizon_rmse_val.json
    # showed the reference-style model ignoring the most recent level).
    "anchor": _config(model=dict(anchor=True)),
    "anchor, d_model 16": _config(model=dict(anchor=True, d_model=16, d_ff=32)),
    "anchor, exog: past only": _config(model=dict(anchor=True), data=dict(exog_mode="past")),
    "anchor, exog: none": _config(model=dict(anchor=True), data=dict(exog_mode="none")),
    # Horizon trend initialised at the last value (persistence default) instead of the mean.
    "trend init last": _config(model=dict(trend_init="last")),
    "trend init last, d_model 16": _config(model=dict(trend_init="last", d_model=16, d_ff=32)),
    "trend init last, exog: past only": _config(model=dict(trend_init="last"),
                                                data=dict(exog_mode="past")),
    "trend init last, exog: none": _config(model=dict(trend_init="last"),
                                           data=dict(exog_mode="none")),
    # Learned decoder trend initialisation: persistence early, window mean late.
    "trend init decay": _config(model=dict(trend_init="decay")),
    "trend init learned": _config(model=dict(trend_init="learned")),
    "trend init decay, d_model 16": _config(model=dict(trend_init="decay", d_model=16, d_ff=32)),
    "trend init learned, d_model 16": _config(model=dict(trend_init="learned", d_model=16,
                                                         d_ff=32)),
}
# Which covariates carry signal: drop one group at a time from the base configuration.
for group, columns in FEATURE_GROUPS.items():
    kept = tuple(f for f in FEATURES if f not in columns)
    CONFIGS[f"drop {group}"] = _config(data=dict(exog_columns=kept))

SUITES = {
    "exog": ["base", "exog: past only", "exog: none"],
    "features": ["base"] + [f"drop {g}" for g in FEATURE_GROUPS],
    "design": ["base", "entry: marks", "target: log1p", "phase: daily only", "phase: none",
               "d_model 16", "d_model 64", "kernel 13", "kernel 49", "seq_len 96",
               "seq_len 336", "factor 3", "lr 1e-3"],
    "anchor": ["base", "anchor", "anchor, d_model 16", "anchor, exog: past only",
               "anchor, exog: none"],
    "trend init": ["base", "trend init last", "trend init last, d_model 16",
                   "trend init last, exog: past only", "trend init last, exog: none"],
    "trend blend": ["base", "d_model 16", "trend init decay", "trend init learned",
                    "trend init decay, d_model 16", "trend init learned, d_model 16"],
}
SEEDS = (0, 1, 2)
