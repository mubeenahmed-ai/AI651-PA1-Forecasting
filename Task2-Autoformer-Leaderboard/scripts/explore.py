"""Exploratory analysis behind the preprocessing choices (figures in results/figures/).

    python scripts/explore.py

Recovers the sampling periodicity from the data (no calendar is given), shows the heavy
tail that motivates log1p, and measures the relationship between each external variable and
the target on the training split only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.data import EXPERIMENT_SPLIT, FEATURES, load_raw  # noqa: E402

FIGURES = ROOT / "results" / "figures"


def autocorrelation(x, lags):
    x = x - x.mean()
    return np.array([x[:-lag] @ x[lag:] / (x @ x) for lag in lags])


def periodogram(x):
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    frequency = np.fft.rfftfreq(len(x))
    return 1 / frequency[1:], power[1:]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    y, external = load_raw()
    train_end = EXPERIMENT_SPLIT.train_end
    y_train = y[:train_end]
    rows = []

    # 1. Periodicity: periodogram of log target and of every continuous covariate.
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), layout="constrained")
    period, power = periodogram(np.log1p(y_train))
    axes[0].loglog(period, power, lw=.6, color="#007c91")
    axes[0].set(xlabel="period (steps)", ylabel="power", title="Periodogram, log1p(target)")
    for column, colour in (("feature_A", "#e07a5f"), ("feature_B", "#3d5a80"),
                           ("feature_C", "#43aa8b")):
        p, s = periodogram(external[column].to_numpy(float)[:train_end])
        axes[1].loglog(p, s / s.max(), lw=.6, color=colour, label=column)
        top = p[np.argsort(s)[::-1][:3]]
        rows.append({"series": column, "top periods": ", ".join(f"{v:.1f}" for v in top)})
    top = period[np.argsort(power)[::-1][:5]]
    rows.insert(0, {"series": "log1p(target)", "top periods": ", ".join(f"{v:.1f}" for v in top)})
    for axis in axes[:2]:
        for mark in (24, 168, 8766):
            axis.axvline(mark, color="#555", ls=":", lw=.8)
    axes[1].set(xlabel="period (steps)", ylabel="normalised power",
                title="Periodogram, continuous covariates A-C")
    axes[1].legend(fontsize=7)
    lags = np.arange(1, 24 * 15)
    axes[2].plot(lags, autocorrelation(y_train, lags), color="#007c91", label="target")
    axes[2].plot(lags, autocorrelation(np.log1p(y_train), lags), color="#d1495b",
                 label="log1p(target)")
    for mark in (24, 168):
        axes[2].axvline(mark, color="#555", ls=":", lw=.8)
    axes[2].axhline(0, color="#999", lw=.6)
    axes[2].set(xlabel="lag (steps)", ylabel="autocorrelation",
                title="ACF decays within ~2 days; weak 24-step bumps")
    axes[2].legend(fontsize=7)
    fig.savefig(FIGURES / "periodogram.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "periods.csv", index=False)

    # 2. Heavy tail: raw vs log1p distribution, and one 168-step block of history.
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), layout="constrained")
    axes[0].hist(y_train, bins=120, color="#007c91")
    axes[0].set(xlabel="target", ylabel="count", title=f"Target (skew {pd.Series(y_train).skew():.2f})")
    axes[1].hist(np.log1p(y_train), bins=120, color="#d1495b")
    axes[1].set(xlabel="log1p(target)", title=f"log1p target (skew {pd.Series(np.log1p(y_train)).skew():.2f})")
    axes[2].plot(np.arange(N := 24 * 28), y[-N:], color="#007c91", lw=.9)
    axes[2].set(xlabel="last 4 weeks of released history (steps)", ylabel="target",
                title="Order-of-magnitude swings inside one horizon")
    fig.savefig(FIGURES / "target_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. Covariate relevance on the training split: same-time and lagged correlation.
    log_y = pd.Series(np.log1p(y_train))
    relevance = []
    for column in FEATURES:
        x = external[column].iloc[:train_end].reset_index(drop=True).astype(float)
        relevance.append({
            "feature": column,
            "corr same step": log_y.corr(x),
            "corr with y 24 steps later": log_y.shift(-24).corr(x),
            "corr of 24-step change": log_y.diff(24).corr(x.diff(24)),
        })
    pd.DataFrame(relevance).round(3).to_csv(ROOT / "results" / "covariate_relevance.csv",
                                            index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(relevance).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
