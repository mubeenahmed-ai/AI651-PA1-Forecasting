"""Aggregate results/runs.jsonl across seeds into tables and figures for the report.

    python scripts/summarize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.experiments import SUITES  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def load_runs() -> pd.DataFrame:
    rows = []
    for line in (RESULTS / "runs.jsonl").read_text().splitlines():
        run = json.loads(line)
        row = {k: run[k] for k in ("config", "seed", "parameters", "epochs run", "best epoch",
                                   "seconds")}
        for region, metrics in run["metrics"].items():
            for metric in ("RMSE", "MAE", "sMAPE"):
                row[f"{region} {metric}"] = metrics[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def seed_summary(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in runs.columns if c.split(" ")[0] in {"val", "test", "last"}]
    grouped = runs.groupby("config", sort=False)
    table = pd.DataFrame({"seeds": grouped.seed.nunique(),
                          "P": grouped.parameters.first(),
                          "E mean": grouped["epochs run"].mean()})
    for column in metrics:
        table[f"{column} mean"] = grouped[column].mean()
        table[f"{column} std"] = grouped[column].std(ddof=1)
    return table.reset_index()


def paired_differences(runs: pd.DataFrame, reference="base", metric="val RMSE"):
    """Per-seed difference to the reference configuration (same seed = same initialisation)."""
    pivot = runs.pivot(index="seed", columns="config", values=metric)
    rows = []
    for config in pivot.columns.drop(reference):
        delta = (pivot[config] - pivot[reference]).dropna()
        rows.append({"config": config, f"{metric} minus {reference}": delta.mean(),
                     "std over seeds": delta.std(ddof=1), "seeds": len(delta),
                     "worse on every seed": bool((delta > 0).all()),
                     "better on every seed": bool((delta < 0).all())})
    return pd.DataFrame(rows)


def plot_suite(runs: pd.DataFrame, names, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained", sharey=False)
    for axis, metric in zip(axes, ("val RMSE", "test RMSE")):
        for i, name in enumerate(names):
            values = runs.loc[runs.config == name, metric]
            axis.bar(i, values.mean(), color="#9db4c0" if name != "base" else "#d1495b",
                     yerr=values.std(ddof=1), capsize=4)
            axis.scatter(np.full(len(values), i), values, color="#202124", s=12, zorder=3)
        axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
        axis.set(ylabel=f"{metric} (mean block RMSE)", title=f"{title}: {metric.split()[0]}")
        low = min(runs.loc[runs.config.isin(names), metric]) * 0.95
        axis.set_ylim(bottom=low)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    runs.to_csv(RESULTS / "runs.csv", index=False)
    summary = seed_summary(runs)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    differences = paired_differences(runs)
    differences.to_csv(RESULTS / "paired_differences_val.csv", index=False)
    paired_differences(runs, metric="test RMSE").to_csv(
        RESULTS / "paired_differences_test.csv", index=False)

    baselines = pd.DataFrame(json.loads((RESULTS / "baselines.json").read_text()))
    baselines.to_csv(RESULTS / "baselines.csv", index=False)

    compact = summary[["config", "seeds", "P", "E mean",
                       "val RMSE mean", "val RMSE std", "val MAE mean", "val sMAPE mean",
                       "test RMSE mean", "test RMSE std", "test MAE mean", "test sMAPE mean",
                       "last block RMSE mean", "last block RMSE std"]]
    with (RESULTS / "summary.md").open("w") as handle:
        handle.write("# Validation study (mean and std over seeds)\n\n")
        handle.write(compact.round(2).to_markdown(index=False))
        handle.write("\n\n# Paired per-seed differences to `base` (validation RMSE)\n\n")
        handle.write(differences.round(2).to_markdown(index=False))
        handle.write("\n\n# Reference baselines (not submitted)\n\n")
        handle.write(baselines[["model", "region", "RMSE", "RMSE block std", "MAE",
                                "sMAPE"]].round(2).to_markdown(index=False))
        handle.write("\n")
    for suite, names in SUITES.items():
        present = [n for n in names if n in set(runs.config)]
        if present:
            plot_suite(runs, present, suite, FIGURES / f"ablation_{suite}.pdf")
    print(compact.round(2).to_string(index=False))
    print()
    print(differences.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
