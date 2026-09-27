"""Validation RMSE at every horizon step: reference vs learned-decay trend initialisation.

    python scripts/horizon_curves.py

Writes results/horizon_curves.json and results/figures/horizon_curves.pdf (seed 0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.data import EXPERIMENT_SPLIT as S, HORIZON, WindowSampler, origins, prepare  # noqa: E402
from pipeline.experiments import CONFIGS  # noqa: E402
from pipeline.train import EVAL_STRIDE, bind, fit, predict, truth  # noqa: E402

MODELS = {"Autoformer, mean init (paper)": "base",
          "Autoformer, learned decay init": "trend init decay",
          "Autoformer, no covariates": "exog: none"}


def main():
    curves = {}
    for label, name in MODELS.items():
        spec = CONFIGS[name]
        prepared = prepare(S, **spec["data"])
        config = bind(spec["model"], prepared)
        result = fit(prepared, config, spec["train"], 0, verbose=False)
        starts = origins(S.train_end, S.val_end, config.seq_len, HORIZON, EVAL_STRIDE)
        sampler = WindowSampler(prepared, config.seq_len, config.label_len, HORIZON,
                                next(result.model.parameters()).device)
        error = predict(result.model, sampler, starts) - truth(prepared, starts)
        curves[label] = np.sqrt((error ** 2).mean(0)).tolist()
    starts = origins(S.train_end, S.val_end, 168, HORIZON, EVAL_STRIDE)
    target = truth(prepared, starts)
    raw = prepared.raw
    curves["persistence (last value)"] = np.sqrt(((raw[starts - 1][:, None] - target) ** 2).mean(0)).tolist()
    curves["train mean"] = np.sqrt(((raw[:S.train_end].mean() - target) ** 2).mean(0)).tolist()
    (ROOT / "results" / "horizon_curves.json").write_text(json.dumps(curves))

    fig, axis = plt.subplots(figsize=(8, 3.6), layout="constrained")
    styles = {"persistence (last value)": dict(color="#999", ls="--"),
              "train mean": dict(color="#555", ls=":"),
              "Autoformer, no covariates": dict(color="#e09f3e"),
              "Autoformer, mean init (paper)": dict(color="#3d5a80"),
              "Autoformer, learned decay init": dict(color="#d1495b", lw=2)}
    steps = np.arange(1, HORIZON + 1)
    for label, curve in curves.items():
        axis.plot(steps, curve, label=label, **styles[label])
    axis.set(xlabel="horizon step", ylabel="validation RMSE (pooled)", xlim=(1, HORIZON),
             title="Error by horizon step (seed 0)")
    axis.legend(fontsize=7, ncol=2)
    fig.savefig(ROOT / "results" / "figures" / "horizon_curves.pdf", bbox_inches="tight")
    for label, curve in curves.items():
        print(f"{label:34s}", [round(curve[h - 1], 1) for h in (1, 3, 6, 12, 24, 48, 168)])


if __name__ == "__main__":
    main()
