"""Fit the selected configuration once and write the 168 leaderboard forecasts.

    python scripts/make_submission.py --config base --seed 0

The production fit trains on all released history except the last 13 weeks, which are used
only for early stopping, then forecasts time_idx 43657..43824 from the end of the history.
P and E in submission/submission.json are read from this exact run.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.data import (DATA_DIR, HORIZON, N_HISTORY, N_TOTAL, PRODUCTION_SPLIT,  # noqa: E402
                           WindowSampler, prepare)
from pipeline.experiments import CONFIGS  # noqa: E402
from pipeline.train import bind, evaluate_splits, fit, predict  # noqa: E402

OUT = ROOT / "submission"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base", choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = CONFIGS[args.config]
    prepared = prepare(PRODUCTION_SPLIT, **spec["data"])
    model_config = bind(spec["model"], prepared)
    print(f"Production fit: config={args.config!r}, seed={args.seed}, "
          f"train [0, {PRODUCTION_SPLIT.train_end}), early stopping on "
          f"[{PRODUCTION_SPLIT.train_end}, {N_HISTORY})", flush=True)
    result = fit(prepared, model_config, spec["train"], args.seed)
    validation = evaluate_splits(result, prepared)

    device = next(result.model.parameters()).device
    sampler = WindowSampler(prepared, model_config.seq_len, model_config.label_len,
                            model_config.pred_len, device)
    forecast = predict(result.model, sampler, np.array([N_HISTORY]))[0]

    time_idx = np.arange(N_HISTORY + 1, N_TOTAL + 1)
    template = pd.read_csv(DATA_DIR / "student_test.csv")
    assert forecast.shape == (HORIZON,) and np.isfinite(forecast).all() and (forecast >= 0).all()
    assert np.array_equal(template.time_idx.to_numpy(), time_idx)

    OUT.mkdir(exist_ok=True)
    values = [f"{value:.4f}" for value in forecast]            # time_idx 43657 first
    (OUT / "predictions.txt").write_text(", ".join(values) + "\n")
    template.assign(value=np.round(forecast, 4)).to_csv(OUT / "student_test_filled.csv",
                                                        index=False)
    torch.save({"state": result.model.state_dict(), "config": asdict(model_config)},
               OUT / "model.pt")
    record = {
        "config": args.config, "seed": args.seed,
        "P (trainable parameters)": result.parameters,
        "E (epochs trained)": result.epochs_run,
        "best epoch": result.best_epoch,
        "early-stopping split": {"train_end": PRODUCTION_SPLIT.train_end,
                                 "val_end": PRODUCTION_SPLIT.val_end},
        "production validation metrics": validation,
        "model": asdict(model_config),
        "training": asdict(spec["train"]),
        "data": {k: list(v) if isinstance(v, tuple) else v for k, v in spec["data"].items()},
        "forecast time_idx": [int(time_idx[0]), int(time_idx[-1])],
    }
    (OUT / "submission.json").write_text(json.dumps(record, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(11, 3.4), layout="constrained")
    shown = 24 * 21
    axis.plot(np.arange(N_HISTORY - shown + 1, N_HISTORY + 1), prepared.raw[-shown:],
              color="#007c91", lw=.9, label="released history")
    axis.plot(time_idx, forecast, color="#d1495b", lw=1.3, label="submitted forecast")
    axis.axvline(N_HISTORY + .5, color="#555", ls=":", lw=.8)
    axis.set(xlabel="time_idx", ylabel="target", title=f"Submitted 168-step forecast "
             f"({args.config}, seed {args.seed}; P={result.parameters}, E={result.epochs_run})")
    axis.legend(fontsize=8)
    fig.savefig(OUT / "forecast.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"P = {result.parameters}, E = {result.epochs_run} "
          f"(best epoch {result.best_epoch}); wrote {OUT / 'predictions.txt'}")


if __name__ == "__main__":
    main()
