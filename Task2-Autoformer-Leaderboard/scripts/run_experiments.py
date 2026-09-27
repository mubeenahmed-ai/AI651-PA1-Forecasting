"""Run the validation study: every configuration in the chosen suites, over several seeds.

    python scripts/run_experiments.py                 # all suites, seeds 0 1 2
    python scripts/run_experiments.py --suite exog --seeds 0 1 2 3 4

Results are appended to results/runs.jsonl; runs already present are skipped, so the
script can be interrupted and resumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.baselines import baseline_table  # noqa: E402
from pipeline.data import EXPERIMENT_SPLIT, prepare  # noqa: E402
from pipeline.experiments import CONFIGS, SEEDS, SUITES  # noqa: E402
from pipeline.train import bind, describe, evaluate_splits, fit  # noqa: E402

RESULTS = ROOT / "results"


def completed(path: Path) -> set:
    if not path.exists():
        return set()
    return {(row["config"], row["seed"]) for row in map(json.loads, path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", nargs="*", default=list(SUITES))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / "runs.jsonl"

    raw = prepare(EXPERIMENT_SPLIT, exog_mode="none").raw
    (RESULTS / "baselines.json").write_text(json.dumps(baseline_table(raw, EXPERIMENT_SPLIT),
                                                       indent=1))

    names = list(dict.fromkeys(name for suite in args.suite for name in SUITES[suite]))
    done = completed(log)
    for name in names:
        spec = CONFIGS[name]
        prepared = prepare(EXPERIMENT_SPLIT, **spec["data"])
        model_config = bind(spec["model"], prepared)
        for seed in args.seeds:
            if (name, seed) in done:
                continue
            print(f"[{name}] seed {seed}", flush=True)
            result = fit(prepared, model_config, spec["train"], seed)
            metrics = evaluate_splits(result, prepared)
            row = {"config": name, "seed": seed, "parameters": result.parameters,
                   "epochs run": result.epochs_run, "best epoch": result.best_epoch,
                   "seconds": round(result.seconds, 1), "metrics": metrics,
                   "history": result.history, "settings": {
                       **describe(model_config, spec["train"]),
                       **{k: list(v) if isinstance(v, tuple) else v
                          for k, v in spec["data"].items()}}}
            with log.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"  -> val RMSE {metrics['val']['RMSE']:.2f}  test RMSE "
                  f"{metrics['test']['RMSE']:.2f}  P={result.parameters}  "
                  f"E={result.epochs_run}", flush=True)


if __name__ == "__main__":
    main()
