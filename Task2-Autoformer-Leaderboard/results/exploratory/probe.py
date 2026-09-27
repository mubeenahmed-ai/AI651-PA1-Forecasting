import sys, json; sys.path.insert(0, ".")
from dataclasses import replace
from pipeline.data import EXPERIMENT_SPLIT, prepare
from pipeline.experiments import BASE_MODEL, BASE_TRAIN
from pipeline.train import fit, evaluate_splits
variants = {
  "lr1e-4": (dict(), dict(learning_rate=1e-4), dict()),
  "lr1e-4 drop0.3": (dict(dropout=0.3), dict(learning_rate=1e-4), dict()),
  "lr1e-4 daily-phase-only": (dict(), dict(learning_rate=1e-4), dict(phase_periods=(24.0,))),
  "lr1e-4 no-log": (dict(), dict(learning_rate=1e-4), dict(transform="none")),
  "lr3e-4 wd1e-2 drop0.2": (dict(dropout=0.2), dict(learning_rate=3e-4, weight_decay=1e-2), dict()),
}
for name, (m, t, d) in variants.items():
    for seed in (0, 1):
        p = prepare(EXPERIMENT_SPLIT, **{"exog_mode": "future", "transform": "log1p", **d})
        mc = replace(BASE_MODEL, mark_dim=p.marks.shape[1], **m)
        r = fit(p, mc, replace(BASE_TRAIN, **t), seed, verbose=False)
        e = evaluate_splits(r, p)
        print(f"{name:28s} s{seed} val {e['val']['RMSE']:.2f} test {e['test']['RMSE']:.2f} "
              f"best_ep {r.best_epoch} E {r.epochs_run} curve {[round(h['val RMSE'],1) for h in r.history]}", flush=True)
