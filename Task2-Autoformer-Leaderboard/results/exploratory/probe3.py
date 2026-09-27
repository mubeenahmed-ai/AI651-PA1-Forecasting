import sys; sys.path.insert(0, ".")
from dataclasses import replace
from pipeline.data import EXPERIMENT_SPLIT, prepare
from pipeline.experiments import BASE_MODEL, BASE_TRAIN
from pipeline.train import bind, fit, evaluate_splits
variants = {
  "nolog value lr1e-3": (dict(), dict(), dict(exog_entry="value", transform="none")),
  "nolog mark lr1e-3": (dict(), dict(), dict(transform="none")),
  "nolog value lr3e-4": (dict(), dict(learning_rate=3e-4), dict(exog_entry="value", transform="none")),
  "nolog mark lr3e-4": (dict(), dict(learning_rate=3e-4), dict(transform="none")),
  "nolog value lr1e-3 d16": (dict(d_model=16, d_ff=32), dict(), dict(exog_entry="value", transform="none")),
}
_old = {
  "value lr1e-3": (dict(), dict(), dict(exog_entry="value")),
  "value lr1e-4": (dict(), dict(learning_rate=1e-4), dict(exog_entry="value")),
  "value lr1e-4 daily-only": (dict(), dict(learning_rate=1e-4), dict(exog_entry="value", phase_periods=(24.0,))),
  "value lr1e-4 no-log": (dict(), dict(learning_rate=1e-4), dict(exog_entry="value", transform="none")),
}
for name, (m, t, d) in variants.items():
    for seed in (0, 1, 2):
        p = prepare(EXPERIMENT_SPLIT, **{"exog_mode": "future", "transform": "log1p", **d})
        mc = replace(bind(BASE_MODEL, p), **m)
        r = fit(p, mc, replace(BASE_TRAIN, **t), seed, verbose=False)
        e = evaluate_splits(r, p)
        print(f"{name:26s} s{seed} P {r.parameters} val {e['val']['RMSE']:.2f} test {e['test']['RMSE']:.2f} "
              f"best_ep {r.best_epoch} E {r.epochs_run} curve {[round(h['val RMSE'],1) for h in r.history]}", flush=True)
