"""Robustness check (not a notebook Output): refit the four neural models with other model
seeds on the same data and compare validation MSE with the submitted seed-0 run.

    PA1_CHECKPOINTS=/tmp/somewhere python audit/seed_check.py 1 2
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PA1_CHECKPOINTS", tempfile.mkdtemp())
from verify_audit import notebook_namespace  # noqa: E402  (also puts harness/ on sys.path)

import pandas as pd  # noqa: E402
import design_controls as design  # noqa: E402
import design_models as models  # noqa: E402

NEURAL = ["Raw Attention", "Attention + decomposition", "Raw delay mixer", "Autoformer-inspired"]

ns = notebook_namespace()
models.register_components(decomposition_type=ns["SeriesDecomposition"],
                           delay_scores=ns["delay_scores"], aggregate=ns["aggregate_delays"],
                           raw_forecaster_type=ns["RawAttentionForecaster"],
                           decomposed_attention_forecaster_type=ns["DecomposedAttentionForecaster"],
                           forecaster_type=ns["AutoformerInspiredForecaster"])
rows = []
for model_seed in map(int, sys.argv[1:] or ["1"]):
    study = design.Study("full", 0, model_seed=model_seed)
    for name in NEURAL:
        study.train(name)
    table = study.model_table("validation")
    table = table[table.model.isin(NEURAL + ["Period-routed ridge"])]
    table.insert(0, "model seed", model_seed)
    rows.append(table)
result = pd.concat(rows)
result.to_csv(HERE / "seed_check.csv", index=False)
print(result.pivot_table(index="model", columns=["model seed", "population"], values="MSE").round(3))
