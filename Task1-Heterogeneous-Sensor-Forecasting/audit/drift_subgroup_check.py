"""Supplementary check for Response 4(a): which switch helps most where the slow part is large?

    python audit/drift_subgroup_check.py

Uses the cached full-preset models (checkpoints/) and the validation split only. Windows are
split by the true slow-component range over the context (a diagnostic the harness records as
``slow_change``; no model sees it). Writes audit/drift_subgroup_check.csv.
"""
import contextlib
import io
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PA1_CHECKPOINTS", str(HERE.parent / "checkpoints"))
from verify_audit import notebook_namespace  # noqa: E402  (also puts harness/ on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import design_controls as design  # noqa: E402
import design_models as models  # noqa: E402

NAMES = ["Raw Attention", "Attention + decomposition", "Raw delay mixer", "Autoformer-inspired"]

ns = notebook_namespace()
models.register_components(decomposition_type=ns["SeriesDecomposition"], delay_scores=ns["delay_scores"],
                           aggregate=ns["aggregate_delays"], raw_forecaster_type=ns["RawAttentionForecaster"],
                           decomposed_attention_forecaster_type=ns["DecomposedAttentionForecaster"],
                           forecaster_type=ns["AutoformerInspiredForecaster"])
study = design.Study("full", 0)
with contextlib.redirect_stdout(io.StringIO()):
    for name in NAMES:
        study.train(name)
part, predictions = study.region("validation"), study.predictions("validation")
established = study.mask("established", "validation")
slow = part["slow_change"]
cut = np.quantile(slow[established], 0.75)
groups = {"top-quartile slow range": established & (slow >= cut),
          "other windows": established & (slow < cut)}
rows = []
for name in NAMES:
    error = ((predictions[name] - part["target"]) ** 2).mean(-1)
    for group, keep in groups.items():
        rows.append({"model": name, "group": group, "windows": int(keep.sum()),
                     "validation MSE": float(error[keep].mean())})
table = pd.DataFrame(rows)
reference = table[table.model == "Raw Attention"].set_index("group")["validation MSE"]
table["gain vs Raw Attention"] = table["validation MSE"] - table.group.map(reference)
table.to_csv(HERE / "drift_subgroup_check.csv", index=False)
print(f"slow-range cut (75th percentile, established validation): {cut:.2f} m/s^2")
print(table.round(3).to_string(index=False))
