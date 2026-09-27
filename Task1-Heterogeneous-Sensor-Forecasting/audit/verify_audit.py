"""Reproduce the evidence behind each finding in audit/CODE_AUDIT.md.

    python audit/verify_audit.py            (from the Task 1 folder; uses the full preset world)

Nothing here trains a full model or reads test errors of the fitted forecasters.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "harness"))
os.environ.setdefault("PA1_CHECKPOINTS", tempfile.mkdtemp())

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import design_controls as design  # noqa: E402
import design_data as data  # noqa: E402
import design_models as models  # noqa: E402
import design_reporting as report  # noqa: E402


def notebook_namespace():
    """Execute only the class/function definition cells of the completed notebook."""
    cells = json.loads((HERE / "Assignment1.ipynb").read_text())["cells"]
    namespace = {"__name__": "notebook"}
    exec("import math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n"
         "import pandas as pd\nimport numpy as np\nimport design_controls as design", namespace)
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(tree, "Assignment1.ipynb", "exec"), namespace)
    return namespace


def heading(text):
    print(f"\n=== {text} ===")


def main():
    ns = notebook_namespace()
    models.register_components(decomposition_type=ns["SeriesDecomposition"],
                               delay_scores=ns["delay_scores"], aggregate=ns["aggregate_delays"],
                               raw_forecaster_type=ns["RawAttentionForecaster"],
                               decomposed_attention_forecaster_type=ns["DecomposedAttentionForecaster"],
                               forecaster_type=ns["AutoformerInspiredForecaster"])

    heading("F1  Part 1/2 classes vs the unified Part 4 network (same seed, same settings)")
    probe = torch.randn(4, 96, 1)
    for name, representation, mixing in (("Raw Attention", "raw", "attention"),
                                         ("Attention + decomposition", "decomposed", "attention")):
        kwargs = dict(context=96, horizon=48, d=64, heads=4, layers=2, dropout=0.1)
        if representation == "decomposed":
            kwargs["kernel"] = 25
        torch.manual_seed(5000)
        separate = models.build(name, **kwargs).eval()
        torch.manual_seed(5000)
        unified = ns["AutoformerInspiredForecaster"](
            **kwargs, mixing=mixing, representation=representation,
            **models.components_for(dict(representation=representation, mixing=mixing))).eval()
        same_params = all(torch.equal(a, b) for a, b in zip(separate.state_dict().values(),
                                                              unified.state_dict().values()))
        with torch.no_grad():
            gap = (separate(probe) - unified(probe)).abs().max().item()
        print(f"{name:26s} identical initial parameters: {same_params}; max output gap {gap:.2e}")

    study = design.Study("full", 0)

    heading("F2  Width selection (stride 8) vs the Output 2.2 table (preset train stride)")
    print("selection table used by the harness:")
    print(study.kernel_selection.to_string(index=False))
    part = study.region("train")
    keep = study.mask("established", "train")
    rows = []
    for width in study.kernel_selection["width"]:
        estimate = data.estimate_period(part["context"], detrend_kernel=int(width))
        rows.append((int(width), float((np.abs(estimate - part["primary_period"])[keep] <= 1).mean())))
    print(f"Output 2.2 recomputation on stride {study.preset.stride_train}: "
          + ", ".join(f"width {w}: {r:.4f}" for w, r in rows))
    best_22 = max(rows, key=lambda item: item[1])[0]
    print(f"selected width {study.decomposition_kernel}; best width in Output 2.2 table {best_22}")

    heading("F3  Period-routed ridge: sparse routing bands fall back to the shared matrix")
    routed = study._weights("routed")
    centres = data.routing_centres(study.world.product_period_bands)
    train_routes = np.bincount(part["route"][keep], minlength=len(centres))
    print("training windows per routing band:",
          dict(zip(np.round(centres, 2).tolist(), train_routes.tolist())))
    missing = sorted(set(range(len(centres))) - set(routed))
    print(f"bands without their own matrix: {[round(float(centres[i]), 2) for i in missing]}")
    for region in ("validation", "test"):
        routes = study.region(region)["route"]
        fallback = np.isin(routes, missing)
        print(f"{region}: {fallback.mean():.1%} of windows use the shared-ridge fallback")
    print(f"reported parameter count {study._parameters('routed')} "
          f"(= {len(routed)} x 96 x 48); shared fallback matrix adds 4608 when used")

    heading("F4  Ridge 'fit seconds' exclude the period estimator that routing depends on")
    started = time.perf_counter()
    data.estimate_period(part["context"], detrend_kernel=study.decomposition_kernel)
    print(f"period estimation on the training windows: {time.perf_counter() - started:.3f}s; "
          f"recorded routed-ridge fit time: {study._linear_runs['routed']:.3f}s")

    heading("F5  Output 2.1 plots the window with the largest raw period-estimation error")
    val = study.region("validation")
    raw_gap = np.abs(data.estimate_period(val["context"], detrend_kernel=0) - val["primary_period"])
    eligible = study.mask("established", "validation")
    print(f"chosen window raw error {raw_gap[eligible].max():.2f} samples; median over "
          f"established validation windows {np.median(raw_gap[eligible]):.2f}")

    heading("F6  _fit crashes if every validation score is NaN (best state never assigned)")

    class NaNAtEval(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(96, 48))

        def forward(self, x):
            out = x[..., 0] @ self.w
            return out if self.training else out * float("nan")

    x = np.zeros((8, 96, 1), np.float32)
    y = np.zeros((8, 48), np.float32)
    try:
        design._fit(NaNAtEval(), x, y, x, y, design.Preset(1, 1, 2, 8, 1, 4), 0, "probe")
        print("no error")
    except Exception as error:  # noqa: BLE001
        print(f"raised {type(error).__name__}: {error}")

    heading("F7  Test selection can be changed after test errors have been read")
    study.select_for_test({"established": "Shared ridge", "held-out": "Shared ridge"})
    first = study.selected_test_table()
    study.select_for_test({"established": "Period-routed ridge", "held-out": "Period-routed ridge"})
    second = study.selected_test_table()
    print("second selection accepted after reading test table:",
          list(first.model) != list(second.model))

    heading("F8  publish() exports every open figure, including unrelated stale ones")
    with tempfile.TemporaryDirectory() as directory:
        plt.figure()                         # e.g. a figure left open by an exploratory cell
        report.publish("probe", None, directory=directory)
        print("files written for an output with no figure of its own:",
              sorted(p.name for p in Path(directory).iterdir()))

    heading("F9  Environment")
    import importlib.util
    print("jinja2 importable (needed by DataFrame.to_latex in publish):",
          importlib.util.find_spec("jinja2") is not None,
          "-- not pinned in requirements.txt; arrives only via jupyterlab")
    print("torch.load(weights_only=False) is used for checkpoints (design_controls.train)")


if __name__ == "__main__":
    main()
