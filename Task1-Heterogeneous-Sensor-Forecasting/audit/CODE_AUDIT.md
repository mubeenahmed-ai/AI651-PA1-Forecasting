# Code audit: Task 1 notebook and supplied harness

**Scope.** `Assignment1.ipynb` (supplied cells) and `harness/design_{data,controls,models,diagnostics,reporting}.py`, as shipped in `DL4STG-PA1.zip`. The harness was **not modified**. The assignment asks for it to be kept as supplied, and every result in the report comes from the unmodified code.

**Method.** I read every module line by line and checked it against the handout (equations 1–8 and the appendix). Each finding below was then reproduced by `audit/verify_audit.py`, which runs on the full-preset world and never reads the fitted forecasters' test errors.

**Severity scale.**
- *High*: can change a reported number or conclusion.
- *Medium*: affects how an output must be read, or robustness.
- *Low*: hygiene or maintainability.

## Summary

| # | Area | Finding | Severity | Affects results in this run? |
|---|------|---------|----------|------------------------------|
| A1 | Protocol | Neural checkpoints are selected on the same validation split used to compare and choose models | Medium | Slightly optimistic neural validation scores; test (Output 4.3) is unaffected |
| A2 | Protocol | The decomposition width is chosen with the synthetic ground-truth period (oracle), and the choice sits on the edge of its grid | Medium | Width 41 = largest candidate; recovery 0.981–0.999 separates widths weakly |
| A3 | Diagnostics | Output 2.1 plots the single window with the **largest** raw period-estimation error | Medium | Chosen window: 4.98-sample error vs. median 0.53. The plot is a worst case |
| A4 | Robustness | `_fit` raises `TypeError` if no validation score is finite (`best_state` stays `None`) | Medium | No (training was finite) |
| A5 | Protocol | `select_for_test` can be called again after the test table has been read | Low | No (choices written once, before Output 4.3) |
| A6 | Reporting | `publish()` saves *every* open Matplotlib figure under the current output number | Low | No (every cell closes its figures) |
| A7 | Accounting | Period-routed ridge's parameter count omits the shared fallback matrix used for sparse bands | Low | No (all 13 bands have ≥ 52 training windows, so no fallback) |
| A8 | Consistency | Output 2.2 recomputes recovery on the preset stride (4), but the width was selected on stride 8 | Low | No (both tables pick width 41) |
| A9 | Environment | `jinja2` (needed by `DataFrame.to_latex`) is not pinned; it only arrives through `jupyterlab` | Low | No |
| A10 | Security | Checkpoints are read with `torch.load(..., weights_only=False)` (pickle) | Low | No (only self-written checkpoints) |
| A11 | Docs | Notebook header points to `main.pdf`; the handout ships as `DL4STG-PA1.pdf` | Low | Fixed in the notebook text |

**Verified correct**, and relevant to how the results can be trusted:
- **Leakage safety.** The anchoring scale uses only Stations 1–3 training readings (`Study.__init__`). Neural training windows come only from Stations 1–3 training targets (`Study.tensors`). Ridge λ is chosen on a chronological 80/20 holdout inside training (`_select_ridge_lambda`). Validation origins start at 9600, so every validation and test *target* lies in its region; contexts may reach back into earlier data, which is legitimate history.
- **No straddling windows.** Eligible windows never straddle a changeover (`make_world`, re-checked by `check_world`).
- **Fair 2×2 comparison.** The Part 1/2 classes (`RawAttentionForecaster`, `DecomposedAttentionForecaster`) are **bit-identical** to the unified Part 4 network with the matching switches: same parameters after the same seed, and 0.0 maximum output difference (verify_audit F1). The comparison really does change only the two switches.
- **Supplied delay mixer matches the handout.** `DelayMixer` implements the appendix exactly: band {8,…,48}, K = ⌈2 ln 96⌉ = 10, softplus temperature initialised to 1.0.
- **Supplied delay scores match the handout.** `delay_scores` matches the direct circular correlation of Eq. (4) (`check_delay_scores`).

## Findings in detail

### A1 — Validation is used both for checkpoint selection and for model comparison (Medium)
`design_controls._fit` evaluates established-station **validation** MSE every `steps // 4` steps and restores the best of those checkpoints. The same validation split then feeds Outputs 1.1, 2.3, 4.1, 4.2 and the deployment summary used to choose models. The neural models' validation numbers are therefore a minimum over four looks, while the ridge baselines never see validation (λ is chosen inside training).

*Impact.* The bias is small: with a one-cycle schedule the last checkpoint is usually the best (see `results/design` learning records). It is still one-sided in favour of the neural models. Output 4.3's **test** numbers are unaffected, which is why the report cites them for the final claims.

*Remedy (not applied).* Early-stop on a slice of the training period, or state the bias when comparing on validation, as the report does.

### A2 — Oracle width selection at the edge of the grid (Medium)
`Study._select_kernel` scores widths {17, 25, 33, 41} by how often the detrended period estimate falls within one sample of the **true** generator period. The notebook labels this an oracle diagnostic, and that is legitimate in a synthetic study. Two consequences for interpretation:
1. The winner, 41, is the **largest** candidate, so the optimum may lie outside the grid.
2. Recovery moves only from 0.982 to 0.999. The criterion is close to saturation and measures period recovery, not forecast error.

The report states that the width is the best of the four candidates *under this oracle criterion*, not a forecast-optimal choice. The same selected width also drives period-routed ridge's routing, so that baseline indirectly benefits from the oracle.

### A3 — Output 2.1 shows a worst case (Medium)
`figures.filter_decomposition` selects the established validation window where the *raw* period estimate is furthest from the truth (`argmax(gap)`). That window's raw error is 4.98 samples; the median over established validation windows is 0.53. The figure illustrates the decomposition well, but it must be described as the hardest window, not a typical one.

### A4 — Crash when validation never produces a finite score (Medium)
In `_fit`, `best = inf` and `state = None`. `score < best` is `False` for NaN, so if every validation evaluation is NaN, `model.load_state_dict(None)` raises `TypeError: Expected state_dict to be dict-like`. Only the training loss is checked for finiteness. This is reproduced with a model that is finite in train mode and NaN in eval mode (verify_audit F6).

### A5 — Test selection is not locked (Low)
`Study.select_for_test` can be called again after `selected_test_table()` has returned test errors, so the "choose before looking" rule is honour-based. In this submission both choices were written from the validation-only summary before Output 4.3 was executed (see `run_notebook.py --until-deployment`).

### A6 — `publish()` exports stale figures (Low)
`design_reporting.publish` iterates over `plt.get_fignums()` and saves every open figure as `<output>-<i>.pdf`. Any figure left open by an exploratory cell would be exported under the next output number. Reproduced with an output that has no figure of its own (verify_audit F8).

### A7 — Routed-ridge parameter count (Low)
Bands with fewer than 24 training windows get no matrix; at inference they fall back to the shared matrix. `_parameters("routed")` counts only `len(routed) × 96 × 48` and would omit that fallback. In this world, all 13 bands have 52–265 windows, so the count (59,904) is exact.

### A8 — Output 2.2 vs. the selection table (Low)
Selection uses origins at stride 8. `recurrence_recovery` recomputes the table on the preset's training stride (4 for `full`). With a different seed the two could disagree about the best width. For seed 0 both select 41.

### A9–A11 — Environment, security, docs (Low)
- `DataFrame.to_latex` requires `jinja2`, which is not listed in `requirements.txt`. It is installed only as a dependency of `jupyterlab`; the repository root `requirements.txt` pins it explicitly.
- `torch.load(weights_only=False)` unpickles arbitrary objects. That is acceptable for checkpoints the harness wrote itself, but a shared `checkpoints/` folder should not be trusted.
- The notebook's first cell referred to `main.pdf`; it now names `DL4STG-PA1.pdf`.

## Reproduce

```bash
cd Task1-Heterogeneous-Sensor-Forecasting
python audit/verify_audit.py
```
