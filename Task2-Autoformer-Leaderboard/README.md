# Task 2 — Leaderboard Challenge (Autoformer, 168-step forecast)

## Submission

| | |
|---|---|
| Forecasts | `submission/predictions.txt`: 168 comma-separated values; time_idx 43657 first, 43824 last |
| Model | Autoformer, config `trend init decay, d_model 16`, seed 0 (`submission/model.pt`) |
| **P** (trainable parameters) | **6,626** |
| **E** (epochs trained) | **6** (a single production run; no refit) |
| Record | `submission/submission.json` (configuration, P, E, early-stopping metrics); `submission/forecast.pdf` (plot) |

Both P and E are read from the production run by `scripts/make_submission.py` and can be reproduced from the repository.

## Layout

| Path | Contents |
|---|---|
| `autoformer/layers.py` | Series decomposition, Auto-Correlation (FFT delay scoring, top-k time-delay aggregation), encoder/decoder layers, embeddings |
| `autoformer/model.py` | Encoder–decoder Autoformer with progressive trend accumulation |
| `pipeline/data.py` | Loading, feature engineering, chronological splits, leakage-safe scaling, window sampling, covariate regimes |
| `pipeline/train.py` | Training loop, early stopping, evaluation |
| `pipeline/metrics.py` | MAE, RMSE, sMAPE (per 168-step block and pooled) |
| `pipeline/baselines.py` | Naive baselines and a linear covariate ridge (comparison only, not submitted) |
| `pipeline/experiments.py` | Every configuration in the validation study |
| `scripts/` | `explore.py`, `run_experiments.py`, `summarize.py`, `make_submission.py` |
| `tests/` | Component and leakage tests (`python -m pytest -q tests`) |
| `results/` | `runs.jsonl` (120 runs), `summary.{csv,md}`, paired-difference tables, `baselines.csv`, `horizon_rmse_val.json`, `horizon_curves.json`, `figures/` |
| `results/exploratory/` | First-round runs and probes that motivated the revised base configuration |

## Autoformer implementation

The implementation follows Wu et al. (2021), §§3.1–3.2, and the authors' reference code (github.com/thuml/Autoformer, MIT licence). It was rewritten component by component. Both required mechanisms are present in every layer:

- **Series decomposition.** A moving average with edge replication (kernel 25) splits the input before the decoder, and runs again after every Auto-Correlation and feed-forward sublayer. The encoder keeps only the seasonal part. The decoder accumulates the extracted trends through a projection into the trend stream.
- **Auto-Correlation.** Delay scores R(τ) are computed with the FFT in O(L log L). The top k = ⌊c·ln L⌋ delays are kept and their scores softmaxed, and values are aggregated by circular time-delay roll. This replaces point-wise dot-product attention entirely.

**Deviations from the reference code, each deliberate:**
- Delays are selected per sample in both training and inference; the reference code uses batch-shared delays in training.
- The roll direction is `v[t − τ]`, which matches the correlation definition and Task 1.
- The external covariates enter as **value channels**: in the encoder with the target, and in the decoder with their known future values. The reference code routes known inputs through additive "mark" embeddings; the study below compares both.

## Data findings (`scripts/explore.py`, training split only)

- **Periodicity.** Recovered without any calendar: a 24-step cycle in the target and in every continuous covariate, and an ~8,766-step annual cycle, the dominant peak of features A–C. These two periods become sin/cos phase features built from the integer index, so they are known over the horizon.
- **Heavy tails.** Skew is 1.8 and the maximum is 994 against a median of 73. The ACF falls to 0.16 at lag 48 and ~0 at lag 168, so the history alone says little about a week ahead.
- **Covariates.** The four binary indicators (G–J) are one-hot. The strongest links to the target are dew point (A: 24-step-change correlation 0.67), pressure (C: −0.42), and cumulative wind (D: −0.30).

## Validation protocol

- **Split.** Chronological 80 / 10 / 10 of the released history: train, validation (early stopping and every design decision), and test (reported, never used for a choice).
- **Scoring.** Validation and test are each scored on ~175 overlapping 168-step blocks, starting every 24 steps. **RMSE** is the mean of per-block RMSEs, which is the expected leaderboard score for one block. MAE and sMAPE are also tracked.
- **Leakage.** All scaling statistics come from the training split only. Covariate regimes: `future` (true horizon values, allowed by §2.3), `past` (held at the last observed value over the horizon), `none`.
- **Seeds.** The external-data ablation uses 5 seeds and every other configuration 3. Every comparison is paired by seed.

## Results (mean ± std over seeds; lower is better)

**External-data ablation (required), 5 seeds:**

| Covariates | Val RMSE | Test RMSE | Val MAE | Val sMAPE |
|---|---:|---:|---:|---:|
| **future (known over horizon)** | **70.4 ± 1.6** | **61.6 ± 1.5** | 56.9 | 64.2 |
| past only (held at origin value) | 91.7 ± 2.3 | 85.8 ± 2.1 | 76.7 | 72.8 |
| none | 89.1 ± 1.6 | 88.4 ± 2.3 | 74.4 | 71.0 |

Using the covariates' **future** values is worth about 19 RMSE on validation and 27 on test, worse without them on every seed. **Past-only covariates give nothing:** they are no better than no covariates. Both history-only Autoformers are worse than a constant training mean (85.9 val / 81.6 test).

That last point is a property of *this model*, not of the data. Persistence beats the mean for the first ~12 steps (see the horizon diagnosis below), so the recent history *does* carry short-term information; the Autoformer does not exploit it. At the 168-step block level, climatology is hard to beat without the future covariates.

**Which covariates matter** (dropping one group from base; paired val Δ over 3 seeds):
- **Dew point A:** +6.0, worse on every seed.
- **Wind D + G–J:** +4.0.
- **Rain/snow E + F:** +2.1.
- **Temperature B (+0.6) and pressure C (−0.2):** within noise.

**Design choices** (paired val Δ vs base):
- **Consistently worse, on every seed:** a log1p target, +8.6. MSE on the raw scale matches the RMSE criterion, while a log-space loss under-weights the peaks that dominate RMSE. log1p does lower sMAPE (53.9 vs 64.2), a place where the metrics disagree. A shorter input (seq_len 96) is also worse, +2.8.
- **Indistinguishable from base within seed spread:** covariates as marks (−1.2 val but +2.2 test, and far less stable in the exploratory round, with one seed at 94), seq_len 336, kernel 13/49, factor 3, lr 1e-3, and phase features (daily only +0.7, none +1.4).
- **Model size:** d_model 16 (P = 6,625) gives +0.9 ± 2.0 val and 62.6 test. d_model 64 (P = 87,937) gives +0.8 val and 60.9 test.

**Size choice:** `d_model 16`. It is statistically indistinguishable from the 23,489-parameter base and 3.5× smaller, and the leaderboard penalises parameters. The final submission adds the learned decoder trend initialisation described below.

**Reference models (not submitted):**

| Model | Val RMSE | Test RMSE |
|---|---:|---:|
| train mean | 85.9 | 81.6 |
| persistence | 105.6 | 104.5 |
| seasonal naive (24) | 109.7 | 105.4 |
| covariate ridge (78 parameters) | **62.9** | **55.3** |

A linear model on the same future covariates **beats every Autoformer configuration**. Neither Autoformer lever closes that gap:
- **Regularisation:** dropout, weight decay and learning rate were probed in `results/exploratory/`; none helped.
- **Size:** d_model 16/32/64 score within noise of one another.

The Autoformer passes the ridge's training loss within two epochs while its validation error rises. With a heavy-tailed target and MSE, the flexible model fits individual pollution episodes that a low-capacity linear model cannot. This bears on reflection question Q2.9 and on Task 1's central question. The ridge is reported for honesty only; the rules require the submitted model to be an Autoformer.

## Horizon diagnosis: what the decoder can and cannot express

Theory says a series with lag-1 autocorrelation 0.97 should be forecast almost perfectly one step ahead. Pooled validation RMSE by horizon step:

| Model | step 1 | step 6 | step 24 | step 168 | block RMSE |
|---|---:|---:|---:|---:|---:|
| persistence (last value) | **19.4** | **57.8** | 107.6 | 136.0 | — |
| train mean | 105.7 | 92.4 | 107.3 | 107.2 | 85.9 |
| Autoformer base (seed 0; `results/horizon_rmse_val.json`) | 87.8 | 80.0 | 86.8 | 104.5 | 70.4 (5 seeds) |
| + last-value anchoring (5 seeds) | 85.6 | 76.1 | 86.7 | 99.6 | 72.0 |
| + horizon trend initialised at the last value (3 seeds) | 48.5 | 65.9 | 99.1 | 129.9 | 89.3 |

The reference decoder initialises the horizon's trend with the **window mean** (paper Eq. 2). Its 25-wide moving averages then blur the label/horizon boundary. So the model cannot start from the current level: even on the *training* windows its step-1 RMSE is 75, against 24 for persistence.

- **Anchoring** the input on its last value, as the Task 1 harness does, changes nothing. The trend init is still the window mean in absolute terms.
- **Initialising the horizon trend at the last value** fixes the first steps (48 vs 88) but loses the reversion to the mean. Long-horizon error then rises to near persistence, and block RMSE worsens by ~19.

Neither fixed initialisation expresses "persistence early, climatology late", which is what theory prescribes for a series whose autocorrelation decays within about a day. That points to the remedy.

**Remedy (adopted): a learned decay initialisation.** The horizon trend starts at

    start_h = last + (mean − last) · (1 − exp(−h/τ)),   τ learned (1 parameter, initialised at 12 steps)

so it begins at the current level and relaxes to the window mean. Only the decoder's starting point changes; series decomposition and Auto-Correlation are untouched.

Paired by seed, 5 seeds (`results/trend_blend.log`):

| Config | P | Val RMSE | Test RMSE | step 1 | step 6 | step 24 | step 168 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (mean init) | 23,489 | 70.36 ± 1.60 | 61.59 ± 1.47 | ~88 | ~80 | ~87 | ~104 |
| + decay init | 23,490 | **69.25 ± 1.47** | **60.33 ± 0.81** | 54.3 | 58.1 | 80.4 | 99.0 |
| + learned per-step weights (168 parameters) | 23,657 | 69.23 ± 1.44 | 59.88 ± 1.12 | 54.1 | 57.3 | 81.2 | 98.8 |
| d_model 16 (mean init) | 6,625 | 71.57 ± 1.07 | 62.14 ± 1.37 | 85.9 | 75.2 | 87.3 | 103.0 |
| **d_model 16 + decay init (submitted)** | **6,626** | **70.70 ± 1.10** | 62.88 ± 1.79 | **47.7** | **56.7** | 84.3 | 103.5 |

- **Validation.** Decay init improves validation RMSE on **every one of the 5 seeds** at both sizes: −1.10 ± 0.31 (d_model 32) and −0.86 ± 0.41 (d_model 16). It cuts the step-1 error roughly in half without hurting long horizons.
- **Test.** At d_model 32 it also improves test (−1.25, better on 4/5 seeds). At d_model 16 test is mixed (+0.75 ± 1.37, better on 2/5).
- **Why it was adopted.** Selection is made on validation, where the gain is consistent across all seeds for one extra parameter.
- **Per-step weights.** The free per-step variant is no better than the single time constant, so the 1-parameter form is kept.
- **Learned τ.** In the production fit τ = 12.2 steps, close to the target's autocorrelation half-life. It is the timescale theory predicts.

The covariate ridge, which sees the last 24 values directly, still leads (62.9 val). The remaining gap is the capacity/overfitting issue discussed above.

## Production fit

The model trains on time_idx 1–41,472 and early-stops on the final 13 weeks, 41,473–43,656. It then forecasts 43,657–43,824 from the end of the history, using the covariates' known future values.
- The best epoch was 3, and training stopped after 3 epochs of patience. E = 6.
- The early-stopping RMSE on those 13 weeks is 77.2. That is a harder autumn–winter window than the experiment split: the block-RMSE std is 27, and one 168-step score moves by ±25–45 across blocks.
- A leaderboard score should therefore be read as confirmation, not as a tuning signal.

## Reproduce

```bash
python -m pytest -q tests
python scripts/explore.py
python scripts/run_experiments.py --suite exog --seeds 0 1 2 3 4
python scripts/run_experiments.py            # every other suite, seeds 0 1 2 (skips finished runs)
python scripts/run_experiments.py --suite anchor --seeds 0 1 2 3 4
python scripts/run_experiments.py --suite "trend init"
python scripts/summarize.py
python scripts/run_experiments.py --suite "trend blend" --seeds 0 1 2 3 4
python scripts/horizon_curves.py
python scripts/make_submission.py --config "trend init decay, d_model 16" --seed 0
```

The runs used an RTX 2070 (PyTorch 2.5.1, CUDA 12.1); one run takes about a minute. Runs are seeded, with cuDNN in deterministic mode, and repeat bit-for-bit on the same machine.
