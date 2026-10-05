# AI651 — Deep Learning for Space, Time and Graphs · Programming Assignment 1

Lahore University of Management Sciences · Fall 2026

| Folder | Contents |
|---|---|
| [`Task1-Heterogeneous-Sensor-Forecasting/`](Task1-Heterogeneous-Sensor-Forecasting/) | Completed and executed notebook (full preset), the supplied harness (unmodified), exported Outputs 1.1–4.3, and a code audit of the supplied harness |
| [`Task2-Autoformer-Leaderboard/`](Task2-Autoformer-Leaderboard/) | Autoformer implementation, data pipeline, multi-seed validation study with the external-data ablation, and the leaderboard submission (168 forecasts with the declared P and E) |
| [`report/`](report/) | Final report (`report.tex`, compiled `report.pdf`, `figures/`) covering both tasks |

Each task folder has its own README with results, design decisions and exact commands.

## Environment

Python 3.11+ is required by the handout. From this folder:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt        # add --index-url https://download.pytorch.org/whl/cpu for CPU-only torch
```

Both tasks use CUDA when it is available and fall back to the CPU.

## Reproduce everything

```bash
# Task 1 — executes the notebook with the full preset (writes results/design/*)
cd Task1-Heterogeneous-Sensor-Forecasting
python run_notebook.py --preset full
python audit/verify_audit.py
cd ..

# Task 2 — tests, exploration, validation study, summary tables, final forecast
cd Task2-Autoformer-Leaderboard
python -m pytest -q tests
python scripts/explore.py
python scripts/run_experiments.py --suite exog --seeds 0 1 2 3 4
python scripts/run_experiments.py
python scripts/summarize.py
python scripts/make_submission.py --config "trend init decay, d_model 16" --seed 0
```

## Academic integrity and AI use

Generative-AI assistance was used in coding, formatting of the report and understanding of the main concepts. 
