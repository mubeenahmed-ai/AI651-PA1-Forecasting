"""Execute Assignment1.ipynb headlessly with the chosen preset.

    python run_notebook.py --preset full               # executes and saves the notebook in place
    python run_notebook.py --preset full --until-deployment

``--until-deployment`` stops before the deployment-choice cell, so the choices can be written
from the validation evidence before Output 4.3 (the only cell that reads test errors) runs.
Fitted models are cached in checkpoints/, so the final complete run reloads them.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Assignment1.ipynb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="full", choices=("smoke", "quick", "full"))
    parser.add_argument("--until-deployment", action="store_true")
    args = parser.parse_args()
    os.environ["PA1_PRESET"] = args.preset
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    target = notebook
    if args.until_deployment:
        stop = next(i for i, cell in enumerate(notebook.cells)
                    if cell.cell_type == "code" and "select_for_test" in cell.source)
        target = nbformat.from_dict({**notebook, "cells": notebook.cells[:stop]})
    NotebookClient(target, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    output = NOTEBOOK if not args.until_deployment else HERE / "partial_run.ipynb"
    nbformat.write(target, output)
    print(f"executed with PA1_PRESET={args.preset}; saved {output.name}")


if __name__ == "__main__":
    main()
