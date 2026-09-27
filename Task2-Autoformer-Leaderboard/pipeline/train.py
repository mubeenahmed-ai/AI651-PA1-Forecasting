"""Training, early stopping and evaluation for one Autoformer run."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from autoformer import Autoformer, AutoformerConfig, count_parameters

from .data import HORIZON, N_HISTORY, Prepared, WindowSampler, origins
from .metrics import block_metrics

EVAL_STRIDE = 24   # one validation/test block starts every 24 steps


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 10
    patience: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-3
    lr_decay: float = 0.5        # multiply the learning rate by this after every epoch
    weight_decay: float = 1e-4
    grad_clip: float = 1.0


@dataclass
class RunResult:
    model: Autoformer
    parameters: int
    epochs_run: int              # E: every epoch trained, including the patience epochs
    best_epoch: int
    seconds: float
    history: list = field(default_factory=list)


def bind(model_config: AutoformerConfig, prepared: Prepared) -> AutoformerConfig:
    """Fill in the input widths that depend on the covariate configuration."""
    return replace(model_config, c_in=prepared.value_channels, mark_dim=prepared.mark_dim)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: Autoformer, sampler: WindowSampler, starts: np.ndarray,
            batch_size: int = 512) -> np.ndarray:
    """Forecasts in physical units, shape [len(starts), pred_len]."""
    model.eval()
    outputs = []
    for i in range(0, len(starts), batch_size):
        x_enc, mark_enc, mark_dec, dec_exog, _ = sampler.batch(starts[i:i + batch_size])
        outputs.append(model(x_enc, mark_enc, mark_dec, dec_exog)[..., 0].cpu().numpy())
    return sampler.p.transform.inverse(np.concatenate(outputs))


def truth(prepared: Prepared, starts: np.ndarray) -> np.ndarray:
    return prepared.raw[starts[:, None] + np.arange(HORIZON)[None, :]]


def evaluate(model, sampler, starts) -> dict:
    return block_metrics(predict(model, sampler, starts), truth(sampler.p, starts))


def fit(prepared: Prepared, model_config: AutoformerConfig, train_config: TrainConfig,
        seed: int, device=None, verbose: bool = True) -> RunResult:
    """Train on the split's training windows; keep the epoch with the best validation RMSE."""
    device = device or default_device()
    seed_everything(seed)
    split, c = prepared.split, model_config
    sampler = WindowSampler(prepared, c.seq_len, c.label_len, c.pred_len, device)
    train_starts = origins(0, split.train_end, c.seq_len, c.pred_len)
    val_starts = origins(split.train_end, split.val_end, c.seq_len, c.pred_len, EVAL_STRIDE)

    model = Autoformer(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                  weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, train_config.lr_decay)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    started = time.perf_counter()
    epoch = 0
    for epoch in range(1, train_config.max_epochs + 1):
        model.train()
        order = torch.as_tensor(train_starts)[torch.randperm(len(train_starts),
                                                             generator=generator)]
        losses = []
        for index in order.split(train_config.batch_size):
            x_enc, mark_enc, mark_dec, dec_exog, target = sampler.batch(index)
            loss = (model(x_enc, mark_enc, mark_dec, dec_exog)[..., 0] - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        validation = evaluate(model, sampler, val_starts)
        history.append({"epoch": epoch, "train loss": float(np.mean(losses)),
                        "val RMSE": validation["RMSE"], "val MAE": validation["MAE"]})
        if verbose:
            print(f"  epoch {epoch:2d}  train loss {np.mean(losses):.4f}  "
                  f"val RMSE {validation['RMSE']:.2f}  MAE {validation['MAE']:.2f}", flush=True)
        if validation["RMSE"] < best:
            best, best_epoch, stale = validation["RMSE"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= train_config.patience:
                break
    model.load_state_dict(best_state)
    return RunResult(model, count_parameters(model), epoch, best_epoch,
                     time.perf_counter() - started, history)


def evaluate_splits(result: RunResult, prepared: Prepared) -> dict:
    """Validation, test and last-block metrics for a model trained on ``prepared.split``."""
    c = result.model.config
    sampler = WindowSampler(prepared, c.seq_len, c.label_len, c.pred_len,
                            next(result.model.parameters()).device)
    split = prepared.split
    regions = {
        "val": origins(split.train_end, split.val_end, c.seq_len, c.pred_len, EVAL_STRIDE),
        "test": origins(split.val_end, N_HISTORY, c.seq_len, c.pred_len, EVAL_STRIDE),
        # The single most recent released block: one 168-step score, as on the leaderboard.
        "last block": np.array([N_HISTORY - HORIZON]),
    }
    return {name: evaluate(result.model, sampler, starts)
            for name, starts in regions.items() if len(starts)}


def describe(model_config: AutoformerConfig, train_config: TrainConfig) -> dict:
    return {**asdict(model_config), **{f"train_{k}": v for k, v in asdict(train_config).items()}}
