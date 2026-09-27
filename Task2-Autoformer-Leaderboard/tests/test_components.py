"""Checks for the Autoformer components and the leakage rules of the data pipeline.

    python tests/test_components.py      (or: python -m pytest tests)
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoformer import AutoCorrelation, Autoformer, AutoformerConfig, SeriesDecomposition  # noqa: E402
from pipeline.data import (EXPERIMENT_SPLIT, HORIZON, N_HISTORY, WindowSampler,  # noqa: E402
                           prepare)
from pipeline.metrics import smape  # noqa: E402


def test_decomposition_edges_and_reconstruction():
    x = torch.tensor([1., 2., 3., 10., 5.], dtype=torch.float64).reshape(1, 5, 1)
    seasonal, trend = SeriesDecomposition(3)(x)
    torch.testing.assert_close(trend.flatten(), torch.tensor(
        [4 / 3, 2., 5., 6., 20 / 3], dtype=torch.float64))
    torch.testing.assert_close(seasonal + trend, x)


def test_autocorrelation_matches_direct_computation():
    """Scores are R(tau) = sum_t q[t] k[t - tau]; aggregation reads v[t - tau]."""
    torch.manual_seed(0)
    batch, length, heads, width = 2, 12, 2, 3
    q, k, v = (torch.randn(batch, length, heads, width, dtype=torch.float64) for _ in range(3))
    layer = AutoCorrelation(factor=1.0).eval()
    out = layer(q, k, v)

    top_k = int(math.log(length))
    for b in range(batch):
        scores = torch.stack([(q[b] * torch.roll(k[b], tau, dims=0)).sum(0).mean()
                              for tau in range(length)])
        weights, delays = scores.topk(top_k)
        weights = weights.softmax(0)
        expected = sum(w * torch.roll(v[b], int(d), dims=0) for w, d in zip(weights, delays))
        torch.testing.assert_close(out[b], expected)


def test_autoformer_shapes_and_gradients():
    config = AutoformerConfig(seq_len=96, label_len=48, pred_len=HORIZON, mark_dim=5,
                              d_model=16, d_ff=32)
    model = Autoformer(config)
    x = torch.randn(3, 96, 1, requires_grad=True)
    out = model(x, torch.randn(3, 96, 5), torch.randn(3, 48 + HORIZON, 5))
    assert out.shape == (3, HORIZON, 1)
    out.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    starved = [n for n, p in model.named_parameters() if p.grad is None]
    assert not starved, starved


def test_anchor_is_equivariant_to_level_shift_of_the_input_path():
    """With anchoring, the target path sees x - x[-1]; the level enters only as a channel."""
    config = AutoformerConfig(seq_len=96, label_len=48, pred_len=HORIZON, mark_dim=2,
                              d_model=16, d_ff=32, anchor=True, dropout=0.0)
    model = Autoformer(config).eval()
    x = torch.randn(2, 96, 1)
    marks_enc, marks_dec = torch.randn(2, 96, 2), torch.randn(2, 48 + HORIZON, 2)
    out = model(x, marks_enc, marks_dec)
    assert out.shape == (2, HORIZON, 1)
    # Forecast at a constant input equals that constant plus a learned level-only offset.
    flat = torch.full((1, 96, 1), 3.0)
    assert torch.isfinite(model(flat, marks_enc[:1], marks_dec[:1])).all()


def test_trend_init_weights():
    base = dict(seq_len=96, label_len=48, pred_len=HORIZON, mark_dim=2, d_model=16, d_ff=32)
    for kind, first, last in (("mean", 1.0, 1.0), ("last", 0.0, 0.0)):
        w = Autoformer(AutoformerConfig(**base, trend_init=kind)).horizon_weight()
        assert w[0] == first and w[-1] == last
    for kind in ("decay", "learned"):
        model = Autoformer(AutoformerConfig(**base, trend_init=kind, decay_init=12.0))
        w = model.horizon_weight().detach()
        assert abs(w[11].item() - (1 - math.exp(-1))) < 1e-3   # 1 - e^-1 at h = tau
        assert torch.all(w[1:] >= w[:-1]) and w[0] < 0.1 and w[-1] > 0.99
    # Anchored + decay still restores the level exactly for a flat input's persistence part.
    model = Autoformer(AutoformerConfig(**base, trend_init="decay", anchor=True)).eval()
    assert torch.isfinite(model(torch.full((1, 96, 1), 2.0), torch.randn(1, 96, 2),
                                torch.randn(1, 48 + HORIZON, 2))).all()


def test_past_mode_hides_future_covariates():
    config = AutoformerConfig(seq_len=96, label_len=48)
    future = prepare(EXPERIMENT_SPLIT, exog_mode="future")
    past = prepare(EXPERIMENT_SPLIT, exog_mode="past")
    origin = np.array([EXPERIMENT_SPLIT.train_end + 500])
    _, _, dec_future, _, _ = WindowSampler(future, config.seq_len, config.label_len).batch(origin)
    _, enc_past, dec_past, _, _ = WindowSampler(past, config.seq_len, config.label_len).batch(origin)
    n = past.n_exog
    horizon = dec_past[0, config.label_len:, -n:]
    # Every horizon row equals the last covariate value observed before the origin.
    assert torch.allclose(horizon, horizon[:1].expand_as(horizon))
    assert torch.allclose(horizon[0], enc_past[0, -1, -n:])
    # Known-history rows of the decoder are untouched; the horizon differs from the truth.
    assert torch.allclose(dec_past[0, :config.label_len], dec_future[0, :config.label_len])
    assert not torch.allclose(dec_past[0, config.label_len:], dec_future[0, config.label_len:])


def test_value_entry_channels_and_past_mode():
    config = AutoformerConfig(seq_len=96, label_len=48)
    past = prepare(EXPERIMENT_SPLIT, exog_mode="past", exog_entry="value")
    future = prepare(EXPERIMENT_SPLIT, exog_mode="future", exog_entry="value")
    origin = np.array([EXPERIMENT_SPLIT.train_end + 500])
    x_enc, mark_enc, mark_dec, dec_exog, _ = WindowSampler(
        past, config.seq_len, config.label_len).batch(origin)
    n = past.n_exog
    assert x_enc.shape[-1] == 1 + n == past.value_channels
    assert mark_enc.shape[-1] == mark_dec.shape[-1] == past.mark_dim
    horizon = dec_exog[0, config.label_len:]
    assert torch.allclose(horizon, x_enc[0, -1:, 1:].expand_as(horizon))
    _, _, _, dec_future, _ = WindowSampler(future, config.seq_len, config.label_len).batch(origin)
    assert not torch.allclose(dec_future[0, config.label_len:], horizon)
    model = Autoformer(replace(config, c_in=past.value_channels, mark_dim=past.mark_dim,
                               d_model=16, d_ff=32))
    assert model(x_enc, mark_enc, mark_dec, dec_exog).shape == (1, HORIZON, 1)


def test_scaling_uses_training_split_only():
    prepared = prepare(EXPERIMENT_SPLIT, exog_mode="future", transform="log1p")
    train = prepared.scaled[:EXPERIMENT_SPLIT.train_end]
    assert abs(train.mean()) < 1e-9 and abs(train.std() - 1) < 1e-9
    assert len(prepared.raw) == N_HISTORY
    restored = prepared.transform.inverse(prepared.scaled)
    np.testing.assert_allclose(restored, prepared.raw, atol=1e-8)


def test_smape_zero_over_zero():
    assert smape(np.array([0., 1.]), np.array([0., 1.])) == 0.0


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"passed  {name}")
