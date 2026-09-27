"""Encoder-decoder Autoformer (Wu et al., 2021, Fig. 1)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .layers import (DataEmbedding, DecoderLayer, EncoderLayer, SeasonalLayerNorm,
                     SeriesDecomposition)


@dataclass(frozen=True)
class AutoformerConfig:
    seq_len: int = 168        # encoder input length
    label_len: int = 48       # known history handed to the decoder as its start token
    pred_len: int = 168       # forecast horizon (fixed by the assignment)
    c_in: int = 1             # value channels: the target first, then any covariate channels
    c_out: int = 1            # forecast channels (the target)
    mark_dim: int = 4         # covariate / phase features per time step
    d_model: int = 32
    d_ff: int = 64
    heads: int = 4
    e_layers: int = 1
    d_layers: int = 1
    kernel: int = 25          # moving-average width used by every decomposition block
    factor: float = 1.0       # c in k = c * ln L
    dropout: float = 0.1
    # Last-value anchoring (as in the Task 1 harness): model the target relative to its last
    # observed value and add that value back to the forecast. The anchor itself is passed as a
    # constant "level" channel so that level-dependent mean reversion stays learnable.
    anchor: bool = False
    # Horizon part of the decoder's trend initialisation:
    #   "mean"    window mean (paper Eq. 2)
    #   "last"    last observed value (persistence default)
    #   "decay"   last + (mean - last) * (1 - exp(-h / tau)), one learned time constant tau
    #   "learned" last + (mean - last) * sigmoid(a_h), one learned weight per horizon step
    trend_init: str = "mean"
    decay_init: float = 12.0  # initial tau (steps) for "decay"/"learned"; ~ACF half-life


class Autoformer(nn.Module):
    """Decomposition encoder plus a decoder that accumulates trend layer by layer.

    Decoder initialisation (paper Eq. 2): its seasonal input is the seasonal part of the last
    ``label_len`` encoder steps followed by zeros, and its trend input is the trend of those
    steps followed by the mean of the encoder window. The forecast is the final seasonal
    projection plus the accumulated trend.
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.config = c = config
        channels = c.c_in + int(c.anchor)
        self.decomposition = SeriesDecomposition(c.kernel)
        self.encoder_embedding = DataEmbedding(channels, c.mark_dim, c.d_model, c.dropout)
        self.decoder_embedding = DataEmbedding(channels, c.mark_dim, c.d_model, c.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(c.d_model, c.d_ff, c.heads, c.kernel,
                                                   c.factor, c.dropout)
                                      for _ in range(c.e_layers)])
        self.encoder_norm = SeasonalLayerNorm(c.d_model)
        self.decoder = nn.ModuleList([DecoderLayer(c.d_model, c.d_ff, c.heads, c.kernel,
                                                   c.factor, c.dropout, c.c_out)
                                      for _ in range(c.d_layers)])
        self.decoder_norm = SeasonalLayerNorm(c.d_model)
        self.projection = nn.Linear(c.d_model, c.c_out)
        if c.trend_init not in {"mean", "last", "decay", "learned"}:
            raise ValueError(c.trend_init)
        steps = torch.arange(1, c.pred_len + 1, dtype=torch.float32)
        if c.trend_init == "decay":
            # softplus(raw) = tau at initialisation
            self.raw_tau = nn.Parameter(torch.tensor(c.decay_init).expm1().log())
        elif c.trend_init == "learned":
            weight = (1 - torch.exp(-steps / c.decay_init)).clamp(1e-4, 1 - 1e-4)
            self.blend_logit = nn.Parameter(torch.logit(weight))
        self.register_buffer("steps", steps, persistent=False)

    def horizon_weight(self):
        """w_h in [0, 1]: 0 = last observed value, 1 = window mean, for h = 1..pred_len."""
        c = self.config
        if c.trend_init == "mean":
            return torch.ones_like(self.steps)
        if c.trend_init == "last":
            return torch.zeros_like(self.steps)
        if c.trend_init == "decay":
            return 1 - torch.exp(-self.steps / nn.functional.softplus(self.raw_tau))
        return torch.sigmoid(self.blend_logit)

    def forward(self, x_enc, mark_enc, mark_dec, dec_exog=None):
        """x_enc [B,seq_len,c_in]; mark_enc [B,seq_len,m]; mark_dec [B,label_len+pred_len,m];
        dec_exog [B,label_len+pred_len,c_in-c_out] known covariate values for the decoder.

        Returns the forecast [B,pred_len,c_out].
        """
        c = self.config
        if c.anchor:
            last = x_enc[:, -1:, :c.c_out]                                  # [B,1,c_out]
            x_enc = torch.cat([x_enc[..., :c.c_out] - last, x_enc[..., c.c_out:],
                               last.expand(-1, x_enc.shape[1], -1)], dim=-1)
            level = last.expand(-1, c.label_len + c.pred_len, -1)
            dec_exog = level if dec_exog is None else torch.cat([dec_exog, level], dim=-1)
        target = x_enc[..., :c.c_out]
        seasonal, trend = self.decomposition(target)
        final, window_mean = target[:, -1:], target.mean(dim=1, keepdim=True)   # [B,1,c_out]
        if c.trend_init == "mean":          # exact reference path (paper Eq. 2)
            start = window_mean.expand(-1, c.pred_len, -1)
        else:
            start = final + (window_mean - final) * self.horizon_weight()[None, :, None]
        zeros = torch.zeros_like(start)
        trend_init = torch.cat([trend[:, -c.label_len:], start], dim=1)
        seasonal_init = torch.cat([seasonal[:, -c.label_len:], zeros], dim=1)
        if dec_exog is not None:
            # Covariate channels carry their known values across label and horizon; only the
            # target's future is unknown (zeros), as in the reference decoder initialisation.
            seasonal_init = torch.cat([seasonal_init, dec_exog], dim=-1)

        encoded = self.encoder_embedding(x_enc, mark_enc)
        for layer in self.encoder:
            encoded = layer(encoded)
        encoded = self.encoder_norm(encoded)

        decoded = self.decoder_embedding(seasonal_init, mark_dec)
        for layer in self.decoder:
            decoded, residual_trend = layer(decoded, encoded)
            trend_init = trend_init + residual_trend
        seasonal_out = self.projection(self.decoder_norm(decoded))
        forecast = (trend_init + seasonal_out)[:, -c.pred_len:]
        return forecast + last if c.anchor else forecast


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count P, exactly as the leaderboard defines it."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
