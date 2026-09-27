"""Autoformer building blocks.

Structure follows Wu et al. (2021), "Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting", Sections 3.1-3.2, and the authors'
reference implementation (https://github.com/thuml/Autoformer, MIT licence), rewritten here
so that every component is explicit. Differences from the reference code are marked
"Deviation" in the docstrings.

Shapes use B = batch, L = query length, S = key length, d = model width, h = heads.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SeriesDecomposition(nn.Module):
    """Centered moving-average split x = seasonal + trend (paper Eq. 1).

    The ends are padded by repeating the first/last value, so the trend has the same
    length as x and seasonal + trend reconstructs x exactly.
    """

    def __init__(self, kernel: int):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # x [B,L,C]
        radius = self.kernel // 2
        padded = torch.cat([x[:, :1].expand(-1, radius, -1), x,
                            x[:, -1:].expand(-1, radius, -1)], dim=1)
        trend = F.avg_pool1d(padded.transpose(1, 2), self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


class SeasonalLayerNorm(nn.Module):
    """LayerNorm over features, then remove the per-sequence mean over time.

    Autoformer's "my_Layernorm": the seasonal stream should carry no level, so the
    time-average left after normalisation is subtracted.
    """

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,L,d]
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


class AutoCorrelation(nn.Module):
    """Period-based dependency discovery and time-delay aggregation (paper Sec. 3.2).

    Scores every delay tau with the circular cross-correlation
        R(tau) = sum_t q[t] * k[(t - tau) mod L],
    computed for all delays at once with the FFT in O(L log L). The top-k delays
    (k = floor(c * ln L)) are kept, their scores go through a softmax, and the output is
        z[t] = sum_j w_j * v[(t - tau_j) mod L].

    Deviation: the reference code selects one set of delays shared by the whole batch during
    training and uses a per-sample set only at inference. Here delays are selected per
    sample in both modes, so training and inference run the same operator. The shift
    direction (t - tau) matches the correlation definition above and the Task 1 notebook.
    """

    def __init__(self, factor: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):  # [B,L,h,e], [B,S,h,e], [B,S,h,e] -> [B,L,h,e]
        batch, length, heads, width = queries.shape
        source = keys.shape[1]
        if length > source:  # decoder cross-correlation: align key length with query length
            pad = torch.zeros_like(queries[:, :length - source])
            keys, values = torch.cat([keys, pad], dim=1), torch.cat([values, pad], dim=1)
        else:
            keys, values = keys[:, :length], values[:, :length]

        q = queries.permute(0, 2, 3, 1)  # [B,h,e,L]
        k = keys.permute(0, 2, 3, 1)
        v = values.permute(0, 2, 3, 1)
        spectrum = torch.fft.rfft(q, dim=-1) * torch.conj(torch.fft.rfft(k, dim=-1))
        correlation = torch.fft.irfft(spectrum, n=length, dim=-1)  # [B,h,e,L], R(tau)

        top_k = max(1, int(self.factor * math.log(length)))
        scores = correlation.mean(dim=(1, 2))                   # [B,L], one score per delay
        weights, delays = scores.topk(top_k, dim=-1)            # [B,k]
        weights = self.dropout(torch.softmax(weights, dim=-1))  # [B,k]

        positions = torch.arange(length, device=v.device)
        mixed = torch.zeros_like(v)
        for j in range(top_k):
            index = (positions[None, :] - delays[:, j, None]) % length  # [B,L]
            shifted = v.gather(-1, index[:, None, None, :].expand_as(v))
            mixed = mixed + weights[:, j, None, None, None] * shifted
        return mixed.permute(0, 3, 1, 2)                        # [B,L,h,e]


class AutoCorrelationLayer(nn.Module):
    """Multi-head wrapper: project to q/k/v, auto-correlate, project back."""

    def __init__(self, d: int, heads: int, factor: float, dropout: float):
        super().__init__()
        if d % heads:
            raise ValueError("model width must be divisible by the number of heads")
        self.heads = heads
        self.correlation = AutoCorrelation(factor, dropout)
        self.query, self.key, self.value, self.out = (nn.Linear(d, d) for _ in range(4))

    def forward(self, queries, keys, values):  # [B,L,d], [B,S,d], [B,S,d] -> [B,L,d]
        batch, length, _ = queries.shape
        source = keys.shape[1]
        q = self.query(queries).view(batch, length, self.heads, -1)
        k = self.key(keys).view(batch, source, self.heads, -1)
        v = self.value(values).view(batch, source, self.heads, -1)
        return self.out(self.correlation(q, k, v).reshape(batch, length, -1))


class FeedForward(nn.Module):
    """Position-wise feed-forward network (kernel-1 convolutions in the reference code)."""

    def __init__(self, d: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Auto-Correlation -> decompose -> feed-forward -> decompose; only the seasonal part is kept."""

    def __init__(self, d, d_ff, heads, kernel, factor, dropout):
        super().__init__()
        self.correlation = AutoCorrelationLayer(d, heads, factor, dropout)
        self.feed_forward = FeedForward(d, d_ff, dropout)
        self.decomp1, self.decomp2 = SeriesDecomposition(kernel), SeriesDecomposition(kernel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B,L,d]
        x, _ = self.decomp1(x + self.dropout(self.correlation(x, x, x)))
        x, _ = self.decomp2(x + self.feed_forward(x))
        return x


class DecoderLayer(nn.Module):
    """Self and cross Auto-Correlation with progressive trend accumulation (paper Eq. 4)."""

    def __init__(self, d, d_ff, heads, kernel, factor, dropout, c_out):
        super().__init__()
        self.self_correlation = AutoCorrelationLayer(d, heads, factor, dropout)
        self.cross_correlation = AutoCorrelationLayer(d, heads, factor, dropout)
        self.feed_forward = FeedForward(d, d_ff, dropout)
        self.decomp1, self.decomp2, self.decomp3 = (SeriesDecomposition(kernel) for _ in range(3))
        self.dropout = nn.Dropout(dropout)
        # Projects the d-dimensional trend extracted inside this layer onto the output channels.
        self.trend_projection = nn.Conv1d(d, c_out, kernel_size=3, padding=1,
                                          padding_mode="circular", bias=False)

    def forward(self, x, cross):  # [B,L,d], [B,S,d] -> ([B,L,d], [B,L,c_out])
        x, trend1 = self.decomp1(x + self.dropout(self.self_correlation(x, x, x)))
        x, trend2 = self.decomp2(x + self.dropout(self.cross_correlation(x, cross, cross)))
        x, trend3 = self.decomp3(x + self.feed_forward(x))
        trend = self.trend_projection((trend1 + trend2 + trend3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class DataEmbedding(nn.Module):
    """Value embedding (circular conv over time) plus a linear embedding of the covariate marks.

    As in the reference "DataEmbedding_wo_pos": no absolute positional encoding, because the
    Auto-Correlation mechanism works with delays rather than positions. Here the "marks" are
    the periodic phase features and, depending on the configuration, the external variables.
    """

    def __init__(self, c_in: int, mark_dim: int, d: int, dropout: float):
        super().__init__()
        self.value = nn.Conv1d(c_in, d, kernel_size=3, padding=1, padding_mode="circular",
                               bias=False)
        self.mark = nn.Linear(mark_dim, d, bias=False) if mark_dim else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, marks):  # [B,L,c_in], [B,L,m] -> [B,L,d]
        embedded = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.mark is not None:
            embedded = embedded + self.mark(marks)
        return self.dropout(embedded)
