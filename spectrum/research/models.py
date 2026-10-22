"""Interchangeable causal encoders for the four-state opportunity experiment.

Inputs are [batch, time, feature]. ``forward`` uses only the final sequence
state; ``sequence_logits`` exposes every prefix state for causality audits.
No architecture normalizes across examples or across time.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _SequenceClassifier(nn.Module):
    receptive_field: int | None

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.input_dim or x.shape[1] < 1:
            raise ValueError(f"Expected [B,T,{self.input_dim}] with T >= 1, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequence_logits(x)[:, -1]


class _CausalConv(nn.Module):
    def __init__(self, incoming: int, outgoing: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(incoming, outgoing, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class MLPClassifier(_SequenceClassifier):
    """Pointwise baseline: earlier input samples cannot affect the prediction."""

    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.input_dim, self.receptive_field = input_dim, 1
        blocks = []
        for layer in range(layers):
            blocks.extend([nn.Linear(input_dim if layer == 0 else hidden, hidden),
                           nn.GELU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*blocks)
        self.output = nn.Linear(hidden, 4)

    def sequence_logits(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.output(self.encoder(x))


class CNNClassifier(_SequenceClassifier):
    """Undilated causal convolutions; field = 1 + layers * (kernel - 1)."""

    def __init__(self, input_dim: int, hidden: int, layers: int, kernel: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.receptive_field = 1 + layers * (kernel - 1)
        blocks = []
        for layer in range(layers):
            blocks.extend([_CausalConv(input_dim if layer == 0 else hidden, hidden, kernel),
                           nn.GELU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*blocks)
        self.output = nn.Linear(hidden, 4)

    def sequence_logits(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.output(self.encoder(x.transpose(1, 2)).transpose(1, 2))


class _TCNBlock(nn.Module):
    def __init__(self, hidden: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.path = nn.Sequential(
            _CausalConv(hidden, hidden, kernel, dilation), nn.GELU(), nn.Dropout(dropout),
            _CausalConv(hidden, hidden, kernel, dilation), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.path(x)


class TCNClassifier(_SequenceClassifier):
    """Two causal convolutions per block, dilation 1,2,4,...; residual paths."""

    def __init__(self, input_dim: int, hidden: int, layers: int, kernel: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** layers - 1)
        self.projection = nn.Linear(input_dim, hidden)
        self.encoder = nn.Sequential(*[
            _TCNBlock(hidden, kernel, 2 ** layer, dropout) for layer in range(layers)
        ])
        self.output = nn.Linear(hidden, 4)

    def sequence_logits(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        h = self.projection(x).transpose(1, 2)
        return self.output(self.encoder(h).transpose(1, 2))


class LSTMClassifier(_SequenceClassifier):
    """Unidirectional, zero-initialized per supplied window; no state carry.

    ``receptive_field=None`` means no fixed architectural cutoff, not infinite
    observed history: the actual information boundary is the supplied window.
    """

    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.input_dim, self.receptive_field = input_dim, None
        self.encoder = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                               bidirectional=False, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 4)

    def sequence_logits(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        h, _ = self.encoder(x)
        return self.output(self.dropout(h))


def build_model(input_dim: int, config: dict) -> nn.Module:
    """Build mlp/cnn/tcn/lstm; unknown architectures fail rather than substitute.

    Defaults: hidden=32, layers=2, kernel_size=3, dropout=0. No padding mask is
    inferred. The data adapter must provide valid windows and explicit missing
    feature indicators where needed.
    """
    architecture = config.get("architecture", "mlp")
    supported = {"mlp", "cnn", "tcn", "lstm"}
    if architecture not in supported:
        raise ValueError(f"Unsupported architecture {architecture!r}; choose {sorted(supported)}")
    values = {"input_dim": input_dim, "hidden": config.get("hidden", 32),
              "layers": config.get("layers", 2), "kernel_size": config.get("kernel_size", 3)}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    hidden, layers, kernel = values["hidden"], values["layers"], values["kernel_size"]
    dropout = float(config.get("dropout", 0.0))
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if architecture == "mlp":
        return MLPClassifier(input_dim, hidden, layers, dropout)
    if architecture == "cnn":
        return CNNClassifier(input_dim, hidden, layers, kernel, dropout)
    if architecture == "tcn":
        return TCNClassifier(input_dim, hidden, layers, kernel, dropout)
    return LSTMClassifier(input_dim, hidden, layers, dropout)
