"""RNN models for Part 1."""

from __future__ import annotations

import torch
from torch import nn


class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


class StockGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


class TurningPointBiRNN(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rnn(x)
        h_n = out[1][0] if self.rnn_type == "lstm" else out[1]
        h_n = h_n.view(self.rnn.num_layers, 2, x.shape[0], self.rnn.hidden_size)
        final = h_n[-1]
        hidden = torch.cat([final[0], final[1]], dim=1)
        return self.head(hidden).squeeze(1)
