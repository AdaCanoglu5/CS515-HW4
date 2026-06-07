"""Mini-GBAF style learned feedback code."""

from __future__ import annotations

import torch
from torch import nn

from part2.channel import awgn, power_normalize

M = 8
K = 4
T_ROUNDS = 4
D_CHANNEL = K
SIGMA2 = 0.25
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
EMB_DIM = 16


def _transformer_encoder() -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=N_HEADS,
        dim_feedforward=D_FF,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=N_LAYERS)


class TXEncoder(nn.Module):
    def __init__(
        self,
        m: int = M,
        k: int = K,
        t_rounds: int = T_ROUNDS,
        no_feedback: bool = False,
    ) -> None:
        super().__init__()
        self.k = k
        self.t_rounds = t_rounds
        self.no_feedback = no_feedback
        raw_dim = EMB_DIM + t_rounds + t_rounds + t_rounds + t_rounds
        self.msg_embed = nn.Embedding(m, EMB_DIM)
        self.pre_mlp = nn.Sequential(nn.Linear(raw_dim, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.pos_embed = nn.Embedding(k, D_MODEL)
        self.transformer = _transformer_encoder()
        self.post_mlp = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1))

    def forward(
        self,
        m: torch.Tensor,
        x_history: torch.Tensor,
        y_history: torch.Tensor,
        round_idx: int,
    ) -> torch.Tensor:
        batch = m.shape[0]
        device = m.device
        if self.no_feedback:
            y_history = torch.zeros_like(y_history)
        history_mask = torch.zeros(batch, self.k, self.t_rounds, device=device)
        history_mask[:, :, :round_idx] = 1.0
        round_onehot = torch.zeros(batch, self.k, self.t_rounds, device=device)
        round_onehot[:, :, round_idx] = 1.0
        knowledge = torch.cat(
            [
                self.msg_embed(m),
                x_history,
                y_history,
                history_mask,
                round_onehot,
            ],
            dim=-1,
        )
        tokens = self.pre_mlp(knowledge)
        positions = torch.arange(self.k, device=device)
        tokens = tokens + self.pos_embed(positions).unsqueeze(0)
        encoded = self.transformer(tokens)
        x = self.post_mlp(encoded).squeeze(-1)
        return power_normalize(x)


class RXDecoder(nn.Module):
    def __init__(self, m: int = M, k: int = K, t_rounds: int = T_ROUNDS) -> None:
        super().__init__()
        self.k = k
        self.t_rounds = t_rounds
        self.pre_mlp = nn.Sequential(nn.Linear(t_rounds, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.pos_embed = nn.Embedding(k, D_MODEL)
        self.transformer = _transformer_encoder()
        self.post_mlp = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, m))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        tokens = y.transpose(1, 2)
        positions = torch.arange(self.k, device=y.device)
        tokens = self.pre_mlp(tokens) + self.pos_embed(positions).unsqueeze(0)
        decoded = self.transformer(tokens)
        return self.post_mlp(decoded)


class FeedbackCodeSystem(nn.Module):
    def __init__(
        self,
        m: int = M,
        k: int = K,
        t_rounds: int = T_ROUNDS,
        no_feedback: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.k = k
        self.t_rounds = t_rounds
        self.encoder = TXEncoder(m=m, k=k, t_rounds=t_rounds, no_feedback=no_feedback)
        self.decoder = RXDecoder(m=m, k=k, t_rounds=t_rounds)

    def encode_rounds(self, m: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        batch = m.shape[0]
        device = m.device
        x_hist = torch.zeros(batch, self.k, self.t_rounds, device=device)
        y_hist = torch.zeros(batch, self.k, self.t_rounds, device=device)
        for t in range(self.t_rounds):
            x_t = self.encoder(m, x_hist, y_hist, t)
            y_t = awgn(x_t, sigma)
            x_hist = x_hist.clone()
            y_hist = y_hist.clone()
            x_hist[:, :, t] = x_t
            y_hist[:, :, t] = y_t
        return x_hist, y_hist

    def forward(self, m: torch.Tensor, sigma: float) -> torch.Tensor:
        _, y_hist = self.encode_rounds(m, sigma)
        y = y_hist.transpose(1, 2)
        return self.decoder(y)


def smoke_test() -> None:
    model = FeedbackCodeSystem()
    messages = torch.randint(0, M, (16, K))
    logits = model(messages, SIGMA2**0.5)
    assert logits.shape == (16, K, M)
    assert torch.isfinite(logits).all()
    x_hist, _ = model.encode_rounds(messages, SIGMA2**0.5)
    power = (x_hist**2).sum(dim=1).mean(dim=0)
    assert torch.allclose(power, torch.ones_like(power), atol=0.15), power
    print("part2 smoke test passed")


if __name__ == "__main__":
    smoke_test()
