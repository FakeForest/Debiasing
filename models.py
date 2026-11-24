
import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def with_hidden(self, x):
        hidden = self.net[:-1](x)
        return self.net[-1](hidden).squeeze(-1), hidden


class Adversary(nn.Module):
    def __init__(self, mode="eq_odds", hidden=0, representation_size=None):
        super().__init__()
        if mode not in ("parity", "eq_odds", "eq_opp"):
            raise ValueError(f"Unknown mode: {mode}")
        if representation_size and mode != "parity":
            raise ValueError("The representation experiment uses parity mode.")
        self.mode = mode
        self.representation_size = representation_size
        if representation_size is None:
            self.c = nn.Parameter(torch.ones(1))
        n_in = representation_size or (4 if mode == "eq_odds" else 1)
        self.head = (nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                     if hidden else nn.Linear(n_in, 1))

    def forward(self, predictor_output, y_true):
        if self.representation_size:
            return self.head(predictor_output).squeeze(-1)
        s = torch.sigmoid((1 + self.c.abs()) * predictor_output)
        if self.mode == "eq_odds":
            feats = torch.stack([s, s * y_true, s * (1 - y_true), y_true], dim=-1)
        else:
            feats = s.unsqueeze(-1)
        return self.head(feats).squeeze(-1)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
