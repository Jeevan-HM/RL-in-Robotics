from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["MLP", "Actor", "QNet"]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_space, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self._act_dim = int(np.prod(action_space.shape))
        self.net = MLP(state_dim, 2 * self._act_dim, hidden)
        self.LOG_STD_MIN = -20.0
        self.LOG_STD_MAX = 2.0

    def _squash_scale(self, u: torch.Tensor):
        a = torch.tanh(u)
        scale = (self.action_high - self.action_low) / 2.0
        mean = (self.action_high + self.action_low) / 2.0
        return a * scale + mean

    def forward(self, s: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        mu_logstd = self.net(s)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        if deterministic:
            u = mu
        else:
            eps = torch.randn_like(mu)
            u = mu + std * eps
        a = torch.tanh(u)
        if with_logprob:
            logp = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
            logp = torch.sum(logp, dim=-1, keepdim=True)
            logp -= torch.sum(torch.log(1.0 - a * a + 1e-6), dim=-1, keepdim=True)
        else:
            logp = None
        scale = (self.action_high - self.action_low) / 2.0
        mean = (self.action_high + self.action_low) / 2.0
        action = a * scale + mean
        return action, logp


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)
