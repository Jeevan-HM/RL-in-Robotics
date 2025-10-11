from typing import Iterable, List, Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal

def mlp(sizes: Iterable[int], act=nn.ReLU, out_act=None):
    layers: List[nn.Module] = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)

class TanhGaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy with state-dependent mean and log_std."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256,256), log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 2*act_dim])
        self.log_std_bounds = log_std_bounds
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return y_t, log_prob, mean_action

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256,256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))

class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256,256)):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden)
        self.q2 = QNetwork(obs_dim, act_dim, hidden)

    def forward(self, obs, act):
        return self.q1(obs, act), self.q2(obs, act)

    def min_q(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)
