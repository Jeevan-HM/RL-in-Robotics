from __future__ import annotations

import numpy as np
import torch

from .agent import SACAgent
from .wrappers import SafetyRewardWrapper

try:
    import matplotlib.pyplot as plt

    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

__all__ = ["plot_action_Q_heatmap", "_HAS_PLT"]


def plot_action_Q_heatmap(env: SafetyRewardWrapper, agent: SACAgent, obs: np.ndarray, grid: int = 41):
    if not _HAS_PLT:
        print("matplotlib not available; skipping heatmap plot.")
        return
    low = env.action_space.low
    high = env.action_space.high
    assert low.shape[0] == 2 and high.shape[0] == 2, "Heatmap expects 2D action space."
    a0 = np.linspace(low[0], high[0], grid)
    a1 = np.linspace(low[1], high[1], grid)
    A0, A1 = np.meshgrid(a0, a1)
    Z = np.zeros_like(A0)
    s = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        for i in range(grid):
            a_batch = np.stack([A0[i], A1[i]], axis=1)
            a_t = torch.as_tensor(a_batch, dtype=torch.float32, device=agent.device)
            s_rep = s.repeat(grid, 1)
            q1 = agent.q1(s_rep, a_t)
            q2 = agent.q2(s_rep, a_t)
            q = torch.min(q1, q2).squeeze(-1)
            Z[i] = q.cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.title("Safety Critic Q(s, a) heatmap")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.imshow(Z, origin="lower", extent=[a0.min(), a0.max(), a1.min(), a1.max()], aspect="auto")
    plt.colorbar(label="Q value (safety)")
    plt.tight_layout()
    plt.show()
