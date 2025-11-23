from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

__all__ = ["Stage2ReplayBuffer"]


class Stage2ReplayBuffer:
    """Replay buffer that keeps both safety and navigation rewards per transition."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.not_done = np.zeros((capacity, 1), dtype=np.float32)
        self.r_safety = np.zeros((capacity, 1), dtype=np.float32)
        self.r_nav = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        r_safety: float,
        r_nav: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.r_safety[self.ptr] = r_safety
        self.r_nav[self.ptr] = r_nav
        self.next_state[self.ptr] = next_state
        self.not_done[self.ptr] = 1.0 - float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[idx], dtype=torch.float32),
            torch.as_tensor(self.action[idx], dtype=torch.float32),
            torch.as_tensor(self.r_safety[idx], dtype=torch.float32),
            torch.as_tensor(self.r_nav[idx], dtype=torch.float32),
            torch.as_tensor(self.next_state[idx], dtype=torch.float32),
            torch.as_tensor(self.not_done[idx], dtype=torch.float32),
        )
