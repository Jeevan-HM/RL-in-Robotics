from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, QNet

__all__ = ["ReplayBuffer", "SACAgent"]


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.not_done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, s2, done):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.not_done[self.ptr] = 1.0 - float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[idx], dtype=torch.float32),
            torch.as_tensor(self.action[idx], dtype=torch.float32),
            torch.as_tensor(self.reward[idx], dtype=torch.float32),
            torch.as_tensor(self.next_state[idx], dtype=torch.float32),
            torch.as_tensor(self.not_done[idx], dtype=torch.float32),
        )


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_space,
        actor_hidden: Sequence[int] = (256, 256),
        critic_hidden: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        self.act_dim = int(np.prod(action_space.shape))

        self.actor = Actor(state_dim, action_space, actor_hidden).to(self.device)
        self.q1 = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.q2 = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.q1_targ = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.q2_targ = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -float(self.act_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.actor(s, deterministic=eval_mode, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer, batch_size: int = 256):
        s, a, r, s2, not_done = replay.sample(batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        not_done = not_done.to(self.device)

        with torch.no_grad():
            a2, logp2 = self.actor(s2, deterministic=False, with_logprob=True)
            q1_t = self.q1_targ(s2, a2)
            q2_t = self.q2_targ(s2, a2)
            q_targ = torch.min(q1_t, q2_t) - self.alpha * logp2
            y = r + not_done * self.gamma * q_targ
        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        a_pi, logp_pi = self.actor(s, deterministic=False, with_logprob=True)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss = (self.alpha * (-logp_pi - self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0)

        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_t.data.mul_(1.0 - self.tau)
                p_t.data.add_(self.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_t.data.mul_(1.0 - self.tau)
                p_t.data.add_(self.tau * p.data)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def estimate_V(self, obs: np.ndarray, samples: int = 64) -> float:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(samples, 1)
        with torch.no_grad():
            a, _ = self.actor(s, deterministic=False, with_logprob=False)
            q1 = self.q1(s, a)
            q2 = self.q2(s, a)
            q = torch.min(q1, q2)
        return float(q.mean().cpu().item())
