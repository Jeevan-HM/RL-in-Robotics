from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .nets import TanhGaussianPolicy, TwinQ
from .replay import ReplayBuffer


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1 - tau)
            p_t.data.add_(tau * p.data)


@dataclass
class CACConfig:
    obs_dim: int
    act_dim: int
    hidden: tuple = (256, 256)
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_ent: float = 0.1  # Reduced entropy for more focused behavior
    batch_size: int = 256
    replay_size: int = 1_000_000
    start_steps: int = 1000
    update_after: int = 1000
    update_every: int = 50
    num_updates_per_step: int = 1
    tau: float = 0.005
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    stage: str = "safety"
    restrict_cos_min: float = -0.1  # Stricter constraint (was -0.25)
    max_grad_norm: float = 10.0


class CACAgent(nn.Module):
    def __init__(self, cfg: CACConfig):
        super().__init__()
        self.cfg = cfg
        self.policy = TanhGaussianPolicy(
            cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden
        ).to(cfg.device)
        self.q1_s = TwinQ(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden).to(cfg.device)
        self.q1_s_t = TwinQ(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden).to(cfg.device)
        self.q1_s_t.load_state_dict(self.q1_s.state_dict())
        self.q2_g = TwinQ(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden).to(cfg.device)
        self.q2_g_t = TwinQ(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden).to(cfg.device)
        self.q2_g_t.load_state_dict(self.q2_g.state_dict())

        self.pi_optim = Adam(self.policy.parameters(), lr=cfg.actor_lr)
        self.q1_optim = Adam(self.q1_s.parameters(), lr=cfg.critic_lr)
        self.q2_optim = Adam(self.q2_g.parameters(), lr=cfg.critic_lr)

        self.replay = ReplayBuffer(cfg.obs_dim, cfg.act_dim, cfg.replay_size)

    def act(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        obs_t = torch.tensor(
            obs, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(obs_t)
                a = torch.tanh(mean)
            else:
                a, _, _ = self.policy.sample(obs_t)
        return a.cpu().numpy()[0]

    def update_critics(self, batch: Dict[str, np.ndarray], stage: str):
        cfg = self.cfg
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=cfg.device)
        act = torch.tensor(batch["act"], dtype=torch.float32, device=cfg.device)
        rew = torch.tensor(
            batch["rew"], dtype=torch.float32, device=cfg.device
        ).unsqueeze(-1)
        done = torch.tensor(
            batch["done"], dtype=torch.float32, device=cfg.device
        ).unsqueeze(-1)
        obs2 = torch.tensor(batch["obs2"], dtype=torch.float32, device=cfg.device)

        with torch.no_grad():
            a2, logp2, _ = self.policy.sample(obs2)
            if stage == "safety":
                q1t = self.q1_s_t.min_q(obs2, a2) - cfg.alpha_ent * logp2
            else:
                q1t = self.q2_g_t.min_q(obs2, a2) - cfg.alpha_ent * logp2
            target = rew + cfg.gamma * (1.0 - done) * q1t

        if stage == "safety":
            q1, q2 = self.q1_s.forward(obs, act)
            loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.q1_optim.zero_grad(set_to_none=True)
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.q1_s.parameters(), cfg.max_grad_norm)
            self.q1_optim.step()
            soft_update(self.q1_s_t, self.q1_s, cfg.tau)
            return {"loss_q_s": float(loss_q.item())}
        else:
            q1, q2 = self.q2_g.forward(obs, act)
            loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.q2_optim.zero_grad(set_to_none=True)
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.q2_g.parameters(), cfg.max_grad_norm)
            self.q2_optim.step()
            soft_update(self.q2_g_t, self.q2_g, cfg.tau)
            return {"loss_q_g": float(loss_q.item())}

    def update_actor_sac(self, batch: Dict[str, np.ndarray], stage: str):
        cfg = self.cfg
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=cfg.device)
        a, logp, _ = self.policy.sample(obs)
        if stage == "safety":
            q = self.q1_s.min_q(obs, a)
        else:
            q = self.q2_g.min_q(obs, a)
        loss_pi = (cfg.alpha_ent * logp - q).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.pi_optim.step()
        return {"loss_pi": float(loss_pi.item())}

    def _flatten_grads(self, params):
        return torch.cat(
            [
                p.grad.view(-1) if p.grad is not None else torch.zeros_like(p.view(-1))
                for p in params
            ]
        )

    def _set_grads_from_flat(self, params, vec):
        offset = 0
        for p in params:
            n = p.numel()
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(vec[offset : offset + n].view_as(p))
            offset += n

    def update_actor_restricted(self, batch: Dict[str, np.ndarray]):
        cfg = self.cfg
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=cfg.device)

        a2, logp2, _ = self.policy.sample(obs)
        q2 = self.q2_g.min_q(obs, a2)
        loss2 = (cfg.alpha_ent * logp2 - q2).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        loss2.backward(retain_graph=True)
        g2 = self._flatten_grads(self.policy.parameters()).detach()

        a1, logp1, _ = self.policy.sample(obs)
        q1 = self.q1_s.min_q(obs, a1)
        loss1 = (cfg.alpha_ent * logp1 - q1).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        loss1.backward()
        g1 = self._flatten_grads(self.policy.parameters()).detach()

        dot = torch.dot(g1, g2)
        if dot.item() < 0.0:
            e = g2 - (dot / (g1.norm() ** 2 + 1e-12)) * g1
        else:
            e = g2

        cos = torch.dot(e, g1) / (e.norm() * g1.norm() + 1e-12)
        if cos.item() < cfg.restrict_cos_min:
            c_min = cfg.restrict_cos_min
            g1n2 = g1.norm() ** 2 + 1e-12
            lam = (c_min * e.norm() * g1.norm() - torch.dot(e, g1)) / g1n2
            e = e + lam * g1

        if e.norm().item() > g2.norm().item():
            e = e * (g2.norm() / (e.norm() + 1e-12))

        self.pi_optim.zero_grad(set_to_none=True)
        self._set_grads_from_flat(self.policy.parameters(), e)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.pi_optim.step()

        return {
            "cos_e_g1": float(
                (torch.dot(e, g1) / (e.norm() * g1.norm() + 1e-12)).item()
            ),
            "g2_norm": float(g2.norm().item()),
            "e_norm": float(e.norm().item()),
        }

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None):
        payload = {
            "cfg": asdict(self.cfg),
            "policy": self.policy.state_dict(),
            "q1_s": self.q1_s.state_dict(),
            "q1_s_t": self.q1_s_t.state_dict(),
            "q2_g": self.q2_g.state_dict(),
            "q2_g_t": self.q2_g_t.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str):
        payload = torch.load(path, map_location="cpu")
        cfg = CACConfig(**payload["cfg"])
        agent = CACAgent(cfg)
        agent.policy.load_state_dict(payload["policy"])
        agent.q1_s.load_state_dict(payload["q1_s"])
        agent.q1_s_t.load_state_dict(payload["q1_s_t"])
        agent.q2_g.load_state_dict(payload["q2_g"])
        agent.q2_g_t.load_state_dict(payload["q2_g_t"])
        return agent, payload.get("extra", {})
