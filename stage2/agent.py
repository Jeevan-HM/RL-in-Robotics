from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from stage1.networks import Actor, QNet

__all__ = ["Stage2Agent", "Stage2UpdateStats"]


@dataclass
class Stage2UpdateStats:
    q_safety_loss: float
    q_nav_loss: float
    j_safety: float
    j_nav: float
    proj_case: str


class Stage2Agent:
    """Stage-2 agent that enforces the restricted policy gradient update."""

    def __init__(
        self,
        state_dim: int,
        action_space,
        actor_hidden: Sequence[int] = (256, 256),
        critic_hidden: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha_safety: float = 1.0,
        alpha_nav: float = 1.0,
        delta_cos: float = 0.05,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha_safety = alpha_safety
        self.alpha_nav = alpha_nav
        self.delta_cos = delta_cos
        self.action_space = action_space
        self.act_dim = int(np.prod(action_space.shape))

        self.actor = Actor(state_dim, action_space, actor_hidden).to(self.device)
        self.safety_q = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.safety_q_targ = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.nav_q = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.nav_q_targ = QNet(state_dim, self.act_dim, critic_hidden).to(self.device)
        self.safety_q_targ.load_state_dict(self.safety_q.state_dict())
        self.nav_q_targ.load_state_dict(self.nav_q.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.safety_q_opt = torch.optim.Adam(self.safety_q.parameters(), lr=lr)
        self.nav_q_opt = torch.optim.Adam(self.nav_q.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    # I/O helpers
    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.actor(s, deterministic=eval_mode, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def load_stage1_weights(self, checkpoint: str | Path):
        """Load Stage-1 actor/safety critic weights from a checkpoint file."""
        ckpt = torch.load(Path(checkpoint), map_location=self.device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.actor.load_state_dict(ckpt["actor"])
        safety_state = ckpt.get("q1") or ckpt.get("safety_q") or ckpt.get("critic")
        if safety_state is None:
            raise KeyError("Checkpoint does not include a safety critic ('q1').")
        self.safety_q.load_state_dict(safety_state)
        self.safety_q_targ.load_state_dict(self.safety_q.state_dict())

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def update(self, batch) -> Stage2UpdateStats:
        s, a, r_safety, r_nav, s2, not_done = batch
        s = s.to(self.device)
        a = a.to(self.device)
        r_safety = r_safety.to(self.device)
        r_nav = r_nav.to(self.device)
        s2 = s2.to(self.device)
        not_done = not_done.to(self.device)

        with torch.no_grad():
            a2, logp2 = self.actor(s2, deterministic=False, with_logprob=True)
            q_nav_t = self.nav_q_targ(s2, a2)
            y_nav = r_nav + not_done * self.gamma * (q_nav_t - self.alpha_nav * logp2)
            q_safety_t = self.safety_q_targ(s2, a2)
            y_safety = r_safety + not_done * self.gamma * (q_safety_t - self.alpha_safety * logp2)

        q_nav = self.nav_q(s, a)
        loss_nav = F.mse_loss(q_nav, y_nav)
        self.nav_q_opt.zero_grad()
        loss_nav.backward()
        self.nav_q_opt.step()

        q_safety = self.safety_q(s, a)
        loss_safety = F.mse_loss(q_safety, y_safety)
        self.safety_q_opt.zero_grad()
        loss_safety.backward()
        self.safety_q_opt.step()

        self._soft_update(self.nav_q, self.nav_q_targ)
        self._soft_update(self.safety_q, self.safety_q_targ)

        actor_stats = self._restricted_actor_step(s)
        return Stage2UpdateStats(
            q_safety_loss=float(loss_safety.item()),
            q_nav_loss=float(loss_nav.item()),
            j_safety=actor_stats["j_safety"],
            j_nav=actor_stats["j_nav"],
            proj_case=actor_stats["proj_case"],
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _restricted_actor_step(self, states: torch.Tensor) -> Dict[str, float]:
        params = list(self.actor.parameters())
        a_pi, logp = self.actor(states, deterministic=False, with_logprob=True)
        j_nav = (self.alpha_nav * logp - self.nav_q(states, a_pi)).mean()
        j_safety = (self.alpha_safety * logp - self.safety_q(states, a_pi)).mean()

        grads_nav = torch.autograd.grad(j_nav, params, retain_graph=True, allow_unused=True)
        grads_safety = torch.autograd.grad(j_safety, params, allow_unused=True)

        g2 = self._flatten_grad(grads_nav, params)
        g1 = self._flatten_grad(grads_safety, params)

        proj_case, eff = self._project_direction(g1, g2)
        grad_tensors = self._unflatten_grad(eff, params)

        self.actor_opt.zero_grad(set_to_none=True)
        for p, g in zip(params, grad_tensors):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)
        self.actor_opt.step()
        return {
            "j_nav": float(j_nav.item()),
            "j_safety": float(j_safety.item()),
            "proj_case": proj_case,
        }

    def _flatten_grad(self, grads: List[Optional[torch.Tensor]], params: List[torch.nn.Parameter]) -> torch.Tensor:
        flat = []
        for g, p in zip(grads, params):
            if g is None:
                flat.append(torch.zeros_like(p).reshape(-1))
            else:
                flat.append(g.reshape(-1))
        return torch.cat(flat)

    def _unflatten_grad(self, flat: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        grads: List[torch.Tensor] = []
        idx = 0
        for p in params:
            num = p.numel()
            grads.append(flat[idx : idx + num].reshape_as(p))
            idx += num
        return grads

    def _project_direction(self, g1: torch.Tensor, g2: torch.Tensor) -> tuple[str, torch.Tensor]:
        g1 = g1.detach()
        g2 = g2.detach()
        g1_norm_sq = torch.dot(g1, g1) + 1e-12
        g2_norm_sq = torch.dot(g2, g2)
        if g2_norm_sq.item() == 0.0:
            return "zero-nav", torch.zeros_like(g1)
        dot12 = torch.dot(g2, g1)
        if dot12 >= 0:
            eff = g2
            case = "aligned"
        else:
            eff = g2 - (dot12 / g1_norm_sq) * g1
            case = "projected"
        if self.delta_cos > 0.0:
            e_norm = torch.linalg.vector_norm(eff) + 1e-12
            g1_norm = torch.sqrt(g1_norm_sq)
            need = self.delta_cos * e_norm * g1_norm
            e_dot_g1 = torch.dot(eff, g1)
            if e_dot_g1 < need:
                tau = (need - e_dot_g1) / g1_norm_sq
                eff = eff + tau * g1
                case = "margin-adjusted"
        return case, eff

    def _soft_update(self, src: torch.nn.Module, dst: torch.nn.Module):
        with torch.no_grad():
            for p, p_t in zip(src.parameters(), dst.parameters()):
                p_t.data.mul_(1.0 - self.tau)
                p_t.data.add_(self.tau * p.data)
