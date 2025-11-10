"""
Stage 1 (Safety Critic Construction) for ModularCar2DEnv
-----------------------------------------------------------------
This script implements:
  • A CBF-based safety reward wrapper (r1 = exp(min(h(s_{t+1}) + (α0-1) h(s_t), 0)))
  • A SafetyCBF helper that reads lidar + clearances from observations and computes h(s)
  • A minimal Soft Actor-Critic (SAC) agent trained ONLY on r1 (Stage 1)
  • Evaluation utilities to validate Stage 1 per the paper:
      - Monte-Carlo safety certificate (return of r1)
      - Safe-episode rate using δh ≥ 0 across entire episode
      - (Optional) action–Q heatmap for qualitative inspection

Reference methodology: Certificated Actor-Critic (CAC)
  - Discrete-time CBF inequality and exponential reward normalization (Eq. 12 in paper)
  - Stage 1 learns a safety critic and a safe policy before any goal reward

Environment dependency:
  from ModularCar2DEnv import ModularCar2DEnv, EnvConfig

Python deps:
  numpy, torch, gymnasium, matplotlib (only for optional plots)

Author: (you)
"""
from __future__ import annotations
import math
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# ---------------------------------------------------------------------------
# Try to import the provided environment
# ---------------------------------------------------------------------------
try:
    from rl_env import ModularCar2DEnv, EnvConfig
except Exception as e:
    raise ImportError("Could not import ModularCar2DEnv. Ensure this file is alongside ModularCar2DEnv.py") from e

# ---------------------------------------------------------------------------
# Safety CBF helper
# ---------------------------------------------------------------------------

@dataclass
class CBFConfig:
    alpha_cbf: float = 5.0         # h(s) = alpha_cbf * (d_clear - d_safe)
    alpha0: float = 0.2            # discrete-time CBF decay, used in δh
    d_safe_point: float = 0.8      # safety buffer (meters) for point-mass model
    d_safe_car: float = 1.0        # safety buffer (meters) for bicycle car model


class SafetyCBF:
    """Computes the discrete-time Control Barrier Function h(s) from env observations.

    Observation layout mirrors the env's _get_obs() construction. We recompute the same
    indexes to extract clearances and lidar rays robustly for any EnvConfig.
    """

    def __init__(self, env: ModularCar2DEnv, cfg: Optional[CBFConfig] = None):
        self.env = env
        self.cfg: CBFConfig = cfg or CBFConfig()
        self._idx = self._compute_index_map()

    # --- index helpers -------------------------------------------------------
    def _compute_index_map(self) -> Dict[str, Any]:
        cfg = self.env.cfg
        i = 0
        # base features
        i += 4  # px, py, vx, vy
        if cfg.include_applied_accel:
            i += 2  # ax, ay
        i += 2  # goal dx, dy
        speed_idx = None
        if cfg.include_speed:
            speed_idx = i
            i += 1
        wall_idx = obstacle_idx = None
        if cfg.include_clearances:
            wall_idx = i
            obstacle_idx = i + 1
            i += 2
        if (self.env._model == "car") and cfg.include_heading:
            i += 1
        if (self.env._model == "car") and cfg.include_steering_angle:
            i += 1
        # obstacle triplets
        obs_triplets = 3 * cfg.n_obstacles
        obstacle_feats_start = i
        i += obs_triplets
        # lidar
        lidar_start = lidar_end = None
        if cfg.sensor_in_obs:
            lidar_start = i
            lidar_end = i + cfg.num_rays
            i = lidar_end
        return {
            "speed_idx": speed_idx,
            "wall_clear_idx": wall_idx,
            "obs_clear_idx": obstacle_idx,
            "obs_triplets_start": obstacle_feats_start,
            "lidar_start": lidar_start,
            "lidar_end": lidar_end,
            "obs_dim": i,
        }

    # --- core CBF ------------------------------------------------------------
    def h(self, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Return h(s) and diagnostics.

        h(s) = alpha_cbf * (d_clear - d_safe),
        where d_clear = min( min_lidar, wall_clearance, obstacle_clearance ).
        """
        assert obs.ndim == 1, "obs must be 1D array"
        lid_s, lid_e = self._idx["lidar_start"], self._idx["lidar_end"]
        wall_i, obs_i = self._idx["wall_clear_idx"], self._idx["obs_clear_idx"]

        # lidar min distance
        if lid_s is not None and lid_e is not None:
            d_ray = float(np.min(obs[lid_s:lid_e]))
        else:
            d_ray = float("inf")

        # clearances (can be negative if intersecting obstacle)
        comps = [d_ray]
        if wall_i is not None:
            comps.append(float(obs[wall_i]))
        if obs_i is not None:
            comps.append(float(obs[obs_i]))
        d_clear = float(np.min(comps)) if len(comps) else d_ray

        # pick stand-off based on model
        d_safe = self.cfg.d_safe_car if self.env._model == "car" else self.cfg.d_safe_point
        h_val = float(self.cfg.alpha_cbf * (d_clear - d_safe))
        return h_val, {"d_ray": d_ray, "d_clear": d_clear, "d_safe": d_safe}


# ---------------------------------------------------------------------------
# Gym wrapper that replaces reward with Stage-1 safety reward r1
# ---------------------------------------------------------------------------

class SafetyRewardWrapper(gym.Wrapper):
    """Wraps env to provide r1 = exp(min(h(s_{t+1}) + (α0-1) h(s_t), 0)).
    Also exposes diagnostics in info dict.
    """
    def __init__(self, env: ModularCar2DEnv, cbf: SafetyCBF):
        super().__init__(env)
        self.cbf = cbf
        self.alpha0 = cbf.cfg.alpha0
        self._prev_obs = None
        self._prev_h = None
        self.cfg = env.cfg  # expose wrapped env config for evaluation helpers

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs.copy()
        self._prev_h, diag = self.cbf.h(obs)
        info = dict(info)
        info.update({
            "cbf_h": float(self._prev_h),
            "cbf_d_clear": float(diag["d_clear"]),
            "cbf_d_ray": float(diag["d_ray"]),
            "safety_delta_h": None,
            "safety_r1": None,
        })
        return obs, info

    def step(self, action):
        obs_next, _, terminated, truncated, info = self.env.step(action)
        h_next, diag = self.cbf.h(obs_next)
        # δh = min(h(s_{t+1}) + (α0 - 1) h(s_t), 0)
        delta_h = float(min(h_next + (self.alpha0 - 1.0) * self._prev_h, 0.0))
        r1 = float(math.exp(delta_h))  # r1 ∈ (0,1]

        info = dict(info)
        info.update({
            "cbf_h_prev": float(self._prev_h),
            "cbf_h_next": float(h_next),
            "cbf_d_clear": float(diag["d_clear"]),
            "cbf_d_ray": float(diag["d_ray"]),
            "safety_delta_h": float(delta_h),
            "safety_r1": float(r1),
        })

        # update memory
        self._prev_h = h_next
        self._prev_obs = obs_next.copy()
        return obs_next, r1, terminated, truncated, info


# ---------------------------------------------------------------------------
# SAC Agent (minimal, twin Q, auto-entropy, tanh-squashed Gaussian policy)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act=nn.ReLU):
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
    def __init__(self, state_dim, action_space: gym.spaces.Box, hidden=(256, 256)):
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
        # tanh to (-1,1) then affine to [low, high]
        a = torch.tanh(u)
        scale = (self.action_high - self.action_low) / 2.0
        mean = (self.action_high + self.action_low) / 2.0
        return a * scale + mean

    def forward(self, s: torch.Tensor, deterministic=False, with_logprob=True):
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
        # log prob with tanh correction
        if with_logprob:
            # base log prob under N(mu, std)
            logp = -0.5 * (((u - mu) / (std + 1e-8))**2 + 2 * log_std + math.log(2 * math.pi))
            logp = torch.sum(logp, dim=-1, keepdim=True)
            # tanh correction: sum log(1 - tanh(u)^2)
            logp -= torch.sum(torch.log(1.0 - a * a + 1e-6), dim=-1, keepdim=True)
        else:
            logp = None
        # scale to action space
        scale = (self.action_high - self.action_low) / 2.0
        mean = (self.action_high + self.action_low) / 2.0
        action = a * scale + mean
        return action, logp


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=1_000_000):
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

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[idx], dtype=torch.float32),
            torch.as_tensor(self.action[idx], dtype=torch.float32),
            torch.as_tensor(self.reward[idx], dtype=torch.float32),
            torch.as_tensor(self.next_state[idx], dtype=torch.float32),
            torch.as_tensor(self.not_done[idx], dtype=torch.float32),
        )


class SACAgent:
    def __init__(self, state_dim, action_space: gym.spaces.Box,
                 actor_hidden=(256, 256), critic_hidden=(256, 256),
                 gamma=0.99, tau=0.005, lr=3e-4, auto_alpha=True, target_entropy=None, device=None):
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

    def select_action(self, obs: np.ndarray, eval_mode=False) -> np.ndarray:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.actor(s, deterministic=eval_mode, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer, batch_size=256):
        s, a, r, s2, not_done = replay.sample(batch_size)
        s = s.to(self.device); a = a.to(self.device)
        r = r.to(self.device); s2 = s2.to(self.device); not_done = not_done.to(self.device)

        # --- Critic update ---
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
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # --- Actor update ---
        a_pi, logp_pi = self.actor(s, deterministic=False, with_logprob=True)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # --- Alpha (entropy) update ---
        if self.auto_alpha:
            alpha_loss = (self.alpha * (-logp_pi - self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # --- Target update ---
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_t.data.mul_(1.0 - self.tau); p_t.data.add_(self.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_t.data.mul_(1.0 - self.tau); p_t.data.add_(self.tau * p.data)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    # --- Convenience: estimate V1(s) ≈ E_{a~π}[ Q(s,a) ] by sampling ---
    def estimate_V(self, obs: np.ndarray, samples: int = 64) -> float:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(samples, 1)
        with torch.no_grad():
            a, _ = self.actor(s, deterministic=False, with_logprob=False)
            q1 = self.q1(s, a)
            q2 = self.q2(s, a)
            q = torch.min(q1, q2)
        return float(q.mean().cpu().item())


# ---------------------------------------------------------------------------
# Training loop (Stage 1)
# ---------------------------------------------------------------------------

def make_stage1_env(env_cfg: Optional[EnvConfig] = None,
                    cbf_cfg: Optional[CBFConfig] = None,
                    seed: Optional[int] = None) -> Tuple[gym.Env, SafetyCBF]:
    env = ModularCar2DEnv(env_cfg or EnvConfig())
    if seed is not None:
        env.reset(seed=seed)
    cbf = SafetyCBF(env, cbf_cfg)
    wrapped = SafetyRewardWrapper(env, cbf)
    return wrapped, cbf


def train_stage1(
    total_steps: int = 200_000,
    start_random_steps: int = 5_000,
    update_after: int = 5_000,
    update_every: int = 50,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 3e-4,
    seed: int = 42,
    env_cfg: Optional[EnvConfig] = None,
    cbf_cfg: Optional[CBFConfig] = None,
    log_every: int = 2000,
) -> Tuple[SACAgent, Dict[str, Any]]:
    env, cbf = make_stage1_env(env_cfg, cbf_cfg, seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=obs_dim,
        action_space=env.action_space,
        gamma=gamma, tau=tau, lr=lr,
        auto_alpha=True,
    )

    replay = ReplayBuffer(obs_dim, act_dim, capacity=500_000)

    # rollouts
    ep_ret = 0.0
    ep_len = 0
    safe_episode_all_r1 = True  # track δh==0 across episode
    o, info = env.reset(seed=seed)
    episode = 0

    t0 = time.time()
    stats = {"avg_ep_ret": [], "avg_ep_len": [], "safe_rate": [], "steps": []}

    for t in range(1, total_steps + 1):
        if t < start_random_steps:
            a = env.action_space.sample()
        else:
            a = agent.select_action(o, eval_mode=False)
        o2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # accumulate metrics
        ep_ret += r
        ep_len += 1
        # r1==exp(δh); safe iff δh==0 → r1==1
        if info.get("safety_r1", 1.0) < 0.999999:
            safe_episode_all_r1 = False

        # store transition
        replay.add(o, a, r, o2, done)
        o = o2

        # update
        if (t >= update_after) and (t % update_every == 0):
            for _ in range(update_every):
                agent.update(replay, batch_size=batch_size)

        # episode end
        if done:
            episode += 1
            stats["avg_ep_ret"].append(ep_ret)
            stats["avg_ep_len"].append(ep_len)
            stats["safe_rate"].append(1.0 if safe_episode_all_r1 else 0.0)
            stats["steps"].append(t)
            ep_ret, ep_len = 0.0, 0
            safe_episode_all_r1 = True
            o, info = env.reset()

        if (t % log_every == 0) and len(stats["avg_ep_ret"]) >= 5:
            avg_ret = float(np.mean(stats["avg_ep_ret"][-5:]))
            avg_len = float(np.mean(stats["avg_ep_len"][-5:]))
            safe_rate = float(np.mean(stats["safe_rate"][-20:])) if len(stats["safe_rate"])>=20 else float(np.mean(stats["safe_rate"]))
            elapsed = time.time() - t0
            print(f"[t={t:7d}] avgR1={avg_ret:6.3f}  avgLen={avg_len:5.1f}  safeRate(all r1=1)={safe_rate*100:5.1f}%  time={elapsed:6.1f}s")

    return agent, {"stats": stats, "env": env, "cbf": cbf}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ensure_checkpoint_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _to_serializable_meta(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def save_agent(agent: SACAgent, path: str | Path, metadata: Optional[Dict[str, Any]] = None):
    """Persist SAC weights (and optional metadata) for later evaluation."""
    meta = {k: _to_serializable_meta(v) for k, v in (metadata or {}).items()}
    checkpoint = {
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_targ": agent.q1_targ.state_dict(),
        "q2_targ": agent.q2_targ.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "actor_opt": agent.actor_opt.state_dict(),
        "q1_opt": agent.q1_opt.state_dict(),
        "q2_opt": agent.q2_opt.state_dict(),
        "alpha_opt": agent.alpha_opt.state_dict(),
        "metadata": meta,
    }
    torch.save(checkpoint, _ensure_checkpoint_path(path))


def load_agent(agent: SACAgent, path: str | Path, map_location: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load SAC weights into an existing agent. Returns stored metadata, if any."""
    load_kwargs = {"map_location": map_location or agent.device, "weights_only": False}
    try:
        checkpoint = torch.load(Path(path), **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        checkpoint = torch.load(Path(path), **load_kwargs)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.q1.load_state_dict(checkpoint["q1"])
    agent.q2.load_state_dict(checkpoint["q2"])
    agent.q1_targ.load_state_dict(checkpoint["q1_targ"])
    agent.q2_targ.load_state_dict(checkpoint["q2_targ"])
    log_alpha = checkpoint.get("log_alpha")
    if log_alpha is not None:
        if isinstance(log_alpha, torch.Tensor):
            agent.log_alpha.data = log_alpha.to(agent.device)
        else:
            agent.log_alpha.data = torch.tensor(float(log_alpha), device=agent.device)
    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "q1_opt" in checkpoint:
        agent.q1_opt.load_state_dict(checkpoint["q1_opt"])
    if "q2_opt" in checkpoint:
        agent.q2_opt.load_state_dict(checkpoint["q2_opt"])
    if agent.auto_alpha and "alpha_opt" in checkpoint:
        agent.alpha_opt.load_state_dict(checkpoint["alpha_opt"])
    return checkpoint.get("metadata")


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def theoretical_Vmax(gamma: float, kmax: int) -> float:
    return (1.0 - (gamma ** kmax)) / (1.0 - gamma) if gamma < 0.999999 else float(kmax)


def evaluate_stage1(
    env: SafetyRewardWrapper,
    agent: SACAgent,
    episodes: int = 20,
    gamma: float = 0.99,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run deterministic rollouts to estimate safety metrics.

    Metrics:
      - avg_return_r1: Monte-Carlo estimate of V1
      - safe_rate_all_r1: fraction of episodes with δh==0 for all steps (i.e., r1==1 always)
      - collision_rate: fraction of episodes that ended in 'collision' (from env info)
      - avg_len
      - avg_predicted_V1: average over episodes of agent.estimate_V at the first obs
      - Vmax: theoretical bound given gamma and kmax
    """
    rng = np.random.RandomState(seed if seed is not None else 0)
    returns = []
    lengths = []
    safe_flags = []
    collisions = 0
    predicted_Vs = []

    for ep in range(episodes):
        o, info = env.reset(seed=rng.randint(0, 10_000))
        # predicted V at start state
        predicted_Vs.append(agent.estimate_V(o, samples=64))

        disc = 1.0
        ret = 0.0
        safe_all = True
        length = 0
        for t in range(0, env.cfg.max_steps if max_steps is None else max_steps):
            a = agent.select_action(o, eval_mode=True)
            o2, r, terminated, truncated, info = env.step(a)
            ret += disc * r
            disc *= gamma
            length += 1
            if info.get("safety_r1", 1.0) < 0.999999:
                safe_all = False
            if terminated or truncated:
                if info.get("terminated_reason", "") == "collision":
                    collisions += 1
                break
            o = o2

        returns.append(ret)
        lengths.append(length)
        safe_flags.append(1.0 if safe_all else 0.0)

    avg_ret = float(np.mean(returns))
    avg_len = float(np.mean(lengths))
    safe_rate = float(np.mean(safe_flags))
    coll_rate = float(collisions / episodes)
    avg_predV = float(np.mean(predicted_Vs))
    Vmax = theoretical_Vmax(gamma, int(avg_len))

    summary = {
        "avg_return_r1": avg_ret,
        "avg_len": avg_len,
        "safe_rate_all_r1": safe_rate,
        "collision_rate": coll_rate,
        "avg_predicted_V1": avg_predV,
        "Vmax_at_avg_len": Vmax,
    }
    print("\n=== Stage-1 Evaluation (Safety Certificate) ===")
    for k, v in summary.items():
        print(f"{k:>22s}: {v:.4f}")
    print("(Note: avg_return_r1 should approach Vmax if episodes are fully safe.)")
    return summary


# Optional: visualize Q(s, a) along a grid of actions at a fixed observation

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
            a_batch = np.stack([A0[i], A1[i]], axis=1)  # (grid, 2)
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
    plt.imshow(Z, origin='lower', extent=[a0.min(), a0.max(), a1.min(), a1.max()], aspect='auto')
    plt.colorbar(label='Q value (safety)')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Quickstart runner
# ---------------------------------------------------------------------------

def _default_env_cfg() -> EnvConfig:
    cfg = EnvConfig()
    # Ensure required signals are present for CBF
    cfg.sensor_in_obs = True
    cfg.include_clearances = True
    # You may switch to the car model here if desired:
    # cfg.vehicle_model = "car"; cfg.allow_reverse = False
    return cfg


def main():
    env_cfg = _default_env_cfg()
    cbf_cfg = CBFConfig(alpha_cbf=5.0, alpha0=0.2, d_safe_point=0.8, d_safe_car=1.0)

    agent, out = train_stage1(
        total_steps=200_000,
        start_random_steps=5_000,
        update_after=5_000,
        update_every=50,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        seed=42,
        env_cfg=env_cfg,
        cbf_cfg=cbf_cfg,
        log_every=5000,
    )

    # Evaluation
    env = out["env"]  # already a SafetyRewardWrapper
    evaluate_stage1(env, agent, episodes=20, gamma=0.99, seed=123)

    # Optional qualitative check: action–Q heatmap at reset state
    if _HAS_PLT and env.action_space.shape[0] == 2:
        obs, _ = env.reset(seed=321)
        try:
            plot_action_Q_heatmap(env, agent, obs, grid=51)
        except AssertionError:
            pass


if __name__ == "__main__":
    main()
