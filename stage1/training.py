from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from modularcar_env import EnvConfig

from .agent import ReplayBuffer, SACAgent
from .config import CBFConfig
from .env_setup import make_stage1_env

__all__ = ["train_stage1"]


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
    log_every: int = 2_000,
) -> Tuple[SACAgent, Dict[str, Any]]:
    env, cbf = make_stage1_env(env_cfg, cbf_cfg, seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=obs_dim,
        action_space=env.action_space,
        gamma=gamma,
        tau=tau,
        lr=lr,
        auto_alpha=True,
    )
    print(f"[train_stage1] Using device: {agent.device}")
    replay = ReplayBuffer(obs_dim, act_dim, capacity=500_000)

    ep_ret = 0.0
    ep_len = 0
    safe_episode_all_r1 = True
    o, info = env.reset(seed=seed)
    t0 = time.time()
    stats = {"avg_ep_ret": [], "avg_ep_len": [], "safe_rate": [], "steps": []}

    for t in range(1, total_steps + 1):
        if t < start_random_steps:
            a = env.action_space.sample()
        else:
            a = agent.select_action(o, eval_mode=False)
        o2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        ep_ret += r
        ep_len += 1
        if info.get("safety_r1", 1.0) < 0.999999:
            safe_episode_all_r1 = False

        replay.add(o, a, r, o2, done)
        o = o2

        if (t >= update_after) and (t % update_every == 0):
            for _ in range(update_every):
                agent.update(replay, batch_size=batch_size)

        if done:
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
            safe_hist = stats["safe_rate"][-20:] if len(stats["safe_rate"]) >= 20 else stats["safe_rate"]
            safe_rate = float(np.mean(safe_hist)) if safe_hist else 0.0
            elapsed = time.time() - t0
            print(
                f"[t={t:7d}] avgR1={avg_ret:6.3f}  avgLen={avg_len:5.1f}  "
                f"safeRate(all r1=1)={safe_rate * 100:5.1f}%  time={elapsed:6.1f}s"
            )

    return agent, {"stats": stats, "env": env, "cbf": cbf}
