from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from realistic_car_env import NavigationConfig

from stage1 import CBFConfig, SafetyCBF, make_stage1_env
from .agent import Stage2Agent
from .goal_clf import CLFConfig, GoalCLF
from .replay import Stage2ReplayBuffer

__all__ = ["train_stage2"]


def _get_safety_reward(cbf: SafetyCBF, info: Dict[str, Any], obs: np.ndarray, next_obs: np.ndarray) -> float:
    if info.get("safety_r1") is not None:
        return float(info["safety_r1"])
    h_curr, _ = cbf.h(obs)
    h_next, _ = cbf.h(next_obs)
    delta = min(h_next + (cbf.cfg.alpha0 - 1.0) * h_curr, 0.0)
    return float(math.exp(delta))


def train_stage2(
    stage1_weights: str | Path,
    *,
    total_steps: int = 200_000,
    start_random_steps: int = 5_000,
    update_after: int = 5_000,
    update_every: int = 50,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 3e-4,
    alpha_safety: float = 1.0,
    alpha_nav: float = 1.0,
    delta_cos: float = 0.05,
    seed: int = 123,
    env_cfg: Optional[NavigationConfig] = None,
    cbf_cfg: Optional[CBFConfig] = None,
    clf_cfg: Optional[CLFConfig] = None,
    log_every: int = 2_000,
) -> Tuple[Stage2Agent, Dict[str, Any]]:
    env, cbf = make_stage1_env(env_cfg, cbf_cfg, seed=seed)
    base_env = cbf.env
    goal_clf = GoalCLF(base_env, clf_cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = Stage2Agent(
        state_dim=obs_dim,
        action_space=env.action_space,
        gamma=gamma,
        tau=tau,
        lr=lr,
        alpha_safety=alpha_safety,
        alpha_nav=alpha_nav,
        delta_cos=delta_cos,
    )
    agent.load_stage1_weights(stage1_weights)

    replay = Stage2ReplayBuffer(obs_dim, act_dim, capacity=500_000)
    print(f"[train_stage2] Using device: {agent.device}")

    o, info = env.reset(seed=seed)
    ep_ret_nav = 0.0
    ep_len = 0
    safe_episode_all_r1 = True
    t0 = time.time()
    stats = {
        "avg_nav_ret": [],
        "avg_ep_len": [],
        "safe_rate": [],
        "steps": [],
    }

    for t in range(1, total_steps + 1):
        if t < start_random_steps:
            a = env.action_space.sample()
        else:
            a = agent.select_action(o, eval_mode=False)
        o2, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        r1 = _get_safety_reward(cbf, info, o, o2)
        r2, _ = goal_clf.reward(o, o2)
        ep_ret_nav += r2
        ep_len += 1
        if info.get("safety_r1", 1.0) < 0.999999:
            safe_episode_all_r1 = False

        replay.add(o, a, r1, r2, o2, done)
        o = o2

        if (t >= update_after) and (t % update_every == 0):
            for _ in range(update_every):
                batch = replay.sample(batch_size)
                agent.update(batch)

        if done:
            stats["avg_nav_ret"].append(ep_ret_nav)
            stats["avg_ep_len"].append(ep_len)
            stats["safe_rate"].append(1.0 if safe_episode_all_r1 else 0.0)
            stats["steps"].append(t)
            ep_ret_nav = 0.0
            ep_len = 0
            safe_episode_all_r1 = True
            o, info = env.reset()

        if (t % log_every == 0) and len(stats["avg_nav_ret"]) >= 5:
            avg_nav = float(np.mean(stats["avg_nav_ret"][-5:]))
            avg_len = float(np.mean(stats["avg_ep_len"][-5:]))
            safe_hist = stats["safe_rate"][-20:] if len(stats["safe_rate"]) >= 20 else stats["safe_rate"]
            safe_rate = float(np.mean(safe_hist)) if safe_hist else 0.0
            elapsed = time.time() - t0
            print(
                f"[t={t:7d}] Stage2 avgNav={avg_nav:7.3f}  avgLen={avg_len:5.1f}  "
                f"safeRate(all r1=1)={safe_rate * 100:5.1f}%  time={elapsed:6.1f}s"
            )

    return agent, {"stats": stats, "env": env, "cbf": cbf, "goal_clf": goal_clf}
