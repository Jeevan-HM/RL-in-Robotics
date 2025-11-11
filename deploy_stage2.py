#!/usr/bin/env python3
"""Visual rollout for a trained Stage-2 navigation agent."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from stage1 import CBFConfig, _default_env_cfg, make_stage1_env
from stage2 import CLFConfig, GoalCLF, Stage2Agent, load_stage2_agent


def _ensure_interactive_backend() -> None:
    """Switch Matplotlib to an interactive backend if necessary."""
    try:
        import matplotlib

        backend = matplotlib.get_backend().lower()
        if backend in {"agg", "cairoagg", "figurecanvasagg"}:
            for candidate in ("QtAgg", "Qt5Agg", "TkAgg"):
                try:
                    matplotlib.use(candidate, force=True)
                    print(f"Switching Matplotlib backend to {candidate}")
                    break
                except Exception:
                    continue
        import matplotlib.pyplot as plt

        backend = plt.get_backend().lower()
        if backend in {"agg", "cairoagg", "figurecanvasagg"}:
            raise RuntimeError(
                "Matplotlib is using a non-interactive backend (Agg). "
                "Set MPLBACKEND=TkAgg (or QtAgg/Qt5Agg) before running, "
                "and ensure the corresponding GUI toolkit is installed."
            )
    except Exception as exc:
        raise RuntimeError(
            "matplotlib with an interactive backend is required for rendering. "
            "Install it via `python -m pip install matplotlib pyqt5` (or tkinter)."
        ) from exc


def _safety_reward_from_obs(cbf, obs, next_obs) -> float:
    h_curr, _ = cbf.h(obs)
    h_next, _ = cbf.h(next_obs)
    delta = min(h_next + (cbf.cfg.alpha0 - 1.0) * h_curr, 0.0)
    return float(math.exp(delta))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a trained Stage-2 SAC agent.")
    parser.add_argument("--checkpoint", type=Path, default=Path("stage2_agent.pt"), help="Checkpoint file produced by train_stage2.py.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of rendered episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps per episode (defaults to env.cfg.max_steps).")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for rollout resets.")
    parser.add_argument("--sleep-scale", type=float, default=1.0, help="Multiply env.dt by this factor between frames (set <1 to speed up).")
    parser.add_argument("--headless", action="store_true", help="Skip env.render() (useful for quick smoke tests).")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-safety", type=float, default=1.0)
    parser.add_argument("--alpha-nav", type=float, default=1.0)
    parser.add_argument("--delta-cos", type=float, default=0.05)
    parser.add_argument("--beta0", type=float, default=0.85, help="CLF decay used for the navigation reward.")
    return parser.parse_args()


def run_visualization(args: argparse.Namespace) -> None:
    if not args.headless:
        _ensure_interactive_backend()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{args.checkpoint}' not found. Run train_stage2.py first.")

    env_cfg = _default_env_cfg()
    cbf_cfg = CBFConfig(alpha_cbf=5.0, alpha0=0.2, d_safe_point=0.8, d_safe_car=1.0)
    env, cbf = make_stage1_env(env_cfg=env_cfg, cbf_cfg=cbf_cfg, seed=args.seed)
    goal_clf = GoalCLF(cbf.env, CLFConfig(beta0=args.beta0))

    agent = Stage2Agent(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        alpha_safety=args.alpha_safety,
        alpha_nav=args.alpha_nav,
        delta_cos=args.delta_cos,
    )
    load_stage2_agent(agent, args.checkpoint)

    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        steps = 0
        nav_return = 0.0
        safety_return = 0.0
        done = False
        terminated = False
        truncated = False
        print(f"\nEpisode {ep}:")
        while not done and (args.max_steps is None or steps < args.max_steps):
            action = agent.select_action(obs, eval_mode=True)
            next_obs, _, terminated, truncated, info = env.step(action)
            r_nav, _ = goal_clf.reward(obs, next_obs)
            r_safe = info.get("safety_r1")
            if r_safe is None:
                r_safe = _safety_reward_from_obs(cbf, obs, next_obs)
            nav_return += r_nav
            safety_return += float(r_safe)
            steps += 1
            done = terminated or truncated
            obs = next_obs
            if not args.headless:
                env.render()
                time.sleep(max(0.0, env.cfg.dt * args.sleep_scale))
        if terminated or truncated:
            reason = info.get("terminated_reason") or ("truncated" if truncated else "terminated")
        else:
            reason = "max_steps"
        print(f"  steps={steps}  nav_return={nav_return:.2f}  safety_r1_sum={safety_return:.2f}  reason={reason}")

    env.close()


def main():
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
