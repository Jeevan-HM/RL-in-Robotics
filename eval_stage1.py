"""Stage-1 evaluation entry point.

Loads saved SAC weights and runs the safety-certificate evaluation loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stage1_safety_cac_modularcar import (
    CBFConfig,
    _default_env_cfg,
    make_stage1_env,
    SACAgent,
    evaluate_stage1,
    load_agent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 safety policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("stage1_sac_agent.pt"), help="Checkpoint produced by train_stage1.py")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation rollouts.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42, help="Seed for env construction/reset before evaluation.")
    parser.add_argument("--eval-seed", type=int, default=123, help="Seed for evaluation RNG.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per episode.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{args.checkpoint}' not found. Run train_stage1.py first.")

    env_cfg = _default_env_cfg()
    cbf_cfg = CBFConfig(alpha_cbf=5.0, alpha0=0.2, d_safe_point=0.8, d_safe_car=1.0)
    env, _ = make_stage1_env(env_cfg=env_cfg, cbf_cfg=cbf_cfg, seed=args.seed)

    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        auto_alpha=True,
    )
    metadata = load_agent(agent, args.checkpoint)
    if metadata is not None:
        print(f"Loaded checkpoint metadata: {list(metadata.keys())}")

    evaluate_stage1(env, agent, episodes=args.episodes, gamma=args.gamma, max_steps=args.max_steps, seed=args.eval_seed)


if __name__ == "__main__":
    main()
