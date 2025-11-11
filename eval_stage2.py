"""Stage-2 evaluation entry point.

Loads Stage-2 weights and runs deterministic rollouts to report navigation and safety metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stage1 import CBFConfig, _default_env_cfg, make_stage1_env
from stage2 import (
    CLFConfig,
    GoalCLF,
    Stage2Agent,
    evaluate_stage2,
    load_stage2_agent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 navigation policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("stage2_agent.pt"), help="Checkpoint produced by train_stage2.py")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation rollouts.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-safety", type=float, default=1.0)
    parser.add_argument("--alpha-nav", type=float, default=1.0)
    parser.add_argument("--delta-cos", type=float, default=0.05)
    parser.add_argument("--beta0", type=float, default=0.85, help="CLF decay used for the navigation reward.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for env construction/reset before evaluation.")
    parser.add_argument("--eval-seed", type=int, default=123, help="Seed for evaluation RNG.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per episode.")
    return parser.parse_args()


def main():
    args = parse_args()
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
    metadata = load_stage2_agent(agent, args.checkpoint)
    if metadata is not None:
        print(f"Loaded checkpoint metadata: {list(metadata.keys())}")

    evaluate_stage2(
        env,
        agent,
        goal_clf,
        cbf,
        episodes=args.episodes,
        gamma=args.gamma,
        max_steps=args.max_steps,
        seed=args.eval_seed,
    )


if __name__ == "__main__":
    main()
