"""Stage-2 training entry point.

Uses the Stage-1 safety policy/critic as a starting point, adds a navigation critic,
and trains with the restricted policy gradient update from CAC Algorithm 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stage1 import CBFConfig, _default_env_cfg
from stage2 import CLFConfig, save_stage2_agent, train_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 (navigation) CAC agent.")
    parser.add_argument("--stage1-weights", type=Path, required=True, help="Stage-1 actor/safety checkpoint (.pt).")
    parser.add_argument("--checkpoint", type=Path, default=Path("stage2_agent.pt"), help="Where to store the Stage-2 checkpoint.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-random-steps", type=int, default=5_000)
    parser.add_argument("--update-after", type=int, default=5_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-safety", type=float, default=1.0)
    parser.add_argument("--alpha-nav", type=float, default=1.0)
    parser.add_argument("--delta-cos", type=float, default=0.05, help="Safety margin for restricted gradient (cosine).")
    parser.add_argument("--beta0", type=float, default=0.85, help="Stage-2 CLF decay (must be in (0, 1)).")
    parser.add_argument("--log-every", type=int, default=5_000)
    return parser.parse_args()


def main():
    args = parse_args()
    env_cfg = _default_env_cfg()
    cbf_cfg = CBFConfig(alpha_cbf=5.0, alpha0=0.2, d_safe_point=0.8, d_safe_car=1.0)
    clf_cfg = CLFConfig(beta0=args.beta0)

    agent, out = train_stage2(
        stage1_weights=args.stage1_weights,
        total_steps=args.total_steps,
        start_random_steps=args.start_random_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        alpha_safety=args.alpha_safety,
        alpha_nav=args.alpha_nav,
        delta_cos=args.delta_cos,
        seed=args.seed,
        env_cfg=env_cfg,
        cbf_cfg=cbf_cfg,
        clf_cfg=clf_cfg,
        log_every=args.log_every,
    )

    metadata = {
        "total_steps": args.total_steps,
        "seed": args.seed,
        "env_cfg": env_cfg,
        "cbf_cfg": cbf_cfg,
        "clf_cfg": clf_cfg,
        "alpha_safety": args.alpha_safety,
        "alpha_nav": args.alpha_nav,
        "delta_cos": args.delta_cos,
    }
    save_stage2_agent(agent, args.checkpoint, metadata=metadata)
    print(f"\nStage-2 checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
