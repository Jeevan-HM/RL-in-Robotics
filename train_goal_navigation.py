import argparse
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from cac.algorithms import CACAgent, CACConfig
from envs.planar_nav import PlanarNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["safety", "goal"], default="goal")
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--safety_model",
        type=str,
        required=True,
        help="Path to pre-trained safety model",
    )
    parser.add_argument(
        "--actor_lr", type=float, default=1e-4
    )  # Lower LR for fine-tuning
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_ent", type=float, default=0.1)
    parser.add_argument(
        "--restrict_cos_min", type=float, default=-0.15
    )  # Stricter constraint
    args = parser.parse_args()

    print(f"🎯 Training GOAL-seeking behavior from safety model: {args.safety_model}")

    # Create environment matching the safety model configuration
    env = PlanarNavEnv(
        seed=args.seed,
        safety_margin=0.25,  # Match safety training
        alpha=0.93,  # Match safety training
        enhanced_obs=True,  # Use enhanced observations
        n_rays=19,  # Match safety training
        fov_deg=130,  # Match safety training
    )
    env.set_stage(args.stage)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Environment: obs_dim={obs_dim}, act_dim={act_dim}")

    # Load the pre-trained safety model
    agent, _ = CACAgent.load(args.safety_model)
    print(f"✅ Loaded safety model with {agent.cfg.obs_dim}-dim observations")

    # Update configuration for goal training
    agent.cfg.stage = args.stage
    agent.cfg.actor_lr = args.actor_lr
    agent.cfg.critic_lr = args.critic_lr
    agent.cfg.alpha_ent = args.alpha_ent
    agent.cfg.restrict_cos_min = args.restrict_cos_min

    # Reinitialize optimizers with new learning rates
    from torch.optim import Adam

    agent.pi_optim = Adam(agent.policy.parameters(), lr=args.actor_lr)
    agent.q1_optim = Adam(agent.q1_s.parameters(), lr=args.critic_lr)
    agent.q2_optim = Adam(agent.q2_g.parameters(), lr=args.critic_lr)

    run_name = args.run_name or f"goal_from_safety_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    ckpt_path = os.path.join("runs", f"{run_name}_last.pt")

    o, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_len = 0
    sum_r1 = 0.0
    sum_r2 = 0.0

    # Goal-specific tracking
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_episodes = 0
    min_clearance_sum = 0.0
    goal_distance_sum = 0.0

    print("🚀 Starting goal training...")
    for t in trange(args.steps, desc=f"Training {args.stage} (from safety)"):
        # Use pre-trained policy from the start (no random exploration)
        a = agent.act(o)

        o2, r, term, trunc, info = env.step(a)
        done = term or trunc
        agent.replay.store(o, a, r, done, o2)
        ep_ret += r
        ep_len += 1
        sum_r1 += info.get("r1", 0.0)
        sum_r2 += info.get("r2", 0.0)

        # Track goal-specific metrics
        current_clearance = env._min_range(env.p, env.theta) - env.R
        min_clearance_sum += current_clearance
        goal_distance = np.linalg.norm(env.goal - env.p)
        goal_distance_sum += goal_distance

        o = o2

        if done:
            total_episodes += 1
            termination = info.get("termination", "unknown")

            if termination == "goal":
                success_count += 1
            elif termination == "collision":
                collision_count += 1
            elif termination == "timeout":
                timeout_count += 1

            # Enhanced logging
            writer.add_scalar("ep/return", ep_ret, t)
            writer.add_scalar("ep/len", ep_len, t)
            writer.add_scalar("ep/r1_sum", sum_r1, t)
            writer.add_scalar("ep/r2_sum", sum_r2, t)
            writer.add_scalar("ep/avg_clearance", min_clearance_sum / ep_len, t)
            writer.add_scalar("ep/avg_goal_distance", goal_distance_sum / ep_len, t)

            if total_episodes > 0:
                writer.add_scalar(
                    "goal/success_rate", success_count / total_episodes, t
                )
                writer.add_scalar(
                    "goal/collision_rate", collision_count / total_episodes, t
                )
                writer.add_scalar(
                    "goal/timeout_rate", timeout_count / total_episodes, t
                )

            # Print comprehensive progress
            if total_episodes % 25 == 0 and total_episodes > 0:
                success_rate = success_count / total_episodes * 100
                collision_rate = collision_count / total_episodes * 100
                timeout_rate = timeout_count / total_episodes * 100
                avg_clearance = min_clearance_sum / ep_len
                avg_goal_dist = goal_distance_sum / ep_len

                print(f"\n📊 Episode {total_episodes:3d}:")
                print(
                    f"   Success: {success_rate:5.1f}% | Collision: {collision_rate:5.1f}% | Timeout: {timeout_rate:5.1f}%"
                )
                print(
                    f"   Return: {ep_ret:6.1f} | Clearance: {avg_clearance:.3f} | Goal Dist: {avg_goal_dist:.3f}"
                )

            o, _ = env.reset()
            env.set_stage(args.stage)
            ep_ret = ep_len = 0
            sum_r1 = sum_r2 = 0
            min_clearance_sum = 0
            goal_distance_sum = 0

        # Training updates (start immediately since we have a pre-trained model)
        if (
            agent.replay.size >= agent.cfg.batch_size
            and t % agent.cfg.update_every == 0
        ):
            for _ in range(agent.cfg.update_every * agent.cfg.num_updates_per_step):
                batch = agent.replay.sample_batch(agent.cfg.batch_size)

                # Update both critics
                stats_q1 = agent.update_critics(batch, stage="safety")
                stats_q2 = agent.update_critics(batch, stage="goal")
                for k, v in {**stats_q1, **stats_q2}.items():
                    writer.add_scalar(f"train/{k}", v, t)

                # Use restricted update that balances safety and goal objectives
                stats_pi = agent.update_actor_restricted(batch)
                for k, v in stats_pi.items():
                    writer.add_scalar(f"train/{k}", v, t)

        # Save checkpoint
        if (t + 1) % 15000 == 0:
            agent.save(ckpt_path)
            print(f"\n💾 Checkpoint saved at step {t + 1}")

    agent.save(ckpt_path)
    print(f"\n🎉 Goal training completed!")
    if total_episodes > 0:
        print(f"Final success rate: {success_count / total_episodes * 100:.1f}%")
        print(f"Final collision rate: {collision_count / total_episodes * 100:.1f}%")
        print(f"Final timeout rate: {timeout_count / total_episodes * 100:.1f}%")
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
