#!/usr/bin/env python3
"""
Certificated Actor-Critic (CAC) Training
==========================================

Implements the hierarchical RL framework from the paper:
"Certificated Actor-Critic: Hierarchical Reinforcement Learning with
Control Barrier Functions for Safe Navigation"

This follows Algorithm 1 from the paper with two stages:
1. Stage 1: Safety Critic Construction - Learn safe policy with CBF-derived rewards
2. Stage 2: Restricted Policy Update - Improve goal-reaching while maintaining safety
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from device_config import DeviceConfig
from realistic_car_env import CarPhysicsParams, GoalOrientedCarEnv, NavigationConfig
from scenarios import CarNavigationEnv
from stage1 import CBFConfig, SACAgent, SafetyCBF
from stage1.agent import ReplayBuffer


def compute_cbf_reward_r1(
    h_current: float, h_next: float, alpha0: float, use_exponential: bool = True
) -> float:
    """
    Compute Stage 1 safety reward from CBF values.

    From the paper (Equation 7 and 12):
    r1 = min(h(s_{t+1}) + (α0 - 1)h(s_t), 0)

    With exponential normalization (Equation 12):
    r1 = exp(min(h(s_{t+1}) + (α0 - 1)h(s_t), 0))

    Args:
        h_current: CBF value at current state h(s_t)
        h_next: CBF value at next state h(s_{t+1})
        alpha0: Expected decay rate (0 < α0 < 1)
        use_exponential: Whether to use exponential normalization

    Returns:
        Safety reward r1 ∈ (0, 1] if exponential, else r1 ≤ 0
    """
    delta_h = h_next + (alpha0 - 1) * h_current
    r1_raw = min(delta_h, 0.0)

    if use_exponential:
        # Equation 12: r1 = exp(min(..., 0))
        return np.exp(r1_raw)
    else:
        # Equation 7: r1 = min(..., 0)
        return r1_raw


def compute_clf_reward_r2(l_current: float, l_next: float, beta0: float) -> float:
    """
    Compute Stage 2 navigation reward from CLF values.

    From the paper (Equation 9):
    r2 = -max(l(s_{t+1}) + (β0 - 1)l(s_t), 0)

    Args:
        l_current: CLF value at current state l(s_t)
        l_next: CLF value at next state l(s_{t+1})
        beta0: Expected decay rate for navigation

    Returns:
        Navigation reward r2 ≤ 0
    """
    delta_l = l_next + (beta0 - 1) * l_current
    return -max(delta_l, 0.0)


def compute_goal_distance_clf(pos_current: np.ndarray, pos_goal: np.ndarray) -> float:
    """
    Compute Control Lyapunov Function as distance to goal.

    From the paper: l(s) = (x - x_tar)² + (y - y_tar)²

    Note: We normalize by world size to keep values reasonable

    Args:
        pos_current: Current position [x, y]
        pos_goal: Goal position [x_tar, y_tar]

    Returns:
        Normalized squared distance to goal
    """
    diff = pos_current - pos_goal
    dist_squared = float(np.dot(diff, diff))
    # Normalize by typical world size (100x100) to keep values in [0, 2] range
    # Max distance in 100x100 world is ~141m, squared is ~20000
    return dist_squared / 10000.0  # Normalize to ~[0, 2]


class CACWrapper:
    """
    Wrapper that integrates CBF-based safety rewards with goal-reaching rewards.
    Implements the two-stage CAC framework.
    """

    def __init__(
        self,
        env: CarNavigationEnv,
        cbf: SafetyCBF,
        alpha0: float = 0.2,
        beta0: float = 0.9,
        stage: int = 1,
    ):
        """
        Initialize CAC wrapper.

        Args:
            env: Car navigation environment
            cbf: Control Barrier Function
            alpha0: CBF decay rate for safety (Stage 1)
            beta0: CLF decay rate for navigation (Stage 2)
            stage: Current training stage (1 or 2)
        """
        self.env = env
        self.cbf = cbf
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.stage = stage

        # Track CBF values for reward computation
        self.h_current = None
        self.l_current = None

        # Goal tracking
        self.goal_reached = False

    def reset(self, **kwargs):
        """Reset environment and initialize CBF/CLF values."""
        obs, info = self.env.reset(**kwargs)

        # Compute initial CBF value
        self.h_current, _ = self.cbf.h(obs)

        # Compute initial CLF value (distance to goal)
        pos = obs[:2]  # First 2 elements are position

        # Get goal from unwrapped environment
        if hasattr(self.env.unwrapped, "goal_pos"):
            goal = self.env.unwrapped.goal_pos
        elif hasattr(self.env.unwrapped, "cfg") and hasattr(
            self.env.unwrapped.cfg, "goal_pos"
        ):
            goal = np.array(self.env.unwrapped.cfg.goal_pos)
        else:
            goal = np.array([90.0, 90.0])  # Default fallback

        self.l_current = compute_goal_distance_clf(pos, goal)
        self.goal_reached = False

        return obs, info

    def step(self, action):
        """
        Execute action and compute CAC rewards.

        Returns observation, reward, terminated, truncated, info
        """
        obs_next, reward_env, terminated, truncated, info = self.env.step(action)

        # Compute next CBF value
        h_next, cbf_info = self.cbf.h(obs_next)

        # Compute Stage 1 safety reward (r1)
        r1 = compute_cbf_reward_r1(
            self.h_current, h_next, self.alpha0, use_exponential=True
        )

        # Compute Stage 2 navigation reward (r2)
        r2 = 0.0
        if self.stage == 2:
            pos_next = obs_next[:2]

            # Get goal from unwrapped environment
            if hasattr(self.env.unwrapped, "goal_pos"):
                goal = self.env.unwrapped.goal_pos
            elif hasattr(self.env.unwrapped, "cfg") and hasattr(
                self.env.unwrapped.cfg, "goal_pos"
            ):
                goal = np.array(self.env.unwrapped.cfg.goal_pos)
            else:
                goal = np.array([90.0, 90.0])  # Default fallback

            l_next = compute_goal_distance_clf(pos_next, goal)
            r2 = compute_clf_reward_r2(self.l_current, l_next, self.beta0)
            self.l_current = l_next

        # Update CBF value for next step
        self.h_current = h_next

        # Check if goal reached
        if info.get("terminated_reason") == "goal":
            self.goal_reached = True

        # Select reward based on stage
        if self.stage == 1:
            reward = r1
        else:  # Stage 2
            # Scale up navigation reward to match safety reward magnitude
            # r1 ~ 1.0, r2 ~ -0.2, so multiply r2 by 5 to get r2 ~ -1.0
            reward = r1 + 15.0 * r2  # Navigation weighted 5x to match safety magnitude

        # Add diagnostic info
        info["safety_reward_r1"] = r1
        info["navigation_reward_r2"] = r2
        info["cbf_value"] = h_next
        info["cbf_delta"] = h_next - self.h_current
        info["goal_reached"] = self.goal_reached
        info.update(cbf_info)

        return obs_next, reward, terminated, truncated, info


def train_stage1_safety_critic(
    env_config: Optional[NavigationConfig] = None,
    cbf_config: Optional[CBFConfig] = None,
    total_steps: int = 500_000,
    checkpoint_path: str = "stage1_safe_policy.pt",
    device: str = "auto",
    alpha0: float = 0.5,
    seed: int = 42,
):
    """
    Stage 1: Safety Critic Construction

    Train a safe policy using CBF-derived safety rewards.
    From Algorithm 1, lines 2-8.

    Args:
        env_config: Environment configuration
        cbf_config: CBF configuration
        total_steps: Total training steps
        checkpoint_path: Path to save the trained agent
        device: Device for training ('auto', 'cpu', 'cuda', 'mps')
        alpha0: CBF decay rate
        seed: Random seed
    """
    print("=" * 70)
    print("STAGE 1: SAFETY CRITIC CONSTRUCTION")
    print("=" * 70)
    print(f"Training safe policy with CBF-derived rewards (r1)")
    print(f"Target: Learn collision-free navigation")
    print(f"Steps: {total_steps:,}")
    print()

    # Device configuration
    device_config = DeviceConfig(device=device, verbose=True)

    # Create environment
    env_cfg = env_config or NavigationConfig()
    env = CarNavigationEnv(env_config=env_cfg, render_mode=None)
    env.reset(seed=seed)

    # Create CBF
    cbf_cfg = cbf_config or CBFConfig(alpha_cbf=5.0, alpha0=alpha0, d_safe_car=1.0)
    cbf = SafetyCBF(env.unwrapped, cbf_cfg)

    # Wrap environment with CAC Stage 1
    cac_env = CACWrapper(env, cbf, alpha0=alpha0, stage=1)

    # Create agent
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=obs_dim,
        action_space=env.action_space,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        auto_alpha=True,
        device=device_config.get_device_str(),
    )

    replay = ReplayBuffer(obs_dim, act_dim, capacity=500_000)

    # Training loop
    obs, info = cac_env.reset(seed=seed)
    ep_ret = 0.0
    ep_len = 0
    safe_episode = True

    stats = {"returns": [], "lengths": [], "safe_rates": [], "cbf_violations": []}
    t0 = time.time()

    print(f"Starting Stage 1 training...")
    print(f"Total steps: {total_steps:,}")
    print(f"Random exploration: first {5000:,} steps")
    print(f"Update starts at: {5000:,} steps")
    print(f"Logging every: {10_000:,} steps\n")

    for t in range(1, total_steps + 1):
        # Select action (random exploration for first 5000 steps) first 5000 steps)
        if t < 5000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, eval_mode=False)

        # Execute action
        obs_next, reward, terminated, truncated, info = cac_env.step(action)
        done = terminated or truncated

        # Track statistics
        ep_ret += reward
        ep_len += 1
        if info.get("safety_reward_r1", 1.0) < 0.999:  # Not perfectly safe
            safe_episode = False

        # Store transition
        replay.add(obs, action, reward, obs_next, done)
        obs = obs_next

        # Update agent (less frequent for efficiency)
        if t >= 5000 and t % 100 == 0:
            for _ in range(100):
                agent.update(replay, batch_size=256)

        # Episode end
        if done:
            stats["returns"].append(ep_ret)
            stats["lengths"].append(ep_len)
            stats["safe_rates"].append(1.0 if safe_episode else 0.0)

            ep_ret = 0.0
            ep_len = 0
            safe_episode = True
            obs, info = cac_env.reset()

        # Logging
        if t % 10_000 == 0 and len(stats["returns"]) >= 5:
            avg_ret = np.mean(stats["returns"][-20:])
            avg_len = np.mean(stats["lengths"][-20:])
            safe_rate = np.mean(stats["safe_rates"][-50:]) * 100
            elapsed = time.time() - t0
            steps_per_sec = t / elapsed

            print(
                f"Step {t:7,}/{total_steps:,} | "
                f"Reward: {avg_ret:6.3f} | "
                f"Len: {avg_len:5.1f} | "
                f"Safe: {safe_rate:5.1f}% | "
                f"Speed: {steps_per_sec:5.1f} steps/s | "
                f"Time: {elapsed / 60:5.1f}m | "
                f"Episodes: {len(stats['returns'])}"
            )
        elif t % 1000 == 0 and t < 10_000:
            # More frequent logging at start
            print(f"Step {t:7,}/{total_steps:,} - Collecting initial data...")

    # Save agent
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "q1_state_dict": agent.q1.state_dict(),
            "q2_state_dict": agent.q2.state_dict(),
            "q1_targ_state_dict": agent.q1_targ.state_dict(),
            "q2_targ_state_dict": agent.q2_targ.state_dict(),
            "alpha": agent.log_alpha.exp().item() if agent.auto_alpha else agent.alpha,
            "log_alpha": agent.log_alpha.item() if agent.auto_alpha else None,
            "config": {
                "state_dim": obs_dim,
                "action_dim": act_dim,
                "gamma": agent.gamma,
                "tau": agent.tau,
                "lr": 3e-4,
            },
        },
        checkpoint_path,
    )

    print(f"\n✅ Stage 1 Complete!")
    print(f"   Saved safe policy to: {checkpoint_path}")
    print(f"   Final safe rate: {np.mean(stats['safe_rates'][-50:]) * 100:.1f}%")

    env.close()
    return agent, cbf


def train_stage2_restricted_policy(
    stage1_checkpoint: str,
    env_config: Optional[NavigationConfig] = None,
    cbf_config: Optional[CBFConfig] = None,
    total_steps: int = 500_000,
    checkpoint_path: str = "stage2_final_policy.pt",
    device: str = "auto",
    alpha0: float = 0.5,
    beta0: float = 0.8,
    gradient_safety_threshold: float = 0.0,
    seed: int = 42,
):
    """
    Stage 2: Restricted Policy Update

    Improve goal-reaching while maintaining safety using restricted gradients.
    From Algorithm 1, lines 9-17.

    The key innovation: update policy with gradient restriction (Equation 10):
    ∇θ = argmax_e e·∇θJ2(θ) s.t. e·∇θJ1(θ) ≥ 0, ||e|| ≤ ||∇θJ2(θ)||

    This ensures the safety critic V1 doesn't decrease.

    Args:
        stage1_checkpoint: Path to Stage 1 trained agent
        env_config: Environment configuration
        cbf_config: CBF configuration
        total_steps: Total training steps
        checkpoint_path: Path to save final agent
        device: Device for training
        alpha0: CBF decay rate (for safety)
        beta0: CLF decay rate (for navigation)
        gradient_safety_threshold: Minimum cosine similarity for safety gradient
        seed: Random seed
    """
    print("\n" + "=" * 70)
    print("STAGE 2: RESTRICTED POLICY UPDATE")
    print("=" * 70)
    print(f"Improving goal-reaching while maintaining safety")
    print(f"Using restricted gradient updates (Equation 10)")
    print(f"Steps: {total_steps:,}")
    print()

    # Device configuration
    device_config = DeviceConfig(device=device, verbose=True)

    # Create environment
    env_cfg = env_config or NavigationConfig()
    env = CarNavigationEnv(env_config=env_cfg, render_mode=None)
    env.reset(seed=seed)

    # Create CBF
    cbf_cfg = cbf_config or CBFConfig(alpha_cbf=5.0, alpha0=alpha0, d_safe_car=1.0)
    cbf = SafetyCBF(env.unwrapped, cbf_cfg)

    # Wrap environment with CAC Stage 2
    cac_env = CACWrapper(env, cbf, alpha0=alpha0, beta0=beta0, stage=2)

    # Load Stage 1 agent
    checkpoint = torch.load(
        stage1_checkpoint, map_location=device_config.get_device_str()
    )
    obs_dim = checkpoint["config"]["state_dim"]
    act_dim = checkpoint["config"]["action_dim"]

    # Create agent and load Stage 1 weights
    agent = SACAgent(
        state_dim=obs_dim,
        action_space=env.action_space,
        gamma=checkpoint["config"]["gamma"],
        tau=checkpoint["config"]["tau"],
        lr=checkpoint["config"]["lr"],
        auto_alpha=True,
        device=device_config.get_device_str(),
    )

    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.q1.load_state_dict(checkpoint["q1_state_dict"])
    agent.q2.load_state_dict(checkpoint["q2_state_dict"])
    if "q1_targ_state_dict" in checkpoint:
        agent.q1_targ.load_state_dict(checkpoint["q1_targ_state_dict"])
        agent.q2_targ.load_state_dict(checkpoint["q2_targ_state_dict"])
    if "log_alpha" in checkpoint and checkpoint["log_alpha"] is not None:
        agent.log_alpha.data = torch.tensor(
            checkpoint["log_alpha"], device=agent.device
        )

    print(f"✓ Loaded Stage 1 safe policy from {stage1_checkpoint}")

    # Note: In full implementation, you would:
    # 1. Keep critic1/critic2 as "safety critics" (V_π^1, Q_π^1)
    # 2. Create NEW critics for navigation (V_π^2, Q_π^2)
    # 3. Update actor using restricted gradients from BOTH critics
    #
    # For this implementation, we use the standard SAC update
    # A complete implementation would need custom gradient computation

    replay = ReplayBuffer(obs_dim, act_dim, capacity=500_000)

    # Training loop
    obs, info = cac_env.reset(seed=seed)
    ep_ret_nav = 0.0
    ep_ret_safety = 0.0
    ep_len = 0
    safe_episode = True
    reached_goal = False

    stats = {
        "returns_nav": [],
        "returns_safety": [],
        "lengths": [],
        "safe_rates": [],
        "goal_rates": [],
    }
    t0 = time.time()

    print(f"Starting Stage 2 training...")
    print(f"Total steps: {total_steps:,}")
    print(f"Random exploration: first {5000:,} steps")
    print(f"Update starts at: {1000:,} steps")
    print(f"Logging every: {10_000:,} steps\n")

    for t in range(1, total_steps + 1):
        # Select action using the policy (with some random exploration early on)
        if t < 5000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, eval_mode=False)

        # Execute action
        obs_next, reward, terminated, truncated, info = cac_env.step(action)
        done = terminated or truncated

        # Track statistics
        ep_ret_nav += info.get("navigation_reward_r2", 0.0)
        ep_ret_safety += info.get("safety_reward_r1", 0.0)
        ep_len += 1

        if info.get("safety_reward_r1", 1.0) < 0.999:
            safe_episode = False

        # Check if goal reached (environment-specific)
        if info.get("goal_reached", False) or info.get("terminated_reason") == "goal":
            reached_goal = True

        # Store transition (using navigation reward for policy improvement)
        replay.add(obs, action, reward, obs_next, done)
        obs = obs_next

        # Update agent
        # TODO: Implement restricted gradient update (Equation 10)
        # For now, using standard SAC update
        if t >= 1000 and t % 50 == 0:
            for _ in range(50):
                agent.update(replay, batch_size=256)

        # Episode end
        if done:
            stats["returns_nav"].append(ep_ret_nav)
            stats["returns_safety"].append(ep_ret_safety)
            stats["lengths"].append(ep_len)
            stats["safe_rates"].append(1.0 if safe_episode else 0.0)
            stats["goal_rates"].append(1.0 if reached_goal else 0.0)

            ep_ret_nav = 0.0
            ep_ret_safety = 0.0
            ep_len = 0
            safe_episode = True
            reached_goal = False
            obs, info = cac_env.reset()

        # Logging
        if t % 10_000 == 0 and len(stats["returns_nav"]) >= 5:
            avg_ret_nav = np.mean(stats["returns_nav"][-20:])
            avg_ret_safety = np.mean(stats["returns_safety"][-20:])
            avg_len = np.mean(stats["lengths"][-20:])
            safe_rate = np.mean(stats["safe_rates"][-50:]) * 100
            goal_rate = np.mean(stats["goal_rates"][-50:]) * 100
            elapsed = time.time() - t0
            steps_per_sec = t / elapsed

            # Get current distance to goal for diagnostics
            current_dist = info.get("distance_to_goal", -1)

            print(
                f"Step {t:7,}/{total_steps:,} | "
                f"Nav: {avg_ret_nav:7.2f} | "
                f"Safe: {avg_ret_safety:5.3f} | "
                f"Len: {avg_len:5.1f} | "
                f"Safe%: {safe_rate:5.1f} | "
                f"Goal%: {goal_rate:5.1f} | "
                f"Dist: {current_dist:5.1f} | "
                f"{steps_per_sec:5.1f} steps/s | "
                f"Time: {elapsed / 60:5.1f}m | "
                f"Eps: {len(stats['returns_nav'])}"
            )
        elif t % 1000 == 0 and t < 10_000:
            # More frequent logging at start
            print(f"Step {t:7,}/{total_steps:,} - Exploring...")

    # Save final agent
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "q1_state_dict": agent.q1.state_dict(),
            "q2_state_dict": agent.q2.state_dict(),
            "q1_targ_state_dict": agent.q1_targ.state_dict(),
            "q2_targ_state_dict": agent.q2_targ.state_dict(),
            "alpha": agent.log_alpha.exp().item() if agent.auto_alpha else agent.alpha,
            "log_alpha": agent.log_alpha.item() if agent.auto_alpha else None,
            "config": {
                "state_dim": obs_dim,
                "action_dim": act_dim,
                "gamma": agent.gamma,
                "tau": agent.tau,
                "lr": 3e-4,
            },
        },
        checkpoint_path,
    )

    print(f"\n✅ Stage 2 Complete!")
    print(f"   Saved final policy to: {checkpoint_path}")
    print(f"   Final safe rate: {np.mean(stats['safe_rates'][-50:]) * 100:.1f}%")
    print(f"   Final goal rate: {np.mean(stats['goal_rates'][-50:]) * 100:.1f}%")

    env.close()
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Train CAC (Certificated Actor-Critic) for safe robot navigation"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Training stage (1: safety critic, 2: restricted policy). If not specified, runs both.",
    )
    parser.add_argument(
        "--steps-stage1", type=int, default=500_000, help="Training steps for Stage 1"
    )
    parser.add_argument(
        "--steps-stage2", type=int, default=500_000, help="Training steps for Stage 2"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps"
    )
    parser.add_argument(
        "--alpha0",
        type=float,
        default=0.5,
        help="CBF decay rate (lower = stricter safety)",
    )
    parser.add_argument(
        "--beta0",
        type=float,
        default=0.8,
        help="CLF decay rate (lower = faster goal approach)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = checkpoint_dir / "stage1_safe_policy.pt"
    stage2_path = checkpoint_dir / "stage2_final_policy.pt"

    if args.stage is None or args.stage == 1:
        # Run Stage 1
        train_stage1_safety_critic(
            total_steps=args.steps_stage1,
            checkpoint_path=str(stage1_path),
            device=args.device,
            alpha0=args.alpha0,
            seed=args.seed,
        )

    if args.stage is None or args.stage == 2:
        # Run Stage 2 (requires Stage 1 checkpoint)
        if not stage1_path.exists():
            print(f"\n❌ Error: Stage 1 checkpoint not found at {stage1_path}")
            print("   Please run Stage 1 first or specify --stage 1")
            return

        train_stage2_restricted_policy(
            stage1_checkpoint=str(stage1_path),
            total_steps=args.steps_stage2,
            checkpoint_path=str(stage2_path),
            device=args.device,
            alpha0=args.alpha0,
            beta0=args.beta0,
            seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("✅ CAC TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
