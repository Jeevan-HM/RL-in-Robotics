#!/usr/bin/env python3
"""
Quick test script to validate the improved collision avoidance system
"""

import numpy as np

from envs.planar_nav import PlanarNavEnv


def test_safety_improvements():
    print("Testing improved safety system...")

    # Create environment with new safety parameters
    env = PlanarNavEnv(seed=42)

    print(f"Robot radius: {env.R}")
    print(f"Safety margin: {env.margin}")
    print(f"Alpha (CBF): {env.alpha}")
    print(f"Observation space: {env.observation_space.shape}")

    # Test a simple scenario
    obs, _ = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial position: {env.p}")
    print(f"Goal position: {env.goal}")

    # Test CBF calculation
    h_val = env.h(env.p)
    print(f"Initial CBF value h(p): {h_val:.3f}")

    # Test minimum range calculation
    min_range = env._min_range(env.p, env.theta)
    clearance = min_range - env.R
    print(f"Initial min range: {min_range:.3f}")
    print(f"Initial clearance: {clearance:.3f}")

    # Simulate a few steps with random actions
    print("\nSimulating random steps:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        min_range = env._min_range(env.p, env.theta)
        clearance = min_range - env.R

        print(
            f"Step {i + 1}: action={action[0]:.3f}, reward={reward:.3f}, "
            f"clearance={clearance:.3f}, terminated={term}"
        )

        if "termination" in info:
            print(f"  Termination reason: {info['termination']}")

        if term or trunc:
            print("  Episode ended")
            break

    print("\n✅ Environment test completed successfully!")


def test_collision_thresholds():
    """Test the collision detection thresholds"""
    print("\n" + "=" * 50)
    print("TESTING COLLISION DETECTION THRESHOLDS")
    print("=" * 50)

    env = PlanarNavEnv(seed=123)

    # Test various clearance scenarios
    test_clearances = [0.5, 0.3, 0.25, 0.15, 0.1, 0.05, 0.0, -0.1]

    for clearance in test_clearances:
        # Simulate obstacle at exact clearance distance
        simulated_min_range = env.R + clearance

        # Check if this would trigger collision
        would_collide = clearance <= env.margin

        # Check safety reward calculation
        h_val = clearance - env.margin
        if h_val <= 0:
            safety_reward = float(np.exp(h_val))
        else:
            safety_reward = 1.0 + 0.1 * h_val

        # Add proximity penalty
        if clearance <= env.margin * 1.5:
            penalty = -5.0 * (1.5 * env.margin - clearance) / env.margin
            safety_reward += penalty

        print(
            f"Clearance: {clearance:6.3f} | Collision: {would_collide} | "
            f"Safety reward: {safety_reward:8.3f}"
        )

    print(f"\nCollision threshold: clearance <= {env.margin:.3f}")
    print(f"Warning zone: clearance <= {env.margin * 1.5:.3f}")


if __name__ == "__main__":
    test_safety_improvements()
    test_collision_thresholds()
