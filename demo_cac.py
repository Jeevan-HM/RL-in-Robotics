#!/usr/bin/env python3
"""
Visualize agent with MOVING obstacles
Demonstrates dynamic obstacle avoidance capabilities
"""

import sys

sys.path.insert(0, ".")

import time

import numpy as np
import torch

from device_config import DeviceConfig
from realistic_car_env import CarPhysicsParams, GoalOrientedCarEnv, NavigationConfig
from stage1 import SACAgent


def visualize_with_moving_obstacles(
    checkpoint_path: str = "checkpoints/responsive_stage2.pt",
    num_episodes: int = 3,
    num_moving_obstacles: int = 3,
    obstacle_speed: float = 3.0,  # m/s
):
    """
    Visualize agent dodging moving obstacles.

    Args:
        checkpoint_path: Path to trained model
        num_episodes: Number of episodes
        num_moving_obstacles: Number of moving obstacles
        obstacle_speed: Speed of moving obstacles (m/s)
    """

    print("=" * 70)
    print("DYNAMIC OBSTACLE AVOIDANCE DEMO")
    print("=" * 70)
    print(f"Loading: {checkpoint_path}")
    print(f"Moving obstacles: {num_moving_obstacles}")
    print(f"Obstacle speed: {obstacle_speed} m/s")
    print("=" * 70)
    print()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    device_config = DeviceConfig(device="auto", verbose=False)

    # Get expected dimensions
    expected_obs_dim = checkpoint["config"]["state_dim"]
    num_lidar_rays = expected_obs_dim - 10

    print(f"Agent observation dim: {expected_obs_dim}")
    print(f"LIDAR rays: {num_lidar_rays}")
    print()

    # Create car params
    car_params = CarPhysicsParams(
        mass=800.0,
        moment_inertia=1000.0,
        max_steering_angle=50.0 * np.pi / 180,
        max_steering_rate=90.0 * np.pi / 180,
        max_engine_force=8000.0,
        max_brake_force=12000.0,
        max_speed=25.0,
        lidar_range=25.0,
        num_lidar_rays=num_lidar_rays,
        length=4.5,
        width=1.8,
        wheelbase=2.7,
        tire_friction=0.9,
        rolling_resistance=0.01,
        drag_coefficient=0.25,
        frontal_area=2.0,
        air_density=1.225,
        safety_radius=2.5,
        collision_radius=2.0,
        lidar_fov=270.0 * np.pi / 180,
    )

    # Environment with MOVING obstacles
    env_cfg = NavigationConfig(
        world_width=60.0,
        world_height=60.0,
        start_pos=(10.0, 10.0),
        goal_pos=(50.0, 50.0),
        goal_radius=6.0,
        # Fewer static obstacles
        n_static_obstacles=2,
        obstacle_min_radius=1.5,
        obstacle_max_radius=2.5,
        # MOVING OBSTACLES!
        n_dynamic_obstacles=num_moving_obstacles,
        dynamic_obstacle_max_speed=obstacle_speed,
        dt=0.1,
        max_steps=3000,
        use_walls=True,
        sensor_in_obs=True,
        num_rays=num_lidar_rays,
        n_obstacles=2 + num_moving_obstacles,
    )

    env = GoalOrientedCarEnv(
        car_params=car_params, env_config=env_cfg, render_mode="human"
    )

    # Create and load agent
    agent = SACAgent(
        state_dim=expected_obs_dim,
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

    print("✓ Agent loaded")

    # Verify dimensions
    test_obs, _ = env.reset(seed=0)
    if test_obs.shape[0] != expected_obs_dim:
        print(f"❌ Dimension mismatch: {test_obs.shape[0]} vs {expected_obs_dim}")
        env.close()
        return

    print("✓ Dimensions match")
    print()

    # Statistics
    stats = {
        "goals": 0,
        "collisions": 0,
        "timeouts": 0,
        "min_distances": [],
        "close_calls": [],  # Times agent got within 3m of moving obstacle
    }

    # Run episodes
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep + 200)
        episode_length = 0
        min_dist_to_goal = float("inf")
        close_calls = 0

        print(f"\n{'=' * 70}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"Start: {env.pos}")
        print(f"Goal:  {env.goal_pos}")

        # Show obstacle info
        print(f"\nObstacles:")
        for i, obs_obj in enumerate(env.obstacles):
            if obs_obj.is_dynamic:
                print(
                    f"  Moving #{i}: pos={obs_obj.pos}, vel={obs_obj.velocity}, "
                    f"speed={np.linalg.norm(obs_obj.velocity):.1f} m/s"
                )
            else:
                print(f"  Static #{i}: pos={obs_obj.pos}, radius={obs_obj.radius:.1f}m")
        print()

        while True:
            # Render
            env.render()
            time.sleep(0.03)  # Faster rendering for dynamic obstacles

            # Check distance to moving obstacles
            for obs_obj in env.obstacles:
                if obs_obj.is_dynamic:
                    dist_to_moving = np.linalg.norm(env.pos - obs_obj.pos)
                    if dist_to_moving < 5.0:  # Within 5m
                        close_calls += 1

            # Agent action
            action = agent.select_action(obs, eval_mode=True)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1

            dist = info.get("distance_to_goal", 999)
            min_dist_to_goal = min(min_dist_to_goal, dist)

            if terminated or truncated:
                break

        # Episode result
        reason = info.get("terminated_reason", "timeout")

        if reason == "goal":
            stats["goals"] += 1
            result_emoji = "🎯"
        elif reason == "collision":
            stats["collisions"] += 1
            result_emoji = "💥"
        else:
            stats["timeouts"] += 1
            result_emoji = "⏱️"

        stats["min_distances"].append(min_dist_to_goal)
        stats["close_calls"].append(close_calls)

        print(f"\n{result_emoji} Result: {reason}")
        print(f"   Steps: {episode_length}")
        print(f"   Min distance to goal: {min_dist_to_goal:.2f}m")
        print(f"   Close calls (within 5m of moving obstacle): {close_calls}")

    env.close()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY - DYNAMIC OBSTACLE AVOIDANCE")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Moving obstacles: {num_moving_obstacles} @ {obstacle_speed} m/s")
    print()
    print(f"Results:")
    print(
        f"  🎯 Goals reached: {stats['goals']}/{num_episodes} "
        f"({100 * stats['goals'] / num_episodes:.1f}%)"
    )
    print(
        f"  💥 Collisions: {stats['collisions']}/{num_episodes} "
        f"({100 * stats['collisions'] / num_episodes:.1f}%)"
    )
    print(
        f"  ⏱️  Timeouts: {stats['timeouts']}/{num_episodes} "
        f"({100 * stats['timeouts'] / num_episodes:.1f}%)"
    )
    print()
    print(f"Navigation:")
    print(f"  Closest approach: {min(stats['min_distances']):.2f}m")
    print(f"  Avg min distance: {np.mean(stats['min_distances']):.2f}m")
    print()
    print(f"Dynamic Avoidance:")
    print(f"  Total close calls: {sum(stats['close_calls'])}")
    print(f"  Avg close calls per episode: {np.mean(stats['close_calls']):.1f}")
    print()

    if stats["collisions"] == 0:
        print("✅ Perfect avoidance! No collisions with moving obstacles!")
    elif stats["goals"] > 0:
        print("✅ Agent successfully navigated around moving obstacles!")
    else:
        print("⚠️  Agent struggled with moving obstacles - may need more training")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate dynamic obstacle avoidance"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/responsive_stage2.pt",
        help="Path to checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument(
        "--moving-obstacles", type=int, default=3, help="Number of moving obstacles"
    )
    parser.add_argument(
        "--obstacle-speed",
        type=float,
        default=3.0,
        help="Speed of moving obstacles (m/s)",
    )

    args = parser.parse_args()

    print("\n🚗 Dynamic Obstacle Avoidance Demo")
    print("Watch the agent dodge moving obstacles in real-time!\n")

    visualize_with_moving_obstacles(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        num_moving_obstacles=args.moving_obstacles,
        obstacle_speed=args.obstacle_speed,
    )
