#!/usr/bin/env python3
"""
Simplified Dual Model Navigation Demo
Shows safety vs goal-seeking behavior in challenging scenarios
"""

import argparse

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from cac.algorithms import CACAgent
from envs.planar_nav import PlanarNavEnv


def run_model_comparison(safety_model_path, goal_model_path, seed=789):
    """Run a side-by-side comparison of both models"""

    print("🔄 Loading models...")
    safety_agent, _ = CACAgent.load(safety_model_path)
    goal_agent, _ = CACAgent.load(goal_model_path)

    # Create environments with more challenging setup
    safety_env = PlanarNavEnv(
        seed=seed,
        safety_margin=0.25,
        alpha=0.93,
        enhanced_obs=True,
        n_rays=19,
        fov_deg=130,
        N_obs=25,  # More obstacles for challenge
    )
    safety_env.set_stage("safety")

    goal_env = PlanarNavEnv(
        seed=seed,
        safety_margin=0.25,
        alpha=0.93,
        enhanced_obs=True,
        n_rays=19,
        fov_deg=130,
        N_obs=25,  # Same obstacles
    )
    goal_env.set_stage("goal")

    # Ensure identical environments
    goal_env.obstacles = safety_env.obstacles.copy()
    goal_env.start = safety_env.start.copy()
    goal_env.goal = safety_env.goal.copy()

    print(f"🎯 Navigation Challenge:")
    print(f"   Start: ({safety_env.start[0]:.2f}, {safety_env.start[1]:.2f})")
    print(f"   Goal:  ({safety_env.goal[0]:.2f}, {safety_env.goal[1]:.2f})")
    print(f"   Distance: {np.linalg.norm(safety_env.goal - safety_env.start):.2f}m")
    print(f"   Obstacles: {len(safety_env.obstacles)}")

    # Run both models
    safety_result = run_single_model(safety_env, safety_agent, "Safety")
    goal_result = run_single_model(goal_env, goal_agent, "Goal")

    # Visualize results
    visualize_comparison(safety_env, goal_env, safety_result, goal_result)

    # Print comparison
    print_comparison(safety_result, goal_result)


def run_single_model(env, agent, model_name, max_steps=300):
    """Run a single model and track its performance"""

    obs, _ = env.reset()
    path = [env.p.copy()]
    clearances = []
    goal_distances = []
    actions = []

    done = False
    step = 0

    print(f"🚀 Running {model_name} model...")

    while not done and step < max_steps:
        # Get action
        action = agent.act(obs, deterministic=True)
        actions.append(action[0])

        # Take step
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step += 1

        # Track metrics
        path.append(env.p.copy())
        clearance = env._min_range(env.p, env.theta) - env.R
        clearances.append(clearance)
        goal_dist = np.linalg.norm(env.goal - env.p)
        goal_distances.append(goal_dist)

        # Print progress for long episodes
        if step % 50 == 0:
            print(
                f"   Step {step}: clearance={clearance:.3f}, goal_dist={goal_dist:.3f}"
            )

    # Classify outcome
    final_goal_dist = np.linalg.norm(env.goal - env.p)
    termination = info.get("termination", "timeout")

    if final_goal_dist < 1.5:
        outcome = "SUCCESS"
        emoji = "🎯"
    elif termination == "collision":
        outcome = "COLLISION"
        emoji = "💥"
    else:
        outcome = "TIMEOUT"
        emoji = "⏰"

    result = {
        "path": np.array(path),
        "clearances": np.array(clearances),
        "goal_distances": np.array(goal_distances),
        "actions": np.array(actions),
        "steps": step,
        "outcome": outcome,
        "emoji": emoji,
        "final_goal_dist": final_goal_dist,
        "min_clearance": np.min(clearances) if clearances else 0,
        "avg_clearance": np.mean(clearances) if clearances else 0,
        "termination": termination,
    }

    print(f"   {emoji} {model_name} Result: {outcome} in {step} steps")
    print(f"      Final goal distance: {final_goal_dist:.3f}m")
    print(f"      Min clearance: {result['min_clearance']:.3f}m")

    return result


def visualize_comparison(safety_env, goal_env, safety_result, goal_result):
    """Create side-by-side visualization of both models"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Safety model visualization
    draw_environment(ax1, safety_env, "🔒 Safety Model Navigation")
    ax1.plot(
        safety_result["path"][:, 0],
        safety_result["path"][:, 1],
        "b-",
        linewidth=3,
        alpha=0.8,
        label="Safety Path",
    )
    ax1.text(
        -4.5,
        4.0,
        f"{safety_result['emoji']} {safety_result['outcome']}",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )

    # Goal model visualization
    draw_environment(ax2, goal_env, "🎯 Goal Model Navigation")
    ax2.plot(
        goal_result["path"][:, 0],
        goal_result["path"][:, 1],
        "g-",
        linewidth=3,
        alpha=0.8,
        label="Goal Path",
    )
    ax2.text(
        -4.5,
        4.0,
        f"{goal_result['emoji']} {goal_result['outcome']}",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )

    # Clearance comparison
    ax3.set_title("📏 Safety Clearance Over Time")
    if len(safety_result["clearances"]) > 0:
        ax3.plot(
            safety_result["clearances"],
            "b-",
            linewidth=2,
            label="Safety Model",
            alpha=0.8,
        )
    if len(goal_result["clearances"]) > 0:
        ax3.plot(
            goal_result["clearances"], "g-", linewidth=2, label="Goal Model", alpha=0.8
        )

    collision_threshold = safety_env.margin * 0.67
    ax3.axhline(
        y=collision_threshold,
        color="red",
        linestyle="--",
        label=f"Collision Threshold ({collision_threshold:.2f}m)",
    )
    ax3.axhline(
        y=safety_env.margin,
        color="orange",
        linestyle=":",
        label=f"Safety Margin ({safety_env.margin:.2f}m)",
    )
    ax3.set_ylabel("Clearance (m)")
    ax3.set_xlabel("Steps")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Goal distance comparison
    ax4.set_title("🎯 Goal Distance Over Time")
    if len(safety_result["goal_distances"]) > 0:
        ax4.plot(
            safety_result["goal_distances"],
            "b-",
            linewidth=2,
            label="Safety Model",
            alpha=0.8,
        )
    if len(goal_result["goal_distances"]) > 0:
        ax4.plot(
            goal_result["goal_distances"],
            "g-",
            linewidth=2,
            label="Goal Model",
            alpha=0.8,
        )

    ax4.axhline(y=1.5, color="gold", linestyle=":", label="Success Threshold (1.5m)")
    ax4.set_ylabel("Goal Distance (m)")
    ax4.set_xlabel("Steps")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def draw_environment(ax, env, title):
    """Draw the environment with obstacles, robot, and goal"""
    ax.clear()
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-env.L * 0.5, env.L * 0.5)
    ax.set_ylim(-env.L * 0.5, env.L * 0.5)

    # World boundaries
    boundary = patches.Rectangle(
        (-env.L * 0.5, -env.L * 0.5),
        env.L,
        env.L,
        linewidth=2,
        edgecolor="black",
        facecolor="lightgray",
        alpha=0.2,
    )
    ax.add_patch(boundary)

    # Obstacles
    for cx, cy, half_size in env.obstacles:
        obstacle = patches.Rectangle(
            (cx - half_size, cy - half_size),
            2 * half_size,
            2 * half_size,
            facecolor="red",
            edgecolor="darkred",
            alpha=0.8,
        )
        ax.add_patch(obstacle)

    # Goal
    goal_circle = patches.Circle(
        env.goal,
        1.5,
        facecolor="gold",
        edgecolor="orange",
        alpha=0.6,
        linestyle="--",
        linewidth=2,
    )
    ax.add_patch(goal_circle)
    ax.plot(
        env.goal[0],
        env.goal[1],
        "yo",
        markersize=12,
        markeredgecolor="orange",
        markeredgewidth=2,
    )

    # Start position
    ax.plot(
        env.start[0],
        env.start[1],
        "gs",
        markersize=10,
        markeredgecolor="darkgreen",
        markeredgewidth=2,
    )

    # Current robot position
    robot_circle = patches.Circle(
        env.p, env.R, facecolor="lightblue", edgecolor="blue", alpha=0.8
    )
    ax.add_patch(robot_circle)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")


def print_comparison(safety_result, goal_result):
    """Print detailed comparison of both models"""

    print(f"\n{'=' * 80}")
    print("📊 DETAILED PERFORMANCE COMPARISON")
    print(f"{'=' * 80}")

    print(f"{'Metric':<25} {'Safety Model':<20} {'Goal Model':<20} {'Winner'}")
    print(f"{'-' * 80}")

    # Success comparison
    safety_success = 1 if safety_result["outcome"] == "SUCCESS" else 0
    goal_success = 1 if goal_result["outcome"] == "SUCCESS" else 0
    success_winner = (
        "🎯 Goal"
        if goal_success > safety_success
        else "🔒 Safety"
        if safety_success > goal_success
        else "🤝 Tie"
    )
    print(f"{'Success':<25} {safety_success:<20} {goal_success:<20} {success_winner}")

    # Collision comparison
    safety_collision = 1 if safety_result["outcome"] == "COLLISION" else 0
    goal_collision = 1 if goal_result["outcome"] == "COLLISION" else 0
    collision_winner = (
        "🔒 Safety"
        if safety_collision < goal_collision
        else "🎯 Goal"
        if goal_collision < safety_collision
        else "🤝 Tie"
    )
    print(
        f"{'No Collision':<25} {1 - safety_collision:<20} {1 - goal_collision:<20} {collision_winner}"
    )

    # Steps comparison
    steps_winner = (
        "🎯 Goal"
        if goal_result["steps"] < safety_result["steps"]
        else "🔒 Safety"
        if safety_result["steps"] < goal_result["steps"]
        else "🤝 Tie"
    )
    print(
        f"{'Steps Taken':<25} {safety_result['steps']:<20} {goal_result['steps']:<20} {steps_winner}"
    )

    # Clearance comparison
    clearance_winner = (
        "🔒 Safety"
        if safety_result["avg_clearance"] > goal_result["avg_clearance"]
        else "🎯 Goal"
        if goal_result["avg_clearance"] > safety_result["avg_clearance"]
        else "🤝 Tie"
    )
    print(
        f"{'Avg Clearance (m)':<25} {safety_result['avg_clearance']:>8.3f}{'':<12} {goal_result['avg_clearance']:>8.3f}{'':<12} {clearance_winner}"
    )

    # Final goal distance
    goal_dist_winner = (
        "🎯 Goal"
        if goal_result["final_goal_dist"] < safety_result["final_goal_dist"]
        else "🔒 Safety"
        if safety_result["final_goal_dist"] < goal_result["final_goal_dist"]
        else "🤝 Tie"
    )
    print(
        f"{'Final Goal Dist (m)':<25} {safety_result['final_goal_dist']:>8.3f}{'':<12} {goal_result['final_goal_dist']:>8.3f}{'':<12} {goal_dist_winner}"
    )

    print(f"\n🎯 Key Insights:")
    if safety_result["outcome"] == "SUCCESS" and goal_result["outcome"] == "SUCCESS":
        print("   • Both models successfully reached the goal!")
        if safety_result["avg_clearance"] > goal_result["avg_clearance"]:
            print("   • Safety model maintained better clearance from obstacles")
        if goal_result["steps"] < safety_result["steps"]:
            print("   • Goal model reached the target more efficiently")
    elif safety_result["outcome"] == "SUCCESS" and goal_result["outcome"] != "SUCCESS":
        print("   • Safety model succeeded while goal model failed")
        print("   • This shows the importance of safety-first training")
    elif goal_result["outcome"] == "SUCCESS" and safety_result["outcome"] != "SUCCESS":
        print("   • Goal model succeeded while safety model failed")
        print("   • Goal-seeking behavior helped navigate to target")
    else:
        print("   • Both models faced challenges in this environment")
        print("   • More training or environment tuning may be needed")

    print("\n🎉 Both models demonstrate learned navigation capabilities!")


def main():
    parser = argparse.ArgumentParser(description="Dual Model Navigation Comparison")
    parser.add_argument(
        "--safety_model", type=str, required=True, help="Path to safety-focused model"
    )
    parser.add_argument(
        "--goal_model", type=str, required=True, help="Path to goal-seeking model"
    )
    parser.add_argument(
        "--seed", type=int, default=789, help="Random seed for environment"
    )

    args = parser.parse_args()

    try:
        run_model_comparison(args.safety_model, args.goal_model, args.seed)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
