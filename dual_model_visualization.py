#!/usr/bin/env python3
"""
Dual Model Navigation Visualization
Demonstrates both safety-focused and goal-seeking navigation models working together
"""

import argparse
import os
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from cac.algorithms import CACAgent
from envs.planar_nav import PlanarNavEnv


class DualModelVisualizer:
    def __init__(self, safety_model_path, goal_model_path, seed=42):
        """Initialize the dual model visualizer"""
        self.seed = seed

        # Load both models
        print("🔒 Loading safety model...")
        self.safety_agent, _ = CACAgent.load(safety_model_path)

        print("🎯 Loading goal-seeking model...")
        self.goal_agent, _ = CACAgent.load(goal_model_path)

        # Create environments for both models
        self.safety_env = PlanarNavEnv(
            seed=seed,
            safety_margin=0.25,
            alpha=0.93,
            enhanced_obs=True,
            n_rays=19,
            fov_deg=130,
        )
        self.safety_env.set_stage("safety")

        self.goal_env = PlanarNavEnv(
            seed=seed,
            safety_margin=0.25,
            alpha=0.93,
            enhanced_obs=True,
            n_rays=19,
            fov_deg=130,
        )
        self.goal_env.set_stage("goal")

        # Visualization setup
        self.fig = None
        self.axes = None
        self.setup_figure()

    def setup_figure(self):
        """Setup the matplotlib figure with side-by-side comparison"""
        self.fig = plt.figure(figsize=(16, 8))

        # Create a grid layout
        gs = gridspec.GridSpec(
            2, 3, figure=self.fig, height_ratios=[4, 1], width_ratios=[1, 1, 1]
        )

        # Main navigation plots
        self.ax_safety = self.fig.add_subplot(gs[0, 0])
        self.ax_goal = self.fig.add_subplot(gs[0, 1])
        self.ax_comparison = self.fig.add_subplot(gs[0, 2])

        # Metrics plots
        self.ax_metrics = self.fig.add_subplot(gs[1, :])

        # Set titles
        self.ax_safety.set_title(
            "🔒 Safety-Focused Model\n(Collision Avoidance Priority)",
            fontsize=12,
            fontweight="bold",
        )
        self.ax_goal.set_title(
            "🎯 Goal-Seeking Model\n(Navigation + Safety)",
            fontsize=12,
            fontweight="bold",
        )
        self.ax_comparison.set_title(
            "📊 Live Comparison", fontsize=12, fontweight="bold"
        )
        self.ax_metrics.set_title("📈 Performance Metrics", fontsize=10)

        plt.tight_layout()

    def draw_environment(self, ax, env, title_suffix=""):
        """Draw the environment with obstacles, robot, and goal"""
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(-env.L * 0.5, env.L * 0.5)
        ax.set_ylim(-env.L * 0.5, env.L * 0.5)

        # Draw world boundaries
        boundary = patches.Rectangle(
            (-env.L * 0.5, -env.L * 0.5),
            env.L,
            env.L,
            linewidth=2,
            edgecolor="black",
            facecolor="lightgray",
            alpha=0.3,
        )
        ax.add_patch(boundary)

        # Draw obstacles
        for cx, cy, half_size in env.obstacles:
            obstacle = patches.Rectangle(
                (cx - half_size, cy - half_size),
                2 * half_size,
                2 * half_size,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.7,
            )
            ax.add_patch(obstacle)

        # Draw safety margin around obstacles (for visualization)
        for cx, cy, half_size in env.obstacles:
            safety_zone = patches.Rectangle(
                (cx - half_size - env.margin, cy - half_size - env.margin),
                2 * (half_size + env.margin),
                2 * (half_size + env.margin),
                facecolor="none",
                edgecolor="orange",
                alpha=0.3,
                linestyle="--",
            )
            ax.add_patch(safety_zone)

        # Draw goal
        goal_circle = patches.Circle(
            env.goal, 1.5, facecolor="gold", edgecolor="orange", alpha=0.8
        )
        ax.add_patch(goal_circle)
        ax.plot(
            env.goal[0],
            env.goal[1],
            "yo",
            markersize=15,
            markeredgecolor="orange",
            markeredgewidth=2,
        )

        # Draw robot
        robot_circle = patches.Circle(
            env.p, env.R, facecolor="lightblue", edgecolor="blue", alpha=0.8
        )
        ax.add_patch(robot_circle)

        # Draw robot heading
        heading_length = 0.4
        head_x = env.p[0] + heading_length * np.cos(env.theta)
        head_y = env.p[1] + heading_length * np.sin(env.theta)
        ax.arrow(
            env.p[0],
            env.p[1],
            head_x - env.p[0],
            head_y - env.p[1],
            head_width=0.1,
            head_length=0.1,
            fc="blue",
            ec="blue",
        )

        # Draw safety circle around robot
        safety_circle = patches.Circle(
            env.p,
            env.R + env.margin,
            facecolor="none",
            edgecolor="cyan",
            alpha=0.5,
            linestyle=":",
        )
        ax.add_patch(safety_circle)

        ax.grid(True, alpha=0.3)

    def draw_path(self, ax, path, color="blue", alpha=0.6):
        """Draw the robot's path"""
        if len(path) > 1:
            path_array = np.array(path)
            ax.plot(
                path_array[:, 0],
                path_array[:, 1],
                color=color,
                alpha=alpha,
                linewidth=2,
            )

    def run_dual_episode(self, max_steps=200):
        """Run both models simultaneously and compare their performance"""

        # Reset both environments with the same seed for fair comparison
        safety_obs, _ = self.safety_env.reset(seed=self.seed)
        goal_obs, _ = self.goal_env.reset(seed=self.seed)

        # Ensure both environments have identical setup
        self.goal_env.obstacles = self.safety_env.obstacles.copy()
        self.goal_env.start = self.safety_env.start.copy()
        self.goal_env.goal = self.safety_env.goal.copy()
        self.goal_env.p = self.safety_env.p.copy()
        self.goal_env.theta = self.safety_env.theta

        # Initialize tracking variables
        safety_path = [self.safety_env.p.copy()]
        goal_path = [self.goal_env.p.copy()]

        safety_clearances = []
        goal_clearances = []
        safety_goal_distances = []
        goal_goal_distances = []

        safety_done = False
        goal_done = False
        step = 0

        print(f"🚀 Starting dual navigation episode...")
        print(f"Start: ({self.safety_env.p[0]:.2f}, {self.safety_env.p[1]:.2f})")
        print(f"Goal:  ({self.safety_env.goal[0]:.2f}, {self.safety_env.goal[1]:.2f})")

        while step < max_steps and (not safety_done or not goal_done):
            step += 1

            # Safety model step
            if not safety_done:
                safety_action = self.safety_agent.act(safety_obs, deterministic=True)
                safety_obs, safety_reward, safety_term, safety_trunc, safety_info = (
                    self.safety_env.step(safety_action)
                )
                safety_done = safety_term or safety_trunc
                safety_path.append(self.safety_env.p.copy())

                # Track metrics
                safety_clearance = self._get_clearance(self.safety_env)
                safety_clearances.append(safety_clearance)
                safety_goal_dist = np.linalg.norm(
                    self.safety_env.goal - self.safety_env.p
                )
                safety_goal_distances.append(safety_goal_dist)

            # Goal model step
            if not goal_done:
                goal_action = self.goal_agent.act(goal_obs, deterministic=True)
                goal_obs, goal_reward, goal_term, goal_trunc, goal_info = (
                    self.goal_env.step(goal_action)
                )
                goal_done = goal_term or goal_trunc
                goal_path.append(self.goal_env.p.copy())

                # Track metrics
                goal_clearance = self._get_clearance(self.goal_env)
                goal_clearances.append(goal_clearance)
                goal_goal_dist = np.linalg.norm(self.goal_env.goal - self.goal_env.p)
                goal_goal_distances.append(goal_goal_dist)

            # Live visualization every 5 steps
            if step % 5 == 0:
                self.update_visualization(
                    safety_path,
                    goal_path,
                    safety_clearances,
                    goal_clearances,
                    safety_goal_distances,
                    goal_goal_distances,
                    safety_info if not safety_done else {"termination": "running"},
                    goal_info if not goal_done else {"termination": "running"},
                )
                plt.pause(0.1)

        # Final results
        safety_result = self._classify_outcome(
            self.safety_env, safety_info if "safety_info" in locals() else {}
        )
        goal_result = self._classify_outcome(
            self.goal_env, goal_info if "goal_info" in locals() else {}
        )

        print(f"\n🏁 Episode Complete after {step} steps")
        print(
            f"🔒 Safety Model: {safety_result['status']} (Clearance: {np.mean(safety_clearances):.3f})"
        )
        print(
            f"🎯 Goal Model: {goal_result['status']} (Clearance: {np.mean(goal_clearances):.3f})"
        )

        return {
            "safety_path": safety_path,
            "goal_path": goal_path,
            "safety_result": safety_result,
            "goal_result": goal_result,
            "safety_clearances": safety_clearances,
            "goal_clearances": goal_clearances,
            "steps": step,
        }

    def _get_clearance(self, env):
        """Get minimum clearance for the robot"""
        return env._min_range(env.p, env.theta) - env.R

    def _classify_outcome(self, env, info):
        """Classify the episode outcome"""
        goal_dist = np.linalg.norm(env.goal - env.p)
        termination = info.get("termination", "timeout")

        if goal_dist < 1.5:
            return {"status": "🎯 SUCCESS", "collision": False, "success": True}
        elif termination == "collision":
            return {"status": "💥 COLLISION", "collision": True, "success": False}
        else:
            return {"status": "⏰ TIMEOUT", "collision": False, "success": False}

    def update_visualization(
        self,
        safety_path,
        goal_path,
        safety_clearances,
        goal_clearances,
        safety_distances,
        goal_distances,
        safety_info,
        goal_info,
    ):
        """Update the live visualization"""

        # Clear and redraw environments
        self.draw_environment(self.ax_safety, self.safety_env)
        self.draw_environment(self.ax_goal, self.goal_env)

        # Draw paths
        self.draw_path(self.ax_safety, safety_path, "blue", alpha=0.8)
        self.draw_path(self.ax_goal, goal_path, "green", alpha=0.8)

        # Add status text
        safety_status = safety_info.get("termination", "running")
        goal_status = goal_info.get("termination", "running")

        self.ax_safety.text(
            -4.5,
            -4.0,
            f"Status: {safety_status}",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        )
        self.ax_goal.text(
            -4.5,
            -4.0,
            f"Status: {goal_status}",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
        )

        # Comparison plot
        self.ax_comparison.clear()
        self.ax_comparison.set_title("📊 Model Comparison")

        if len(safety_clearances) > 0 and len(goal_clearances) > 0:
            steps = range(len(safety_clearances))
            self.ax_comparison.plot(
                steps, safety_clearances, "b-", label="Safety Model", linewidth=2
            )

            goal_steps = range(len(goal_clearances))
            self.ax_comparison.plot(
                goal_steps, goal_clearances, "g-", label="Goal Model", linewidth=2
            )

            # Add safety threshold line
            safety_threshold = self.safety_env.margin * 0.67
            self.ax_comparison.axhline(
                y=safety_threshold,
                color="red",
                linestyle="--",
                label=f"Collision Threshold ({safety_threshold:.2f}m)",
            )

            self.ax_comparison.set_ylabel("Clearance (m)")
            self.ax_comparison.set_xlabel("Steps")
            self.ax_comparison.legend()
            self.ax_comparison.grid(True, alpha=0.3)

        # Metrics plot
        self.ax_metrics.clear()
        if len(safety_distances) > 0 and len(goal_distances) > 0:
            steps = range(len(safety_distances))
            self.ax_metrics.plot(
                steps,
                safety_distances,
                "b-",
                label="Safety Model Goal Distance",
                alpha=0.7,
            )

            goal_steps = range(len(goal_distances))
            self.ax_metrics.plot(
                goal_steps,
                goal_distances,
                "g-",
                label="Goal Model Goal Distance",
                alpha=0.7,
            )

            self.ax_metrics.axhline(
                y=1.5, color="gold", linestyle=":", label="Success Threshold"
            )
            self.ax_metrics.set_ylabel("Goal Distance (m)")
            self.ax_metrics.set_xlabel("Steps")
            self.ax_metrics.legend()
            self.ax_metrics.grid(True, alpha=0.3)

        plt.tight_layout()

    def run_comparison(self, episodes=3):
        """Run multiple episodes comparing both models"""

        results = []

        for ep in range(episodes):
            print(f"\n{'=' * 60}")
            print(f"🎮 Episode {ep + 1}/{episodes}")
            print(f"{'=' * 60}")

            # Use different seeds for variety
            self.seed = 42 + ep * 10

            result = self.run_dual_episode()
            results.append(result)

            # Wait for user input between episodes
            if ep < episodes - 1:
                input(f"\nPress Enter to continue to Episode {ep + 2}...")

        # Final summary
        self.print_summary(results)

    def print_summary(self, results):
        """Print summary of all episodes"""
        print(f"\n{'=' * 80}")
        print("🏆 FINAL SUMMARY - SAFETY vs GOAL MODELS")
        print(f"{'=' * 80}")

        safety_successes = sum(1 for r in results if r["safety_result"]["success"])
        safety_collisions = sum(1 for r in results if r["safety_result"]["collision"])

        goal_successes = sum(1 for r in results if r["goal_result"]["success"])
        goal_collisions = sum(1 for r in results if r["goal_result"]["collision"])

        n_episodes = len(results)

        print(f"📊 Results over {n_episodes} episodes:")
        print(f"")
        print(f"🔒 Safety Model:")
        print(f"   Success Rate:    {safety_successes / n_episodes * 100:6.1f}%")
        print(f"   Collision Rate:  {safety_collisions / n_episodes * 100:6.1f}%")
        print(
            f"   Avg Clearance:   {np.mean([np.mean(r['safety_clearances']) for r in results]):6.3f}m"
        )
        print(f"")
        print(f"🎯 Goal Model:")
        print(f"   Success Rate:    {goal_successes / n_episodes * 100:6.1f}%")
        print(f"   Collision Rate:  {goal_collisions / n_episodes * 100:6.1f}%")
        print(
            f"   Avg Clearance:   {np.mean([np.mean(r['goal_clearances']) for r in results]):6.3f}m"
        )
        print(f"")
        print(f"🎯 Key Insights:")

        if safety_collisions < goal_collisions:
            print(
                f"   • Safety model has {goal_collisions - safety_collisions} fewer collision(s)"
            )
        if goal_successes > safety_successes:
            print(
                f"   • Goal model has {goal_successes - safety_successes} more success(es)"
            )
        if safety_successes == 0 and goal_successes == 0:
            print(f"   • Both models need further training for this environment")

        print(f"\n🎉 Both models demonstrate learned obstacle avoidance behavior!")


def main():
    parser = argparse.ArgumentParser(description="Dual Model Navigation Visualization")
    parser.add_argument(
        "--safety_model", type=str, required=True, help="Path to safety-focused model"
    )
    parser.add_argument(
        "--goal_model", type=str, required=True, help="Path to goal-seeking model"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        visualizer = DualModelVisualizer(args.safety_model, args.goal_model, args.seed)
        visualizer.run_comparison(args.episodes)

        # Keep the final plot open
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
