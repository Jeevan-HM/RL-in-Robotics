"""
experiments.py - Run hyperparameter experiments to compare performance

This script runs multiple experiments with different hyperparameters
and generates comparison plots for your assignment report.
"""

import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import DQNAgent
from config import get_default_config
from environment import GridWorld
from utils import print_training_progress

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def run_experiment(
    config: Dict, name: str, n_episodes: int = 1000, eval_freq: int = 100
) -> Dict:
    """
    Run single experiment with given configuration.

    Args:
        config: hyperparameter configuration
        name: experiment name
        n_episodes: number of training episodes
        eval_freq: evaluation frequency

    Returns:
        training history
    """
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {name}")
    print(f"{'=' * 60}")

    env = GridWorld()
    agent = DQNAgent(env, config, device)

    history = {
        "training_rewards": [],
        "eval_rewards": [],
        "eval_stds": [],
        "eval_episodes": [],
        "epsilon_values": [],
        "name": name,
    }

    for episode in range(n_episodes):
        reward = agent.generate_episode(training=True)
        history["training_rewards"].append(reward)
        history["epsilon_values"].append(agent.epsilon)

        if (episode + 1) % eval_freq == 0:
            eval_mean, eval_std = agent.evaluate(n_episodes=50)
            history["eval_rewards"].append(eval_mean)
            history["eval_stds"].append(eval_std)
            history["eval_episodes"].append(episode + 1)

            print_training_progress(
                episode + 1, n_episodes, reward, eval_mean, eval_std, agent.epsilon
            )

    return history


def plot_comparison(
    histories: List[Dict],
    title: str,
    save_path: str,
    base_fontsize: int = 16,  # New parameter to control all font sizes
):
    """
    Plot comparison of multiple experiments.

    Args:
        histories: list of history dictionaries
        title: plot title
        save_path: path to save plot
        base_fontsize: base font size for plot elements
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Define relative font sizes ---
    label_size = base_fontsize
    title_size = base_fontsize + 2
    suptitle_size = base_fontsize + 4
    legend_size = base_fontsize - 2
    tick_size = base_fontsize - 2
    # ----------------------------------

    # Training rewards comparison
    ax = axes[0]
    for history in histories:
        window = 50
        rewards = history["training_rewards"]
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(rewards)),
            smoothed,
            label=history["name"],
            linewidth=2,
        )
    ax.set_xlabel("Episode", fontsize=label_size)
    ax.set_ylabel("Training Reward (smoothed)", fontsize=label_size)
    ax.set_title("Training Performance Comparison", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid(True, alpha=0.3)
    ax.tick_params(
        axis="both", which="major", labelsize=tick_size
    )  # Set tick font size

    # Evaluation rewards comparison
    ax = axes[1]
    for history in histories:
        ax.plot(
            history["eval_episodes"],
            history["eval_rewards"],
            marker="o",
            label=history["name"],
            linewidth=2,
        )
    ax.set_xlabel("Episode", fontsize=label_size)
    ax.set_ylabel("Evaluation Reward", fontsize=label_size)
    ax.set_title("Evaluation Performance Comparison", fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.grid(True, alpha=0.3)
    ax.tick_params(
        axis="both", which="major", labelsize=tick_size
    )  # Set tick font size

    plt.suptitle(title, fontsize=suptitle_size, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to '{save_path}'")
    plt.show()


def experiment_learning_rates():
    """Experiment with different learning rates."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Learning Rate Comparison")
    print("=" * 60)

    learning_rates = [0.0001, 0.001, 0.01]
    histories = []

    for lr in learning_rates:
        config = get_default_config()
        config["learning_rate"] = lr
        history = run_experiment(config, f"LR={lr}", n_episodes=1000)
        histories.append(history)

    plot_comparison(
        histories,
        "Effect of Learning Rate on Performance",
        "Report/images/experiment_learning_rate.png",
    )


def experiment_network_architectures():
    """Experiment with different network architectures."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Network Architecture Comparison")
    print("=" * 60)

    architectures = [
        ([32], "1 Layer (32)"),
        ([64, 64], "2 Layers (64,64)"),
        ([128, 128], "2 Layers (128,128)"),
        ([64, 64, 64], "3 Layers (64,64,64)"),
    ]
    histories = []

    for arch, name in architectures:
        config = get_default_config()
        config["hidden_sizes"] = arch
        history = run_experiment(config, name, n_episodes=1000)
        histories.append(history)

    plot_comparison(
        histories,
        "Effect of Network Architecture on Performance",
        "Report/images/experiment_architecture.png",
    )


def experiment_epsilon_decay():
    """Experiment with different epsilon decay rates."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Epsilon Decay Comparison")
    print("=" * 60)

    decay_rates = [0.99, 0.995, 0.999]
    histories = []

    for decay in decay_rates:
        config = get_default_config()
        config["epsilon_decay"] = decay
        history = run_experiment(config, f"Decay={decay}", n_episodes=1000)
        histories.append(history)

    plot_comparison(
        histories,
        "Effect of Epsilon Decay Rate on Performance",
        "Report/images/experiment_epsilon.png",
    )


def experiment_batch_sizes():
    """Experiment with different batch sizes."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Batch Size Comparison")
    print("=" * 60)

    batch_sizes = [16, 32, 64, 128]
    histories = []

    for batch_size in batch_sizes:
        config = get_default_config()
        config["batch_size"] = batch_size
        history = run_experiment(config, f"Batch={batch_size}", n_episodes=1000)
        histories.append(history)

    plot_comparison(
        histories,
        "Effect of Batch Size on Performance",
        "Report/images/experiment_batch_size.png",
    )


def experiment_discount_factors():
    """Experiment with different discount factors."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Discount Factor (Gamma) Comparison")
    print("=" * 60)

    gammas = [0.9, 0.95, 0.99, 0.999]
    histories = []

    for gamma in gammas:
        config = get_default_config()
        config["gamma"] = gamma
        history = run_experiment(config, f"γ={gamma}", n_episodes=1000)
        histories.append(history)

    plot_comparison(
        histories,
        "Effect of Discount Factor on Performance",
        "Report/images/experiment_gamma.png",
    )


def main():
    """Run all experiments."""
    print("=" * 60)
    print("GRIDWORLD DQN HYPERPARAMETER EXPERIMENTS")
    print("=" * 60)
    print("\nThis will run multiple experiments to compare hyperparameters.")
    print("Each experiment takes several minutes to complete.\n")

    # Choose which experiments to run
    experiments = [
        ("Learning Rates", experiment_learning_rates),
        ("Network Architectures", experiment_network_architectures),
        ("Epsilon Decay", experiment_epsilon_decay),
        ("Batch Sizes", experiment_batch_sizes),
        ("Discount Factors", experiment_discount_factors),
    ]

    print("Available experiments:")
    for i, (name, _) in enumerate(experiments, 1):
        print(f"{i}. {name}")
    print(f"{len(experiments) + 1}. Run all experiments")

    choice = input(f"\nSelect experiment (1-{len(experiments) + 1}) or 'all': ")

    if choice.lower() == "all" or choice == str(len(experiments) + 1):
        for name, exp_func in experiments:
            exp_func()
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                experiments[idx][1]()
            else:
                print("Invalid choice!")
        except ValueError:
            print("Invalid input!")

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
