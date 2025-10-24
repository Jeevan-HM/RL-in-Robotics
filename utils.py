"""
utils.py - Visualization and utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_results(history: Dict, config: Dict, save_path: str = 'images/gridworld_dqn_results.png'):
    """
    Plot training and evaluation results.
    
    Args:
        history: training history dictionary
        config: hyperparameter configuration
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training rewards with smoothing
    ax = axes[0, 0]
    window = 50
    training_rewards = history['training_rewards']
    smoothed = np.convolve(training_rewards, np.ones(window)/window, mode='valid')
    ax.plot(training_rewards, alpha=0.3, label='Raw', linewidth=0.5)
    ax.plot(range(window-1, len(training_rewards)), smoothed, 
            label=f'{window}-episode MA', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Training Reward')
    ax.set_title('Training Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evaluation rewards with error bars
    ax = axes[0, 1]
    eval_eps = history['eval_episodes']
    eval_rewards = history['eval_rewards']
    eval_stds = history['eval_stds']
    ax.plot(eval_eps, eval_rewards, marker='o', label='Mean', linewidth=2)
    ax.fill_between(eval_eps, 
                     np.array(eval_rewards) - np.array(eval_stds),
                     np.array(eval_rewards) + np.array(eval_stds),
                     alpha=0.3, label='±1 Std')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Evaluation Reward')
    ax.set_title('Evaluation Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax = axes[1, 0]
    ax.plot(history['epsilon_values'], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate Decay')
    ax.grid(True, alpha=0.3)
    
    # Hyperparameters
    ax = axes[1, 1]
    ax.axis('off')
    params_text = "Hyperparameters:\n\n"
    for key, value in config.items():
        params_text += f"{key}: {value}\n"
    ax.text(0.1, 0.9, params_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to '{save_path}'")
    plt.show()


def print_training_progress(episode: int, n_episodes: int, train_reward: float,
                          eval_mean: float, eval_std: float, epsilon: float):
    """Print formatted training progress."""
    print(f"Episode {episode:>4}/{n_episodes} | "
          f"Train: {train_reward:>6.3f} | "
          f"Eval: {eval_mean:>6.3f} ± {eval_std:>5.3f} | "
          f"ε: {epsilon:.3f}")