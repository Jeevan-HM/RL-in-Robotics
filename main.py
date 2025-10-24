"""
main.py - Main training script for Gridworld DQN

Run with: python main.py
"""

import random
import numpy as np
import torch
from typing import Dict

from environment import GridWorld
from agent import DQNAgent
from config import get_default_config, get_fast_config, get_deep_config
from utils import plot_results, print_training_progress


# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_agent(agent: DQNAgent, n_episodes: int, eval_freq: int = 100) -> Dict:
    """
    Train DQN agent.
    
    Args:
        agent: DQN agent
        n_episodes: number of training episodes
        eval_freq: frequency of evaluation (in episodes)
        
    Returns:
        training history dictionary
    """
    print("\nStarting training...")
    print(f"Episodes: {n_episodes}")
    print(f"Configuration: {agent.config}\n")
    
    history = {
        'training_rewards': [],
        'eval_rewards': [],
        'eval_stds': [],
        'eval_episodes': [],
        'epsilon_values': []
    }
    
    for episode in range(n_episodes):
        # Training episode
        reward = agent.generate_episode(training=True)
        history['training_rewards'].append(reward)
        history['epsilon_values'].append(agent.epsilon)
        
        # Periodic evaluation
        if (episode + 1) % eval_freq == 0:
            eval_mean, eval_std = agent.evaluate(n_episodes=100)
            history['eval_rewards'].append(eval_mean)
            history['eval_stds'].append(eval_std)
            history['eval_episodes'].append(episode + 1)
            
            print_training_progress(episode + 1, n_episodes, reward, 
                                   eval_mean, eval_std, agent.epsilon)
    
    print("\nTraining completed!")
    return history


def main():
    """Main training loop."""
    
    # Choose configuration
    config = get_default_config()
    # config = get_fast_config()      # Uncomment for faster training
    # config = get_deep_config()      # Uncomment for deeper network
    
    # Create environment and agent
    env = GridWorld()
    agent = DQNAgent(env, config, device)
    
    # Training parameters
    n_episodes = 2000
    eval_freq = 100
    
    # Train agent
    history = train_agent(agent, n_episodes, eval_freq)
    
    # Plot results
    plot_results(history, config)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation (1000 episodes)")
    print("="*60)
    final_mean, final_std = agent.evaluate(n_episodes=1000)
    print(f"Mean Reward: {final_mean:.4f} ± {final_std:.4f}")
    
    # Save model
    model_path = 'gridworld_q_network.pth'
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"\nModel saved to '{model_path}'")


if __name__ == "__main__":
    main()