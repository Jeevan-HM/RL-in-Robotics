"""
agent.py - Deep Q-Network Agent
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from environment import GridWorld
from network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(self, env: GridWorld, config: Dict, device: torch.device):
        """
        Initialize DQN agent.
        
        Args:
            env: GridWorld environment
            config: hyperparameter configuration dictionary
            device: torch device (cuda or cpu)
        """
        self.env = env
        self.config = config
        self.device = device
        
        # State representation: one-hot encoding
        self.state_size = env.rows * env.cols
        self.action_size = env.n_actions
        
        # Q-network and target network
        hidden_sizes = config['hidden_sizes']
        self.q_network = QNetwork(self.state_size, hidden_sizes, self.action_size).to(device)
        self.target_network = QNetwork(self.state_size, hidden_sizes, self.action_size).to(device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config['buffer_capacity'])
        
        # Training parameters
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        
        # Tracking
        self.steps = 0
        self.episodes = 0
    
    def state_to_tensor(self, state: Tuple[int, int]) -> torch.Tensor:
        """Convert state to one-hot encoded tensor."""
        state_idx = self.env.state_to_index(state)
        one_hot = torch.zeros(1, self.state_size, device=self.device)
        one_hot[0, state_idx] = 1.0
        return one_hot
    
    def select_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step using a minibatch."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample minibatch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.cat([self.state_to_tensor(s) for s in states])
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.cat([self.state_to_tensor(s) for s in next_states])
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Compute current Q-values
        current_q = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q = self.target_network(next_state_batch).max(1)[0]
            target_q = reward_batch + self.gamma * next_q * (1 - done_batch)
        
        # Compute loss and optimize
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def generate_episode(self, max_steps: int = 100, training: bool = True) -> float:
        """Generate one episode and return total reward."""
        state = self.env.reset(random_start=training)
        total_reward = 0
        
        for step in range(max_steps):
            action = self.select_action(state, training=training)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            
            if training:
                self.replay_buffer.push(state, action, reward, next_state, done)
                self.train_step()
                
                self.steps += 1
                if self.steps % self.target_update_freq == 0:
                    self.update_target_network()
            
            state = next_state
            if done:
                break
        
        if training:
            self.episodes += 1
            self.decay_epsilon()
        
        return total_reward
    
    def evaluate(self, n_episodes: int = 100) -> Tuple[float, float]:
        """Evaluate current policy and return mean and std of rewards."""
        self.q_network.eval()
        rewards = [self.generate_episode(training=False) for _ in range(n_episodes)]
        self.q_network.train()
        return np.mean(rewards), np.std(rewards)