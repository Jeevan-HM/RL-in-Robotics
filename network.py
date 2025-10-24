"""
network.py - Q-Network Implementation
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """
    Neural network to approximate Q-function.
    
    Input: state representation (one-hot encoded)
    Output: Q-values for all actions
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Initialize Q-network.
        
        Args:
            input_size: dimension of state representation
            hidden_sizes: list of hidden layer sizes
            output_size: number of actions
        """
        super(QNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(x)