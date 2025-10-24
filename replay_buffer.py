"""
replay_buffer.py - Experience Replay Buffer
"""

import random
from collections import deque
from typing import List, Tuple


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    Improves sample efficiency and breaks correlation between consecutive samples.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Tuple[int, int], action: int, reward: float, 
             next_state: Tuple[int, int], done: bool):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample random batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)