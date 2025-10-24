"""
environment.py - GridWorld Environment Implementation
"""

import random
from typing import Tuple


class GridWorld:
    """
    3x4 Gridworld environment with stochastic transitions.
    
    Grid layout (row, col):
    (0,0) (0,1) (0,2) (0,3)[+1]
    (1,0) (1,1) (1,2) (1,3)[-1]
    (2,0) (2,1) (2,2) (2,3)
    
    Terminal states: (0,3) with reward +1, (1,3) with reward -1
    Obstacle: (1,1) is blocked
    """
    
    def __init__(self):
        self.rows = 3
        self.cols = 4
        self.terminal_states = {(0, 3): 1.0, (1, 3): -1.0}
        self.obstacle = (1, 1)
        self.step_reward = -0.04
        
        # Action space: 0=North, 1=South, 2=West, 3=East
        self.actions = ['N', 'S', 'W', 'E']
        self.n_actions = len(self.actions)
        self.state = (2, 0)
        
    def reset(self, random_start: bool = False) -> Tuple[int, int]:
        """Reset environment to starting state."""
        if random_start:
            valid_states = [(r, c) for r in range(self.rows) 
                          for c in range(self.cols)
                          if (r, c) not in self.terminal_states 
                          and (r, c) != self.obstacle]
            self.state = random.choice(valid_states)
        else:
            self.state = (2, 0)
        return self.state
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal."""
        return state in self.terminal_states
    
    def is_valid(self, state: Tuple[int, int]) -> bool:
        """Check if state is within bounds and not an obstacle."""
        r, c = state
        return (0 <= r < self.rows and 0 <= c < self.cols 
                and state != self.obstacle)
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next state given current state and action."""
        r, c = state
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        next_state = (r + dr, c + dc)
        
        # If next state is invalid, stay in current state
        return next_state if self.is_valid(next_state) else state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action with stochastic transitions.
        80% desired direction, 10% left, 10% right.
        """
        if self.is_terminal(self.state):
            return self.state, 0, True
        
        # Stochastic action selection
        rand = random.random()
        if rand < 0.8:
            actual_action = action
        elif rand < 0.9:
            actual_action = (action - 1) % 4
        else:
            actual_action = (action + 1) % 4
        
        next_state = self.get_next_state(self.state, actual_action)
        
        # Determine reward
        if next_state in self.terminal_states:
            reward = self.terminal_states[next_state]
            done = True
        else:
            reward = self.step_reward
            done = False
        
        self.state = next_state
        return next_state, reward, done
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to single index."""
        return state[0] * self.cols + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert single index to (row, col) state."""
        return (index // self.cols, index % self.cols)