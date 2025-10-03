import numpy as np
import random

class GridWorld:
    """
    GridWorld environment for the Q-learning problem.
    - Manages states, actions, rewards, and transitions.
    - The grid coordinates are (column, row), starting from (1, 1).
    """
    def __init__(self, penalty=-1.0):
        # Define grid dimensions and obstacle
        self.grid_size = (4, 3)
        self.wall_state = (2, 2)
        self.start_state = (1, 1)

        # Define terminal states and their rewards
        self.terminal_states = {(4, 3): 1.0, (4, 2): penalty}

        # Define reward for non-terminal moves
        self.move_reward = -0.04

        # Define actions as movements (dx, dy) and their corresponding names
        self.actions = {
            'N': (0, 1),   # North
            'S': (0, -1),  # South
            'E': (1, 0),   # East
            'W': (-1, 0)   # West
        }
        self.action_names = list(self.actions.keys())

    def get_states(self):
        """Returns a list of all valid states."""
        states = []
        for x in range(1, self.grid_size[0] + 1):
            for y in range(1, self.grid_size[1] + 1):
                if (x, y) != self.wall_state:
                    states.append((x, y))
        return states

    def step(self, state, action_name):
        """
        Performs a step in the environment given a state and action.
        - Returns the next state and the reward.
        - Implements the stochastic transition model. [cite: 13]
        """
        if state in self.terminal_states:
            return state, 0

        # Determine the possible outcomes based on the intended action
        outcomes = self._get_stochastic_outcomes(action_name)
        
        # Choose one outcome based on the probabilities (80%, 10%, 10%)
        chosen_action_name = np.random.choice(
            [outcomes['intent'], outcomes['left'], outcomes['right']],
            p=[0.8, 0.1, 0.1]
        )
        
        # Calculate the next position based on the chosen action
        action = self.actions[chosen_action_name]
        next_state = (state[0] + action[0], state[1] + action[1])

        # Check for collisions with walls or the boundary
        if (next_state == self.wall_state or 
            not (1 <= next_state[0] <= self.grid_size[0]) or
            not (1 <= next_state[1] <= self.grid_size[1])):
            next_state = state  # Agent stays in the same spot [cite: 15]

        # Get the reward for this transition
        reward = self.terminal_states.get(next_state, self.move_reward)
        
        return next_state, reward

    def _get_stochastic_outcomes(self, action_name):
        """Helper to determine the intended, left, and right actions."""
        if action_name == 'N':
            return {'intent': 'N', 'left': 'W', 'right': 'E'}
        if action_name == 'S':
            return {'intent': 'S', 'left': 'E', 'right': 'W'}
        if action_name == 'E':
            return {'intent': 'E', 'left': 'N', 'right': 'S'}
        if action_name == 'W':
            return {'intent': 'W', 'left': 'S', 'right': 'N'}

class QLearningAgent:
    """
    Q-Learning Agent that interacts with the GridWorld.
    - Manages the Q-table and the learning process.
    """
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.states = states
        self.actions = actions
        self.alpha = alpha      # Learning rate [cite: 7]
        self.gamma = gamma      # Discount factor [cite: 8]
        self.epsilon = epsilon  # Exploration rate [cite: 8]
        
        # Epsilon decay parameters
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        
        # Initialize Q-table with zeros
        self.q_table = {state: {action: 0.0 for action in self.actions} for state in self.states}

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            # Exploit: choose the best action from Q-table
            q_values = self.q_table[state]
            return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        """Updates the Q-value for a given state-action pair."""
        # Find the max Q-value for the next state
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update rule
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Decays the exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def print_policy(agent, env):
    """Prints the learned policy in a grid format."""
    policy_grid = [[' ' for _ in range(env.grid_size[0])] for _ in range(env.grid_size[1])]
    action_arrows = {'N': '↑', 'S': '↓', 'E': '→', 'W': '←'}

    for state in agent.states:
        x, y = state
        if state in env.terminal_states:
            policy_grid[env.grid_size[1] - y][x - 1] = str(env.terminal_states[state])
        elif state == env.wall_state:
            policy_grid[env.grid_size[1] - y][x - 1] = 'WALL'
        else:
            best_action = agent.choose_action(state)
            policy_grid[env.grid_size[1] - y][x - 1] = action_arrows[best_action]

    print("\nOptimal Policy:")
    for row in policy_grid:
        print("+---" * len(row) + "+")
        print("| " + " | ".join(row) + " |")
    print("+---" * len(row) + "+")

# --- Main Training Loop ---
if __name__ == "__main__":
    # --- Configuration ---
    # Set penalty to -1 [cite: 16] or -200 [cite: 10]
    PENALTY = -1.0 
    EPISODES = 50000
    
    # --- Initialization ---
    env = GridWorld(penalty=PENALTY)
    agent = QLearningAgent(
        states=env.get_states(),
        actions=env.action_names,
        alpha=0.1, 
        gamma=0.9, 
        epsilon=1.0
    )
    
    print(f"Starting training for {EPISODES} episodes with penalty = {PENALTY}...")

    # --- Training ---
    for episode in range(EPISODES):
        state = env.start_state
        done = False
        
        while not done:
            # Agent chooses an action based on the current state
            action = agent.choose_action(state)
            
            # Environment gives back the next state and reward
            next_state, reward = env.step(state, action)
            
            # Agent learns from the experience
            agent.learn(state, action, reward, next_state)
            
            # Move to the next state
            state = next_state
            
            # Check if the episode has ended
            if state in env.terminal_states:
                done = True

        # Decay epsilon after each episode to reduce exploration over time
        agent.decay_epsilon()
        
        if (episode + 1) % 10000 == 0:
            print(f"Episode {episode + 1}/{EPISODES} | Epsilon: {agent.epsilon:.4f}")

    print("\nTraining finished.")
    
    # --- Results ---
    # Set epsilon to 0 to ensure the printed policy is purely exploitative
    agent.epsilon = 0.0
    print_policy(agent, env)