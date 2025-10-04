"""
Q-Learning implementation for GridWorld environment.

This module implements a classic reinforcement learning problem where an agent
learns to navigate a 4x3 grid to reach a goal state while avoiding penalties.
The agent uses the Q-learning algorithm to learn an optimal policy through
trial and error interactions with the environment.

Key Components:
- GridWorld: The environment that defines states, actions, and rewards
- QLearningAgent: The learning agent that updates its Q-table based on experience
- Stochastic transitions: Actions have uncertainty (80% intended, 10% left, 10% right)

Author: Jeevan Hebbal Manjunath
Date: October 2024
"""

import random

import numpy as np


class GridWorld:
    """GridWorld environment for the Q-learning problem.

    Manages states, actions, rewards, and transitions in a 4x3 grid world.
    The grid coordinates are (column, row), starting from (1, 1).

    Grid Layout:
    +---+---+---+---+
    |   |   |   | +1| (4,3) - Goal state (+1 reward)
    +---+---+---+---+
    |   |XXX|   | -1| (4,2) - Terminal penalty state
    +---+---+---+---+
    | S |   |   |   | (1,1) - Start state
    +---+---+---+---+

    Where: S=Start, XXX=Wall, +1=Goal, -1=Penalty
    """

    def __init__(self, penalty=-1.0):
        """Initialize the GridWorld environment.

        Args:
            penalty (float): Negative reward for reaching the penalty state (4,2)
        """
        # Grid dimensions: 4 columns x 3 rows
        self.grid_size = (4, 3)

        # Obstacle/wall that blocks movement - agent cannot occupy this state
        self.wall_state = (2, 2)

        # Starting position for all episodes
        self.start_state = (1, 1)

        # Terminal states with their associated rewards
        # (4,3): Goal state with +1 reward
        # (4,2): Penalty state with customizable negative reward
        self.terminal_states = {(4, 3): 1.0, (4, 2): penalty}

        # Small negative reward for each step to encourage efficiency
        # This creates a "living penalty" that motivates the agent to find
        # the shortest path to the goal
        self.move_reward = -0.04

        # Action space: 4 possible movements with their coordinate changes
        # Format: "action_name": (delta_x, delta_y)
        self.actions = {
            "N": (0, 1),  # North: move up
            "S": (0, -1),  # South: move down
            "E": (1, 0),  # East: move right
            "W": (-1, 0),  # West: move left
        }
        self.action_names = list(self.actions.keys())

    def get_states(self):
        """Generate all valid states in the grid world.

        Iterates through all grid positions and excludes the wall state.
        This creates the state space for the Q-learning algorithm.

        Returns:
            list: All valid (x, y) coordinate pairs the agent can occupy
        """
        states = []
        # Iterate through all possible grid positions
        for x in range(1, self.grid_size[0] + 1):  # Columns 1-4
            for y in range(1, self.grid_size[1] + 1):  # Rows 1-3
                # Exclude the wall/obstacle state
                if (x, y) != self.wall_state:
                    states.append((x, y))
        return states

    def step(self, state, action_name):
        """Execute one step in the environment with stochastic transitions.

        Implements a realistic robotic movement model where actions are not
        always executed perfectly. The agent has an 80% chance of moving in
        the intended direction, and 10% chance each of moving left or right
        relative to the intended direction.

        Args:
            state (tuple): Current (x, y) position
            action_name (str): Intended action ("N", "S", "E", or "W")

        Returns:
            tuple: (next_state, reward) where:
                - next_state: New (x, y) position after movement
                - reward: Immediate reward for this transition
        """
        # Terminal states are absorbing - no further movement possible
        if state in self.terminal_states:
            return state, 0

        # Get the three possible action outcomes (intended, left turn, right turn)
        outcomes = self._get_stochastic_outcomes(action_name)

        # Stochastic action selection based on movement uncertainty
        # 80% intended direction, 10% left turn, 10% right turn
        # This models real-world robotics where actions have uncertainty
        rng = np.random.default_rng()
        chosen_action_name = rng.choice(
            [outcomes["intent"], outcomes["left"], outcomes["right"]],
            p=[0.8, 0.1, 0.1],
        )

        # Calculate the resulting position based on chosen action
        action = self.actions[chosen_action_name]
        next_state = (state[0] + action[0], state[1] + action[1])

        # Collision detection: check for walls and boundaries
        # If collision detected, agent remains in current position
        if (
            next_state == self.wall_state  # Hit the wall obstacle
            or not (1 <= next_state[0] <= self.grid_size[0])  # Out of bounds X
            or not (1 <= next_state[1] <= self.grid_size[1])  # Out of bounds Y
        ):
            next_state = state  # Stay in place due to collision

        # Determine reward based on the resulting state
        reward = self.terminal_states.get(next_state, self.move_reward)

        return next_state, reward

    def _get_stochastic_outcomes(self, action_name):
        """Determine possible action outcomes for stochastic transitions.

        For each intended action, calculates what the "left" and "right"
        relative actions would be. This models the uncertainty in robotic
        movement where the robot might accidentally turn.

        Args:
            action_name (str): The intended action ("N", "S", "E", "W")

        Returns:
            dict: Mapping of outcome types to action names:
                - "intent": The intended action
                - "left": 90-degree left turn from intended
                - "right": 90-degree right turn from intended
        """
        # Map each direction to its left and right turns
        if action_name == "N":  # North: left=West, right=East
            return {"intent": "N", "left": "W", "right": "E"}
        elif action_name == "S":  # South: left=East, right=West
            return {"intent": "S", "left": "E", "right": "W"}
        elif action_name == "E":  # East: left=North, right=South
            return {"intent": "E", "left": "N", "right": "S"}
        elif action_name == "W":  # West: left=South, right=North
            return {"intent": "W", "left": "S", "right": "N"}
        else:
            raise ValueError(f"Invalid action name: {action_name}")


class QLearningAgent:
    """Q-Learning Agent that learns optimal policies through experience.

    Implements the Q-learning algorithm, a model-free reinforcement learning
    method that learns the quality of actions (Q-values) for each state.
    Uses epsilon-greedy exploration to balance exploration vs exploitation.

    The Q-learning update rule:
    Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]

    Where:
    - s: current state, a: action taken
    - r: immediate reward, s': next state
    - α: learning rate, γ: discount factor
    """

    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        """Initialize the Q-Learning agent with hyperparameters.

        Args:
            states (list): All valid states in the environment
            actions (list): All possible actions ("N", "S", "E", "W")
            alpha (float): Learning rate (0 < α ≤ 1)
                - Controls how much new information overrides old
                - Higher values = faster learning but less stability
            gamma (float): Discount factor (0 ≤ γ ≤ 1)
                - Determines importance of future rewards
                - γ=0: only immediate rewards matter
                - γ=1: future rewards as important as immediate
            epsilon (float): Initial exploration rate (0 ≤ ε ≤ 1)
                - Probability of taking random action vs best known action
                - ε=1: pure exploration, ε=0: pure exploitation
        """
        self.states = states
        self.actions = actions

        # Q-learning hyperparameters
        self.alpha = alpha  # Learning rate: how fast to update Q-values
        self.gamma = gamma  # Discount factor: importance of future rewards
        self.epsilon = epsilon  # Exploration rate: probability of random actions

        # Epsilon decay schedule - gradually shift from exploration to exploitation
        self.epsilon_min = (
            0.01  # Minimum exploration rate (always keep some randomness)
        )
        self.epsilon_decay = 0.99995  # Decay factor per episode

        # Initialize Q-table: Q(state, action) → expected future reward
        # Start with optimistic initialization (zeros) to encourage exploration
        self.q_table = {
            state: {action: 0.0 for action in self.actions} for state in self.states
        }

    def choose_action(self, state):
        """Select action using epsilon-greedy policy.

        The epsilon-greedy strategy balances exploration and exploitation:
        - With probability ε: choose random action (exploration)
        - With probability (1-ε): choose best known action (exploitation)

        This ensures the agent continues to discover new strategies while
        leveraging its current knowledge.

        Args:
            state (tuple): Current state (x, y) position

        Returns:
            str: Selected action ("N", "S", "E", or "W")
        """
        # Exploration: take random action with probability epsilon
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Random exploration

        # Exploitation: choose action with highest Q-value for this state
        q_values = self.q_table[state]
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning algorithm.

        Implements the core Q-learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]

        This updates our estimate of the value of taking action 'a' in state 's'
        based on the immediate reward and the best possible future value.

        Args:
            state (tuple): State where action was taken
            action (str): Action that was executed
            reward (float): Immediate reward received
            next_state (tuple): Resulting state after action
        """
        # Get the maximum Q-value for all possible actions in the next state
        # This represents the best possible future value from next_state
        max_next_q = max(self.q_table[next_state].values())

        # Apply the Q-learning update rule
        current_q = self.q_table[state][action]  # Current Q-value estimate

        # Temporal Difference (TD) target: r + γ*max(Q(s',a'))
        td_target = reward + self.gamma * max_next_q

        # TD error: difference between target and current estimate
        td_error = td_target - current_q

        # Update Q-value: move towards the target by learning rate α
        new_q = current_q + self.alpha * td_error
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Gradually reduce exploration rate over time.

        As the agent learns more about the environment, we want to shift
        from exploration (trying new things) to exploitation (using
        learned knowledge). This implements exponential decay.
        """
        # Only decay if we haven't reached the minimum exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Exponential decay


def print_policy(agent, env):
    """Visualize the learned policy as a grid with directional arrows.

    Creates a visual representation of the optimal policy learned by the agent.
    Each cell shows the best action to take from that state:
    - Arrows (↑↓←→) indicate the optimal movement direction
    - Numbers show terminal state rewards (+1.0, -1.0, etc.)
    - "WALL" marks the obstacle state

    Args:
        agent (QLearningAgent): Trained agent with learned Q-values
        env (GridWorld): Environment containing state and reward information
    """
    # Create empty grid matching environment dimensions
    # Note: grid[row][col] where row=0 is the TOP of the visual grid
    policy_grid = [
        [" " for _ in range(env.grid_size[0])] for _ in range(env.grid_size[1])
    ]

    # Map action names to visual arrow symbols
    action_arrows = {"N": "↑", "S": "↓", "E": "→", "W": "←"}

    # Fill grid with policy information for each state
    for state in agent.states:
        x, y = state  # x=column (1-4), y=row (1-3)

        # Convert (x,y) coordinates to grid array indices
        # y=1 (bottom) -> row index 2, y=3 (top) -> row index 0
        grid_row = env.grid_size[1] - y  # Flip y-axis for display
        grid_col = x - 1  # Convert 1-based to 0-based indexing

        if state in env.terminal_states:
            # Show reward value for terminal states
            policy_grid[grid_row][grid_col] = str(env.terminal_states[state])
        elif state == env.wall_state:
            # Mark wall/obstacle states
            policy_grid[grid_row][grid_col] = "WALL"
        else:
            # Show optimal action as directional arrow
            # Agent should be in pure exploitation mode (epsilon=0)
            best_action = agent.choose_action(state)
            policy_grid[grid_row][grid_col] = action_arrows[best_action]

    # Print formatted grid with borders
    print("\nLearned Optimal Policy:")
    print("(Arrows show best action from each state)")

    for row in policy_grid:
        # Print top border of each row
        print("+---" * len(row) + "+")
        # Print cell contents with side borders
        print("| " + " | ".join(f"{cell:^1}" for cell in row) + " |")
    # Print bottom border
    print("+---" * len(policy_grid[0]) + "+")


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == "__main__":
    """
    Main script that trains a Q-learning agent to navigate the GridWorld.
    
    The training process follows these steps:
    1. Initialize environment and agent with hyperparameters
    2. Run episodes where agent explores and learns
    3. Gradually reduce exploration (epsilon decay)
    4. Display learned optimal policy
    
    Each episode consists of:
    - Start at (1,1)
    - Take actions using epsilon-greedy policy  
    - Update Q-values after each step
    - Continue until reaching terminal state
    """

    # =================================================================
    # HYPERPARAMETER CONFIGURATION
    # =================================================================

    # Environment parameters
    PENALTY = -200.0  # Large negative penalty to strongly discourage bad terminal state
    # PENALTY = -1.0  # Smaller negative penalty to discourage bad terminal state

    # Training parameters
    EPISODES = 50000  # Number of episodes for training
    # More episodes = better learning but longer training time

    # Note: Agent hyperparameters are set in QLearningAgent constructor:
    # - alpha=0.1 (learning rate): moderate learning speed
    # - gamma=0.9 (discount factor): values future rewards highly
    # - epsilon=1.0 (initial exploration): start with pure exploration

    # =================================================================
    # ENVIRONMENT AND AGENT INITIALIZATION
    # =================================================================

    print("=" * 60)
    print("Q-LEARNING GRIDWORLD TRAINING")
    print("=" * 60)

    # Create the GridWorld environment
    env = GridWorld(penalty=PENALTY)
    print(f"Environment: {env.grid_size[0]}x{env.grid_size[1]} grid")
    print(f"Start state: {env.start_state}")
    print(f"Goal state: (4,3) with reward +1.0")
    print(f"Penalty state: (4,2) with reward {PENALTY}")
    print(f"Wall state: {env.wall_state}")
    print(f"Move reward: {env.move_reward}")

    # Create the Q-learning agent
    agent = QLearningAgent(
        states=env.get_states(),  # All valid grid positions
        actions=env.action_names,  # ["N", "S", "E", "W"]
        alpha=0.1,  # Learning rate
        gamma=0.9,  # Discount factor
        epsilon=1.0,  # Initial exploration rate
    )
    print(
        f"\nAgent initialized with {len(agent.states)} states and {len(agent.actions)} actions"
    )
    print(f"Learning rate (alpha): {agent.alpha}")
    print(f"Discount factor (gamma): {agent.gamma}")
    print(f"Initial exploration (epsilon): {agent.epsilon}")

    print(f"\nStarting training for {EPISODES} episodes...")
    print("Progress will be reported every 10,000 episodes\n")

    # =================================================================
    # TRAINING LOOP - Q-LEARNING ALGORITHM
    # =================================================================

    for episode in range(EPISODES):
        # Reset environment for new episode
        state = env.start_state  # Always start at (1,1)
        done = False
        steps_in_episode = 0

        # Run one complete episode (until terminal state reached)
        while not done:
            # STEP 1: Agent chooses action using epsilon-greedy policy
            action = agent.choose_action(state)

            # STEP 2: Environment executes action with stochastic transitions
            next_state, reward = env.step(state, action)

            # STEP 3: Agent updates Q-table based on experience
            # This is the core Q-learning update!
            agent.learn(state, action, reward, next_state)

            # STEP 4: Transition to next state
            state = next_state
            steps_in_episode += 1

            # STEP 5: Check if episode completed (reached terminal state)
            if state in env.terminal_states:
                done = True

        # STEP 6: Reduce exploration after each episode
        # Gradually shift from exploration to exploitation
        agent.decay_epsilon()

        # Progress reporting every 10,000 episodes
        if (episode + 1) % 10000 == 0:
            avg_steps = steps_in_episode  # Could track running average here
            print(
                f"Episode {episode + 1:,}/{EPISODES:,} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Steps: {steps_in_episode}"
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    # =================================================================
    # RESULTS AND POLICY VISUALIZATION
    # =================================================================

    # Switch to pure exploitation for policy display
    # (no more random exploration)
    agent.epsilon = 0.0

    print("\nFinal hyperparameters:")
    print(f"Final epsilon: {agent.epsilon}")
    print(f"Learning rate: {agent.alpha}")
    print(f"Discount factor: {agent.gamma}")

    # Display the learned optimal policy
    print_policy(agent, env)

    print("\nTraining complete! The agent has learned an optimal policy.")
    print("Each arrow shows the best action to take from that state.")
