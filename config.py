"""
config.py - Hyperparameter configurations
"""


def get_default_config():
    """
    Get default hyperparameter configuration.
    
    Returns:
        dict: hyperparameter configuration
    """
    return {
        # Network architecture
        'hidden_sizes': [64, 64],           # Two hidden layers with 64 neurons each
        
        # Optimization
        'learning_rate': 0.001,             # Adam optimizer learning rate
        'batch_size': 32,                   # Minibatch size for training
        
        # Q-learning
        'gamma': 0.99,                      # Discount factor
        
        # Exploration
        'epsilon_start': 1.0,               # Initial exploration rate
        'epsilon_min': 0.01,                # Minimum exploration rate
        'epsilon_decay': 0.995,             # Epsilon decay per episode
        
        # Experience replay
        'buffer_capacity': 10000,           # Replay buffer capacity
        
        # Target network
        'target_update_freq': 100,          # Update target network every N steps
    }


def get_fast_config():
    """
    Get configuration for faster training (for testing).
    
    Returns:
        dict: hyperparameter configuration
    """
    return {
        'hidden_sizes': [32, 32],
        'learning_rate': 0.001,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.99,              # Faster decay
        'buffer_capacity': 5000,
        'target_update_freq': 50,
    }


def get_deep_config():
    """
    Get configuration with deeper network.
    
    Returns:
        dict: hyperparameter configuration
    """
    return {
        'hidden_sizes': [128, 128, 64],     # Three hidden layers
        'learning_rate': 0.0005,            # Lower learning rate for stability
        'batch_size': 64,                   # Larger batch size
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 20000,
        'target_update_freq': 200,
    }