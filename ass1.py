# Omitted some parts from the code so it won't run till you fill them up

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment setup
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="6x6",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# Hyperparameters
# -----------------------------
# alpha = ?          # learning rate
# gamma = ?         # discount factor
# epsilon = ?        # initial exploration
# epsilon_min = ?
# epsilon_decay = ?

# num_episodes = ?
# max_steps = ?

# -----------------------------
# Q-table initialization
# -----------------------------
Q = np.zeros((n_states, n_actions))

# -----------------------------
# Logging
# -----------------------------
episode_rewards = []
success_rate = []

# -----------------------------
# Q-learning loop
# -----------------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps):

        # Implement ε-greedy action selection

        # Q-learning update

    # Explore epsilon decay


    # Track success rate 


env.close()
