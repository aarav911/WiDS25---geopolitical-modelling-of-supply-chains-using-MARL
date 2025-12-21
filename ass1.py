import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment setup
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# Hyperparameters
# -----------------------------
alpha = 0.1            # learning rate
gamma = 0.99           # discount factor
epsilon = 1.0          # initial exploration
epsilon_min = 0.01
epsilon_decay = 0.995

num_episodes = 5000
max_steps = 200

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

        # ε-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Take action in environment
        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Logging
    episode_rewards.append(total_reward)
    success_rate.append(1 if reward == 1 else 0)

env.close()
