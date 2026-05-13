# Reinforcement Learning on Stochastic FrozenLake
## Q-Learning vs SARSA in Slippery Environments

This project implements and analyzes two classical model-free reinforcement learning algorithms — Q-Learning and SARSA — on the stochastic 8×8 FrozenLake environment from Gymnasium.

The primary objective is to study how off-policy and on-policy learning behave under uncertainty, particularly in environments with stochastic transitions (`is_slippery=True`).

The implementation includes:

- Tabular Q-Learning
- Tabular SARSA
- Large-scale training and evaluation
- Hyperparameter sensitivity analysis
- Policy visualization
- Learning curve analysis
- Comparative performance benchmarking

---

# Environment

The experiments use:

- FrozenLake-v1
- Map size: 8x8
- Stochastic transitions enabled (`is_slippery=True`)

In the slippery setting, intended actions are not always executed exactly, making the environment highly stochastic and significantly more difficult than the deterministic variant.

State space:
- 64 discrete states

Action space:
- 4 discrete actions:
  - Left
  - Down
  - Right
  - Up

Reward structure:
- +1 for reaching the goal
- 0 otherwise

---

# Algorithms Implemented

## Q-Learning

Off-policy temporal-difference learning algorithm:

Q(s,a) ← Q(s,a) + α[r + γ max(Q(s',a')) − Q(s,a)]

Characteristics:
- Optimistic updates
- Learns toward the greedy policy
- Better performance in stochastic environments

---

## SARSA

On-policy temporal-difference learning algorithm:

Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]

Characteristics:
- Learns from actual executed policy
- More conservative behavior
- Safer but less aggressive exploration

---

# Training Configuration

| Parameter | Value |
|---|---|
| Episodes | 500,000 |
| Max Steps | 200 |
| Learning Rate (α) | 0.1 |
| Discount Factor (γ) | 0.99 |
| Initial Epsilon | 1.0 |
| Minimum Epsilon | 0.05 |
| Epsilon Decay | 0.99995 |

Additional adjustments were necessary because slippery FrozenLake has extremely noisy transitions and sparse rewards.

---

# Results

Final evaluation over 1000 greedy rollouts:

| Algorithm | Success Rate |
|---|---|
| Q-Learning | ~48–51% |
| SARSA | ~32–35% |

Key observations:

- Q-Learning consistently outperformed SARSA.
- Q-Learning learned more aggressive policies closer to holes.
- SARSA learned safer but longer routes.
- Higher stochasticity amplified the difference between off-policy and on-policy learning.

---

# Hyperparameter Sensitivity

Learning rate sweep results:

| Learning Rate | Success Rate |
|---|---|
| 0.05 | ~41% |
| 0.10 | Best performance |
| 0.30 | Significant instability |

Observations:
- Very high learning rates destabilized training.
- Moderate learning rates produced the best convergence.
- Slower epsilon decay was essential for effective exploration.

---

# Features

## Learning Curve Visualization

The project generates smoothed rolling-average learning curves for:
- Q-Learning
- SARSA

These plots help compare:
- Convergence speed
- Stability
- Final asymptotic performance

---

## Policy Visualization

The learned policies are visualized directly on the 8×8 grid using directional arrows.

This enables qualitative comparison between:
- Aggressive policies
- Conservative policies
- Risk-sensitive navigation behavior

---

# Evaluation Pipeline

Includes:
- Greedy policy evaluation
- Success rate measurement
- Policy divergence analysis
- Statistical summaries

---

# Repository Structure

.
├── train_q_learning.py
├── train_sarsa.py
├── evaluation.py
├── visualization.py
├── learning_curves.png
├── policies.png
├── final_comparison.png
├── README.md
└── report.pdf

---

# Installation

Install dependencies:

pip install gymnasium numpy matplotlib

---

# Running the Project

Train and evaluate agents:

python train_q_learning.py
python train_sarsa.py

Generate plots:

python visualization.py

---

# Example Output

Q-Learning final: 48.2%
SARSA final: 31.7%

Example policy divergence:

Policy divergence: 28.1% of states differ

---

# Key Takeaways

This project demonstrates several important RL concepts:

- Difference between on-policy and off-policy learning
- Exploration vs exploitation trade-offs
- Effects of stochasticity on learning dynamics
- Sensitivity to hyperparameters
- Challenges of sparse-reward environments

The experiments also show why optimistic off-policy methods like Q-Learning can outperform conservative on-policy methods in noisy environments.

---

# Future Work

Potential extensions include:

- Deep Q-Networks (DQN)
- Double DQN
- PPO implementation
- Multi-agent reinforcement learning
- Continuous control environments
- Supply-chain optimization simulations

---

# References

1. Sutton & Barto — Reinforcement Learning: An Introduction
2. Watkins & Dayan — Q-Learning
3. OpenAI Gym / Gymnasium Documentation
4. PPO Paper (Schulman et al.)
