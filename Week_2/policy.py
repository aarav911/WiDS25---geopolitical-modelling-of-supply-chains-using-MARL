import torch
from torch.distributions import Categorical, Normal
from ppo_model import ActorCritic


class PPOPolicy:
    def __init__(self, state_dim, lr=3e-4, device="cpu"):
        self.device = device
        self.net = ActorCritic(state_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ------------------------------------------------------
    # ACT FOR ENV INTERACTION
    # ------------------------------------------------------
    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.to(self.device)

        # Ensure batch dim
        if state.dim() == 1:
            state = state.unsqueeze(0)

        speed_logits, steer_mean, steer_std, value = self.net(state)

        # --- Discrete speed ---
        speed_dist = Categorical(logits=speed_logits)
        speed_action = speed_dist.sample()

        # --- Continuous steering ---
        steer_dist = Normal(steer_mean, steer_std)
        steer_action = steer_dist.sample()

        # Joint log-prob (ensure scalar)
        # steer_action has shape (batch,1) -> squeeze before adding
        log_prob = (
            speed_dist.log_prob(speed_action)
            + steer_dist.log_prob(steer_action.squeeze(-1))
        )

        return (
            int(speed_action.item()),
            float(steer_action.item()),
            float(log_prob.item()),
            float(value.item())
        )

    # ------------------------------------------------------
    # EVALUATE FOR PPO TRAINING
    # ------------------------------------------------------
    def evaluate(self, states, speed_actions, steer_actions):
        states = states.to(self.device)
        speed_actions = speed_actions.to(self.device)
        steer_actions = steer_actions.to(self.device)
        speed_logits, steer_mean, steer_std, values = self.net(states)

        # Flatten steering params to (batch,)
        steer_mean = steer_mean.squeeze(-1)
        steer_std = steer_std.squeeze(-1)

        # Distributions
        speed_dist = Categorical(logits=speed_logits)
        steer_dist = Normal(steer_mean, steer_std)

        # Log probs (make shapes compatible)
        speed_log_probs = speed_dist.log_prob(speed_actions)
        steer_log_probs = steer_dist.log_prob(steer_actions.squeeze(-1))

        total_log_probs = speed_log_probs + steer_log_probs

        # Entropy bonus
        entropy = speed_dist.entropy().mean() + steer_dist.entropy().mean()

        return total_log_probs, values.squeeze(-1), entropy
