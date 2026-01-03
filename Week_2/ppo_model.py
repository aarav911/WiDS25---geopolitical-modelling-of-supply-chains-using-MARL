import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim, speed_actions=3):
        super().__init__()

        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # --- Actor: Speed (Discrete) ---
        self.speed_head = nn.Linear(128, speed_actions)

        # --- Actor: Steering (Gaussian Continuous) ---
        self.steer_mean = nn.Linear(128, 1)
        self.steer_log_std = nn.Parameter(torch.zeros(1))

        # --- Critic ---
        self.value_head = nn.Linear(128, 1)


    def forward(self, x):
        x = self.shared(x)

        # Discrete
        speed_logits = self.speed_head(x)

        # Continuous
        steer_mean = self.steer_mean(x)
        steer_log_std = self.steer_log_std.expand_as(steer_mean)
        steer_std = torch.exp(steer_log_std)

        value = self.value_head(x)

        return speed_logits, steer_mean, steer_std, value
