import torch
import torch.nn.functional as F


class PPOTrainer:
    def __init__(
        self,
        policy,
        clip_range=0.2,
        epochs=8,
        batch_size=128,
        entropy_coef=0.01,
        value_coef=0.5,
    ):
        self.policy = policy
        self.clip = clip_range
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef


    def train(self, buffer):
        device = self.policy.device
        size = buffer.ptr

        states = buffer.states[:size].to(device)
        speed_actions = buffer.speed_actions[:size].to(device)
        steer_actions = buffer.steer_actions[:size].to(device)

        old_log_probs = buffer.log_probs[:size].to(device)
        returns = buffer.returns[:size].to(device)
        advantages = buffer.advantages[:size].to(device)

        for _ in range(self.epochs):

            perm = torch.randperm(size)
            for start in range(0, size, self.batch_size):
                idx = perm[start:start+self.batch_size]

                mb_states = states[idx]
                mb_speed = speed_actions[idx]
                mb_steer = steer_actions[idx]
                mb_old_logp = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]

                # ---- Evaluate policy ----
                logp, values, entropy = self.policy.evaluate(
                    mb_states,
                    mb_speed,
                    mb_steer
                )

                # ---- Policy loss (clipped PPO) ----
                ratio = torch.exp(logp - mb_old_logp)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)

                policy_loss = -torch.min(
                    ratio * mb_adv,
                    clipped_ratio * mb_adv
                ).mean()

                # ---- Value loss ----
                value_loss = F.mse_loss(values, mb_returns)

                # ---- Final loss ----
                loss = policy_loss \
                       + self.value_coef * value_loss \
                       - self.entropy_coef * entropy

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.net.parameters(), 0.5)
                self.policy.optimizer.step()
