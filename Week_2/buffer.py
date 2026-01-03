import torch


class RolloutBuffer:
    def __init__(self, size, state_dim, device="cpu"):
        self.device = device
        self.max = size
        self.ptr = 0

        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)

        self.speed_actions = torch.zeros(size, dtype=torch.long, device=device)
        self.steer_actions = torch.zeros(size, 1, dtype=torch.float32, device=device)

        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)

        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)

        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)


    def add(self, state, speed, steer, reward, done, value, log_prob):
        if self.ptr >= self.max:
            return

        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.speed_actions[self.ptr] = int(speed)
        self.steer_actions[self.ptr] = float(steer)

        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)

        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)

        self.ptr += 1


    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        # Only use real collected data
        size = self.ptr

        adv = 0.0
        self.advantages[:size] = 0

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t+1]
                next_done = self.dones[t+1]

            delta = (
                self.rewards[t] 
                + gamma * next_value * (1 - next_done)
                - self.values[t]
            )

            adv = (
                delta
                + gamma * lam * (1 - next_done) * adv
            )

            self.advantages[t] = adv

        self.returns[:size] = self.advantages[:size] + self.values[:size]


    def normalize_advantages(self):
        size = self.ptr
        adv = self.advantages[:size]
        self.advantages[:size] = (adv - adv.mean()) / (adv.std() + 1e-8)
