import numpy as np
import torch
from circular_drive_env import CircularDriveEnv
from policy import PPOPolicy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

env = CircularDriveEnv(render_mode="human")

policy = PPOPolicy(6)
policy.net.load_state_dict(torch.load("ppo_car.pt", map_location=DEVICE))
policy.net.to(DEVICE)
policy.net.eval()

obs, _ = env.reset()

while True:
    if getattr(env, "closed", False):
        break

    with torch.no_grad():
        state = torch.tensor(obs, dtype=torch.float32).to(DEVICE)

        # ---- DETERMINISTIC ACTION ----
        logits, mean, std, _ = policy.net(state)

        speed = torch.argmax(logits).item()       # Best discrete action
        steer = mean.item()                       # Mean of Gaussian

    action = {
        "speed": speed,
        "steer_rate": np.array([steer], dtype=np.float32),
    }

    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
