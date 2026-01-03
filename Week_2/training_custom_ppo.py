import numpy as np
import torch

from circular_drive_env import CircularDriveEnv
from policy import PPOPolicy
from ppo_trainer import PPOTrainer
from buffer import RolloutBuffer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ================== ENV ==================
env = CircularDriveEnv(render_mode=None)
state_dim = 6

# ================== POLICY ==================
policy = PPOPolicy(state_dim, device=DEVICE)
trainer = PPOTrainer(policy)

ROLL_SIZE = 4096
buffer = RolloutBuffer(ROLL_SIZE, state_dim, device=DEVICE)

timesteps = 500000
t = 0

obs, _ = env.reset()
episode_reward = 0
episode_count = 0

while t < timesteps:

    buffer.ptr = 0

    while buffer.ptr < ROLL_SIZE:

        # -------- Get Action from Policy --------
        with torch.no_grad():
            speed, steer, logp, value = policy.act(obs)

        # convert to env action
        action = {
            "speed": int(speed),
            "steer_rate": np.array([float(steer)], dtype=np.float32),
        }

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done_flag = terminated or truncated

        # -------- Store Transition --------
        buffer.add(
            state=obs,
            speed=int(speed),
            steer=float(steer),
            reward=float(reward),
            done=done_flag,
            value=float(value),
            log_prob=float(logp)
        )

        obs = next_obs
        t += 1
        episode_reward += reward

        if done_flag:
            episode_count += 1
            print(f"Episode {episode_count} finished | Reward = {episode_reward:.2f}")
            episode_reward = 0
            obs, _ = env.reset()

    # =====================================================
    # Bootstrap last value for GAE
    # =====================================================
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        last_value = policy.net(obs_tensor)[-1].item()

    buffer.compute_gae(last_value)
    buffer.normalize_advantages()

    # ================= PPO Update =================
    print(f"\n===== PPO UPDATE at timestep {t} =====")
    trainer.train(buffer)

    print(f"Mean buffer reward: {float(buffer.rewards[:buffer.ptr].mean()):.3f}")
    print(f"Mean advantage: {float(buffer.advantages[:buffer.ptr].mean()):.3f}")

    # -------- Save --------
    torch.save(policy.net.state_dict(), "ppo_car.pt")
    print("Model saved to ppo_car.pt\n")

env.close()
print("Training complete!")
