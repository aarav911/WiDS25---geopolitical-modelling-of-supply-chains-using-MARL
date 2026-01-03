from circular_drive_env import CircularDriveEnv
import numpy as np

env = CircularDriveEnv(render_mode="human")
obs, _ = env.reset()

target_delta = np.arctan(env.L / env.target_r)

print("Ideal steering delta =", target_delta)

for step in range(2000):
    # FORCE CORRECT PHYSICS CONDITIONS
    env.v = 8.0
    env.delta = target_delta

    #   IMPORTANT:
    #   steer_rate = 0 so ENV CANNOT CHANGE delta
    action = np.array([0.0, 0.0], dtype=np.float32)

    prev_x, prev_y = env.x, env.y
    prev_theta = env.theta

    obs, reward, done, trunc, _ = env.step(action)

    r = np.sqrt(env.x**2 + env.y**2)

    print(
        f"step={step:04d}  "
        f"v={env.v:.2f}  "
        f"delta={env.delta:.4f}  "
        f"theta={env.theta:.4f}  "
        f"r={r:.3f}  "
        f"dx={env.x - prev_x:.4f}  "
        f"dy={env.y - prev_y:.4f}"
    )

    env.render()

    if done or trunc:
        break

env.close()
