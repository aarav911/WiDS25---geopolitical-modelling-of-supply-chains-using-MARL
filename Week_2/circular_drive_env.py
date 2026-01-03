import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class CircularDriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # ---------- Track ----------
        self.inner_r = 90
        self.outer_r = 110
        self.target_r = 100

        # ---------- Physics ----------
        self.dt = 0.1
        self.target_speed = 8.0
        self.max_speed = 18.0
        self.max_delta = np.pi / 2
        self.L = 2.5

        # ---------- Observation ----------
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        # ---------- Action (STILL Dict for your trainer compatibility) ----------
        self.action_space = spaces.Dict({
            "speed": spaces.Discrete(3),
            "steer_rate": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # ---------- Rendering ----------
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.size = 700
        self.scale = 2.5
        self.closed = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.target_r
        self.y = 0
        self.v = self.target_speed
        self.theta = np.pi / 2
        self.delta = 0.0

        self.steps = 0
        self.max_steps = 2000

        return self._get_state(), {}

    def step(self, action):
        # ---------- FIXED: Backward Compatible Dict Action ----------
        if isinstance(action, dict):
            steer_rate = float(action["steer_rate"][0])
        else:
            # Fallback for Box actions
            steer_rate = float(action[0])

        # ---------- IMPROVED Steering ----------
        self.delta += steer_rate * 0.08
        self.delta *= 0.95
        self.delta = np.clip(self.delta, -self.max_delta, self.max_delta)

        # ---------- Speed ----------
        self.v = self.target_speed

        # ---------- Heading ----------
        self.theta += (self.v / self.L) * np.tan(self.delta) * self.dt

        # ---------- Position ----------
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

        # ---------- State ----------
        r = np.sqrt(self.x**2 + self.y**2)
        theta_pos = np.arctan2(self.y, self.x)
        heading_error = (self.theta - (theta_pos + np.pi/2)) % (2*np.pi) - np.pi
        radial_error = abs(r - self.target_r)

        # ---------- IMPROVED Reward ----------
        reward = 1.0
        reward -= 0.1 * radial_error
        reward -= 0.3 * abs(heading_error)
        reward += 0.2 * np.cos(heading_error)

        done = False
        truncated = False

        if r <= self.inner_r or r >= self.outer_r:
            reward -= 50.0
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_state(), reward, done, truncated, {}

    def _get_state(self):
        return np.array([
            self.x / 120.0,
            self.y / 120.0,
            self.v / self.max_speed,
            np.sin(self.theta),
            np.cos(self.theta),
            self.delta / self.max_delta
        ], dtype=np.float32)

    def render(self):
        if self.render_mode is None or self.closed:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.size, self.size))
            pygame.display.set_caption("Circular Drive FIXED - Compatible")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((25, 25, 25))
        center = self.size // 2

        pygame.draw.circle(self.screen, (180, 180, 180), (center, center), int(self.outer_r * self.scale), width=8)
        pygame.draw.circle(self.screen, (180, 180, 180), (center, center), int(self.inner_r * self.scale), width=8)
        pygame.draw.circle(self.screen, (100, 100, 255), (center, center), int(self.target_r * self.scale), width=2)

        car_x = center + int(self.x * self.scale)
        car_y = center - int(self.y * self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), (car_x, car_y), 8)

        nose_x = car_x + int(18 * np.cos(self.theta))
        nose_y = car_y - int(18 * np.sin(self.theta))
        pygame.draw.line(self.screen, (255, 0, 0), (car_x, car_y), (nose_x, nose_y), 3)

        steer_end_x = car_x + int(12 * np.cos(self.theta + np.pi/2) * (self.delta / self.max_delta))
        steer_end_y = car_y - int(12 * np.sin(self.theta + np.pi/2) * (self.delta / self.max_delta))
        pygame.draw.line(self.screen, (0, 255, 255), (car_x, car_y), (steer_end_x, steer_end_y), 2)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
        self.closed = True
