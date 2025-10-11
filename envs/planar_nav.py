import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PlanarNavEnv(gym.Env):
    """2D planar obstacle-avoidance env similar to the AUV task in spirit.

    - Robot: disc of radius R, constant forward speed v, control is yaw rate ω ∈ [-1,1] rad/s.
    - World: square of size L, with N axis-aligned square obstacles (side ∈ [a_min, a_max]).
    - Observation: rangefinder distances over FOV, relative goal vector, heading, yaw rate.
    - CBF: h(s) = min_range - margin.
    - CLF: l(s) = 0.5 * ||p - g||^2.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None,
                 L=10.0,
                 N_obs=20,
                 obs_side_min=0.3,
                 obs_side_max=1.0,
                 robot_R=0.25,
                 safety_margin=0.15,
                 v=0.9,
                 dt=0.1,
                 max_steps=800,
                 n_rays=17,
                 fov_deg=120,
                 alpha=0.9,
                 beta=0.9):
        super().__init__()
        self.L = float(L)
        self.N_obs = int(N_obs)
        self.obs_side_min = float(obs_side_min)
        self.obs_side_max = float(obs_side_max)
        self.R = float(robot_R)
        self.margin = float(safety_margin)
        self.v = float(v)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.n_rays = int(n_rays)
        self.fov = np.deg2rad(fov_deg)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = 0.99

        high = np.ones(self.n_rays + 6, dtype=np.float32)  # ranges + [dx, dy, cos, sin, omega, dist_norm]
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                       high=np.array([1.0], dtype=np.float32))

        self.rng = np.random.default_rng(seed)
        self._build_world()
        self.state = None
        self.steps = 0
        self.omega_prev = 0.0
        self._reward_mode = "safety"

    def _build_world(self):
        self.obstacles = []
        for _ in range(self.N_obs):
            half = self.rng.uniform(self.obs_side_min, self.obs_side_max) * 0.5
            cx = self.rng.uniform(-self.L*0.5+half, self.L*0.5-half)
            cy = self.rng.uniform(-self.L*0.5+half, self.L*0.5-half)
            self.obstacles.append([cx, cy, half])
        self.obstacles = np.array(self.obstacles, dtype=np.float32)

        def free_point():
            for _ in range(1000):
                p = self.rng.uniform(-self.L*0.45, self.L*0.45, size=(2,))
                if self._clear_of_obstacles(p, self.R + self.margin + 0.1):
                    return p
            return np.zeros(2, dtype=np.float32)
        self.start = free_point()
        self.goal  = free_point()

    def _clear_of_obstacles(self, p, rad):
        for cx, cy, h in self.obstacles:
            if (abs(p[0]-cx) <= h + rad) and (abs(p[1]-cy) <= h + rad):
                return False
        return True

    def h(self, p):
        min_range = self._min_range(p, self.theta)
        return min_range - self.margin

    def l(self, p):
        d = p - self.goal
        return 0.5 * float(np.dot(d, d))

    def _min_range(self, p, theta):
        ranges = self._rangefinder(p, theta)
        return float(np.min(ranges))

    def _rangefinder(self, p, theta):
        max_range = self.L
        origins = p.astype(np.float32)
        angles = theta + np.linspace(-self.fov/2, self.fov/2, self.n_rays)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        ranges = np.full((self.n_rays,), max_range, dtype=np.float32)
        for i, d in enumerate(dirs):
            ro = origins
            rd = d / (np.linalg.norm(d) + 1e-8)
            for cx, cy, h in self.obstacles:
                minb = np.array([cx-h, cy-h], dtype=np.float32)
                maxb = np.array([cx+h, cy+h], dtype=np.float32)
                tmin = (minb - ro) / (rd + 1e-8)
                tmax = (maxb - ro) / (rd + 1e-8)
                t1 = np.minimum(tmin, tmax)
                t2 = np.maximum(tmin, tmax)
                t_near = np.max(t1)
                t_far  = np.min(t2)
                if t_far >= max(0.0, t_near):
                    hit_t = t_near if t_near > 0 else t_far
                    if 0.0 < hit_t < ranges[i]:
                        ranges[i] = hit_t
        ranges = np.clip(ranges, 0.0, max_range).astype(np.float32)
        return ranges

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._build_world()
        self.p = self.start.copy()
        self.theta = self.rng.uniform(-np.pi, np.pi)
        self.steps = 0
        self.omega_prev = 0.0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        ranges = self._rangefinder(self.p, self.theta)
        d = self.goal - self.p
        dist = np.linalg.norm(d) + 1e-6
        rel = d / dist
        obs = np.concatenate([
            (ranges / (self.L)).astype(np.float32),
            rel.astype(np.float32),
            np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32),
            np.array([self.omega_prev/1.0], dtype=np.float32),
            np.array([min(dist / (self.L*0.7), 1.0)], dtype=np.float32),
        ], axis=0)
        return obs.astype(np.float32)

    def step(self, action):
        a = float(np.clip(action, -1.0, 1.0))
        self.omega_prev = a
        omega = a * 1.0
        self.theta = float(self.theta + omega * self.dt)
        heading = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        prev_p = self.p.copy()
        self.p = self.p + heading * self.v * self.dt

        h_now = self.h(prev_p)
        h_nxt = self.h(self.p)
        delta_h = h_nxt + (self.alpha-1.0)*h_now

        l_now = self.l(prev_p)
        l_nxt = self.l(self.p)
        delta_l = l_nxt + (self.beta-1.0)*l_now

        r1 = float(np.exp(min(delta_h, 0.0)))
        r2 = float(-max(delta_l, 0.0))
        r = self._reward_mode == "safety" and r1 or r2

        self.steps += 1
        done = False
        trunc = False
        min_r = self._min_range(self.p, self.theta)
        if min_r <= self.R + self.margin * 0.5:
            done = True
        if np.linalg.norm(self.goal - self.p) < 1.5:
            done = True
        if self.steps >= self.max_steps:
            trunc = True

        info = {"r1": r1, "r2": r2, "delta_h": float(delta_h), "delta_l": float(delta_l)}
        obs = self._get_obs()
        return obs, float(r), done, trunc, info

    def set_stage(self, mode: str):
        assert mode in ("safety","goal")
        self._reward_mode = mode

    def render(self):
        pass
