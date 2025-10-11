import numpy as np
import gymnasium as gym
from gymnasium import spaces

class IsaacPlanarNavEnv(gym.Env):
    """A minimal Isaac Sim environment that mimics the 2D PlanarNav task.

    It uses Isaac Sim's Python modules to create a stage with:
      - a kinematic "robot" (cylinder) we move by setting its transform,
      - multiple cube obstacles,
      - synthetic rangefinder by computing distances to cubes in Python.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, headless=True, seed=None, L=10.0, N_obs=20, dt=0.1, n_rays=17, fov_deg=120,
                 robot_radius=0.25, margin=0.15, v=0.9, alpha=0.9, beta=0.9, max_steps=800):
        super().__init__()
        try:
            from omni.isaac.kit import SimulationApp
            self._sim_app = SimulationApp({"headless": headless})
            from omni.isaac.core import World
            from omni.isaac.core.objects import DynamicCuboid, VisualCylinder
        except Exception as e:
            raise RuntimeError("Isaac Sim modules not found. Run inside Isaac Sim's Python.") from e

        self.World = World
        self.DynamicCuboid = DynamicCuboid
        self.VisualCylinder = VisualCylinder

        self.L = float(L); self.N_obs=int(N_obs); self.dt=float(dt); self.n_rays=int(n_rays)
        self.fov = np.deg2rad(fov_deg); self.R=float(robot_radius); self.margin=float(margin)
        self.v=float(v); self.alpha=float(alpha); self.beta=float(beta); self.max_steps=int(max_steps)
        self.rng = np.random.default_rng(seed)

        high = np.ones(self.n_rays + 6, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                       high=np.array([1.0], dtype=np.float32))

        self.world = self.World(stage_units_in_meters=1.0)
        self._stage_built = False
        self._reward_mode = "safety"

    def _build_stage(self):
        self.world.scene.clear()
        self.world.scene.add_default_ground_plane()

        self.obstacles = []
        half_min, half_max = 0.15, 0.5
        for i in range(self.N_obs):
            half = float(self.rng.uniform(half_min, half_max))
            cx = float(self.rng.uniform(-self.L*0.5+half, self.L*0.5-half))
            cy = float(self.rng.uniform(-self.L*0.5+half, self.L*0.5-half))
            name = f"box_{i}"
            box = self.DynamicCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=np.array([cx, cy, half], dtype=np.float32),
                scale=np.array([half*2, half*2, half*2], dtype=np.float32),
                color=np.array([1.0, 0.1, 0.1]),
            )
            self.world.scene.add(box)
            self.obstacles.append((cx, cy, half))

        self.robot = self.VisualCylinder(
            prim_path="/World/robot",
            name="robot",
            position=np.array([0, 0, self.R], dtype=np.float32),
            scale=np.array([self.R*2, self.R*2, self.R*2], dtype=np.float32),
            color=np.array([0.1, 0.1, 1.0]),
        )
        self.world.scene.add(self.robot)

        self.world.reset()
        self._stage_built = True

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if not self._stage_built:
            self._build_stage()

        self.start = self._free_point(self.R + self.margin + 0.1)
        self.goal = self._free_point(self.R + self.margin + 0.1)

        self.p = self.start.copy()
        self.theta = float(self.rng.uniform(-np.pi, np.pi))
        self.steps = 0
        self.omega_prev = 0.0

        self.robot.set_world_pose(position=np.array([self.p[0], self.p[1], self.R], dtype=np.float32))
        self.world.step(render=False)

        obs = self._get_obs()
        return obs, {}

    def _free_point(self, rad):
        for _ in range(1000):
            p = self.rng.uniform(-self.L*0.45, self.L*0.45, size=(2,))
            ok = True
            for cx, cy, h in self.obstacles:
                if (abs(p[0]-cx) <= h + rad) and (abs(p[1]-cy) <= h + rad):
                    ok = False; break
            if ok:
                return p.astype(np.float32)
        return np.zeros(2, dtype=np.float32)

    def _rangefinder(self, p, theta):
        max_range = self.L
        angles = theta + np.linspace(-self.fov/2, self.fov/2, self.n_rays)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        ranges = np.full((self.n_rays,), max_range, dtype=np.float32)
        for i, d in enumerate(dirs):
            ro = p.astype(np.float32)
            rd = d / (np.linalg.norm(d) + 1e-8)
            for cx, cy, h in self.obstacles:
                minb = np.array([cx-h, cy-h], dtype=np.float32)
                maxb = np.array([cx+h, cy+h], dtype=np.float32)
                tmin = (minb - ro) / (rd + 1e-8)
                tmax = (maxb - ro) / (rd + 1e-8)
                t1 = np.minimum(tmin, tmax)
                t2 = np.maximum(tmin, tmax)
                t_near = np.max(t1); t_far = np.min(t2)
                if t_far >= max(0.0, t_near):
                    hit_t = t_near if t_near > 0 else t_far
                    if 0.0 < hit_t < ranges[i]:
                        ranges[i] = hit_t
        return np.clip(ranges, 0.0, max_range).astype(np.float32)

    def _min_range(self, p, theta):
        return float(np.min(self._rangefinder(p, theta)))

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

    def set_stage(self, mode: str):
        assert mode in ("safety","goal")
        self._reward_mode = mode

    def step(self, action):
        a = float(np.clip(action, -1.0, 1.0))
        self.omega_prev = a
        omega = a * 1.0
        self.theta = float(self.theta + omega * self.dt)
        heading = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        prev_p = self.p.copy()
        self.p = self.p + heading * self.v * self.dt

        self.robot.set_world_pose(position=np.array([self.p[0], self.p[1], self.R], dtype=np.float32))
        self.world.step(render=False)

        def h(p): return self._min_range(p, self.theta) - self.margin
        def l(p):
            d=(p-self.goal)
            return 0.5*float(np.dot(d,d))

        h_now = h(prev_p); h_nxt = h(self.p)
        delta_h = h_nxt + (self.alpha-1.0)*h_now

        l_now = l(prev_p); l_nxt = l(self.p)
        delta_l = l_nxt + (self.beta-1.0)*l_now

        r1 = float(np.exp(min(delta_h, 0.0)))
        r2 = float(-max(delta_l, 0.0))
        r = self._reward_mode == "safety" and r1 or r2

        self.steps += 1
        done = False
        trunc = False
        if self._min_range(self.p, self.theta) <= self.R + self.margin*0.5:
            done = True
        if np.linalg.norm(self.goal - self.p) < 1.5:
            done = True
        if self.steps >= self.max_steps:
            trunc = True

        info = {"r1": r1, "r2": r2, "delta_h": float(delta_h), "delta_l": float(delta_l)}
        obs = self._get_obs()
        return obs, float(r), done, trunc, info

    def close(self):
        try:
            pass
        except Exception:
            pass
