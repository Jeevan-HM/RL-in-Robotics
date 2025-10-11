import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PlanarNavEnv(gym.Env):
    """2D planar obstacle-avoidance env similar to the AUV task in spirit.

    - Robot: disc of radius R, constant forward speed v, control is yaw rate ω ∈ [-1,1] rad/s.
    - World: square of size L, with N axis-aligned square obstacles (side ∈ [a_min, a_max]).
    - Observation: rangefinder distances over FOV, relative goal vector, heading, yaw rate.
    - CBF: h(s) = min_range - margin.
    - CLF: l(s) = 0.5 * ||p - g||^2.
    """

    # metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        seed=None,
        L=10.0,
        N_obs=20,
        obs_side_min=0.3,
        obs_side_max=1.0,
        robot_R=0.25,
        safety_margin=0.3,  # Increased from 0.15 to 0.3
        v=0.9,
        dt=0.1,
        max_steps=800,
        n_rays=17,
        fov_deg=120,
        alpha=0.95,  # Increased from 0.9 for stronger CBF constraint
        beta=0.9,
        enhanced_obs=True,
        render_mode=None,
    ):  # Control whether to use enhanced observations
        super().__init__()

        self.L = float(L)
        self.N_obs = int(N_obs)
        self.render_mode = render_mode
        self._fig = None
        self._ax = None
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
        self.enhanced_obs = enhanced_obs

        obs_dim = self.n_rays + 7 if enhanced_obs else self.n_rays + 6
        high = np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.rng = np.random.default_rng(seed)
        self._build_world()
        self.state = None
        self.steps = 0
        self.omega_prev = 0.0
        self._reward_mode = "safety"

    def _dist_to_walls(self, p):
        """Minimum Euclidean distance from point p to the square world's walls."""
        # Walls at x = ±L/2, y = ±L/2
        return float(min(self.L * 0.5 - abs(p[0]), self.L * 0.5 - abs(p[1])))

    def _build_world(self):
        self.obstacles = []
        for _ in range(self.N_obs):
            half = self.rng.uniform(self.obs_side_min, self.obs_side_max) * 0.5
            cx = self.rng.uniform(-self.L * 0.5 + half, self.L * 0.5 - half)
            cy = self.rng.uniform(-self.L * 0.5 + half, self.L * 0.5 - half)
            self.obstacles.append([cx, cy, half])
        self.obstacles = np.array(self.obstacles, dtype=np.float32)

        def free_point():
            for _ in range(1000):
                p = self.rng.uniform(-self.L * 0.45, self.L * 0.45, size=(2,))
                if self._clear_of_obstacles(p, self.R + self.margin + 0.1):
                    return p
            return np.zeros(2, dtype=np.float32)

        self.start = free_point()
        self.goal = free_point()

    def _clear_of_obstacles(self, p, rad):
        for cx, cy, h in self.obstacles:
            if (abs(p[0] - cx) <= h + rad) and (abs(p[1] - cy) <= h + rad):
                return False
        return True

    def h(self, p):
        # min forward range from center
        min_range = self._min_range(p, self.theta)
        # Convert to *clearance* by subtracting robot radius
        clearance = min_range - self.R
        # Safety certificate with margin: h>0 means safe
        return clearance - self.margin

    def l(self, p):
        d = p - self.goal
        return 0.5 * float(np.dot(d, d))

    def _min_range(self, p, theta):
        ranges = self._rangefinder(p, theta)
        return float(np.min(ranges))

    def _rangefinder(self, p, theta):
        max_range = self.L
        angles = theta + np.linspace(-self.fov / 2, self.fov / 2, self.n_rays)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
        ranges = np.full((self.n_rays,), max_range, dtype=np.float32)

        px, py = float(p[0]), float(p[1])
        halfL = self.L * 0.5
        eps = 1e-8

        for i, d in enumerate(dirs):
            rox, roy = px, py
            rdx, rdy = float(d[0]), float(d[1])

            # --- Hit test with square room walls (x = ±halfL, y = ±halfL) ---
            cand_t = []

            # x walls
            if rdx > eps:
                cand_t.append((halfL - rox) / rdx)
            elif rdx < -eps:
                cand_t.append((-halfL - rox) / rdx)
            # y walls
            if rdy > eps:
                cand_t.append((halfL - roy) / rdy)
            elif rdy < -eps:
                cand_t.append((-halfL - roy) / rdy)

            # Keep the smallest positive wall t
            t_wall = min([t for t in cand_t if t > 0.0], default=max_range)
            ranges[i] = min(ranges[i], t_wall)

            # --- Hit test with axis-aligned box obstacles ---
            for cx, cy, h in self.obstacles:
                minb = np.array([cx - h, cy - h], dtype=np.float32)
                maxb = np.array([cx + h, cy + h], dtype=np.float32)

                # slab method
                tmin = (minb - np.array([rox, roy], dtype=np.float32)) / (
                    np.array([rdx, rdy], dtype=np.float32) + eps
                )
                tmax = (maxb - np.array([rox, roy], dtype=np.float32)) / (
                    np.array([rdx, rdy], dtype=np.float32) + eps
                )
                t1 = np.minimum(tmin, tmax)
                t2 = np.maximum(tmin, tmax)
                t_near = float(np.max(t1))
                t_far = float(np.min(t2))

                if t_far >= max(0.0, t_near):
                    hit_t = t_near if t_near > 0 else t_far
                    if 0.0 < hit_t < ranges[i]:
                        ranges[i] = hit_t

        return np.clip(ranges, 0.0, max_range).astype(np.float32)

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

        if self.enhanced_obs:
            # Improve range sensitivity by using a smaller normalization factor
            range_norm_factor = self.L * 0.3  # More sensitive than full world size
            normalized_ranges = np.clip(ranges / range_norm_factor, 0.0, 1.0)

            # Add minimum range as explicit safety signal
            min_range = np.min(ranges)
            safety_indicator = np.array(
                [min_range / (self.R + self.margin * 2)], dtype=np.float32
            )

            obs = np.concatenate(
                [
                    normalized_ranges.astype(np.float32),
                    rel.astype(np.float32),
                    np.array(
                        [np.cos(self.theta), np.sin(self.theta)], dtype=np.float32
                    ),
                    np.array([self.omega_prev / 1.0], dtype=np.float32),
                    np.array([min(dist / (self.L * 0.7), 1.0)], dtype=np.float32),
                    safety_indicator,  # Additional safety signal
                ],
                axis=0,
            )
        else:
            # Original observation format for compatibility
            obs = np.concatenate(
                [
                    (ranges / self.L).astype(np.float32),
                    rel.astype(np.float32),
                    np.array(
                        [np.cos(self.theta), np.sin(self.theta)], dtype=np.float32
                    ),
                    np.array([self.omega_prev / 1.0], dtype=np.float32),
                    np.array([min(dist / (self.L * 0.7), 1.0)], dtype=np.float32),
                ],
                axis=0,
            )
        return obs.astype(np.float32)

    def step(self, action):
        a = float(np.clip(action, -1.0, 1.0))
        self.omega_prev = a
        omega = a * 1.0
        self.theta = float(self.theta + omega * self.dt)
        heading = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        prev_p = self.p.copy()
        self.p = self.p + heading * self.v * self.dt

        px = np.clip(self.p[0], -self.L * 0.5 + self.R, self.L * 0.5 - self.R)
        py = np.clip(self.p[1], -self.L * 0.5 + self.R, self.L * 0.5 - self.R)
        hit_wall = (px != self.p[0]) or (py != self.p[1])
        self.p[0], self.p[1] = px, py

        # --- certificates + rewards ---
        h_now = self.h(prev_p)
        h_nxt = self.h(self.p)
        delta_h = h_nxt + (self.alpha - 1.0) * h_now

        l_now = self.l(prev_p)
        l_nxt = self.l(self.p)
        delta_l = l_nxt + (self.beta - 1.0) * l_now

        # Calculate current clearance for safety monitoring
        clearance_now = self._min_range(self.p, self.theta) - self.R

        # Strengthen safety reward signal
        if delta_h <= 0:
            r1 = float(np.exp(delta_h))  # Same exponential for negative delta_h
        else:
            r1 = 1.0 + 0.1 * delta_h  # Small positive reward for improving safety

        # Add immediate penalty for getting too close
        if clearance_now <= self.margin * 1.5:  # Warning zone
            safety_penalty = -5.0 * (1.5 * self.margin - clearance_now) / self.margin
            r1 += safety_penalty

        r2 = float(-max(delta_l, 0.0))
        r = self._reward_mode == "safety" and r1 or r2

        # make info FIRST so we can safely add termination later
        info = {
            "r1": r1,
            "r2": r2,
            "delta_h": float(delta_h),
            "delta_l": float(delta_l),
        }

        # --- terminations ---
        self.steps += 1
        done = False
        trunc = False
        termination_reason = None

        # obstacle collision (via min range)
        # Use 2/3 of safety margin as collision threshold (balanced approach)
        collision_threshold = self.margin * 0.67
        if clearance_now <= collision_threshold:
            done = True
            info["termination"] = "collision"

        # wall contact (treat as collision; remove 'done' if you prefer sliding)
        wall_dist = self._dist_to_walls(self.p)
        if (
            wall_dist <= self.R + collision_threshold or hit_wall
        ):  # Use same threshold for walls
            done = True
            termination_reason = "wall"

        # goal success (immediate end)
        if np.linalg.norm(self.goal - self.p) < 1.5:
            done = True
            termination_reason = "goal"

        # timeout
        if self.steps >= self.max_steps:
            trunc = True
            termination_reason = termination_reason or "timeout"

        # observation for next step
        obs = self._get_obs()

        # attach reason if any
        if termination_reason is not None:
            info["termination"] = termination_reason

        return obs, float(r), done, trunc, info

    def set_stage(self, mode: str):
        assert mode in ("safety", "goal")
        self._reward_mode = mode

    def render(self):
        """Render the environment state."""
        if self.render_mode is None:
            return

        try:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available for rendering")
            return

        # Initialize plot if not exists
        if self._fig is None:
            plt.ion()  # Turn on interactive mode
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._ax.set_xlim(-self.L, self.L)
            self._ax.set_ylim(-self.L, self.L)
            self._ax.set_aspect("equal")
            self._ax.grid(True, alpha=0.3)
            self._ax.set_title("CAC Agent Navigation")
            self._ax.set_xlabel("X")
            self._ax.set_ylabel("Y")

        # Clear previous frame
        self._ax.clear()
        self._ax.set_xlim(-self.L, self.L)
        self._ax.set_ylim(-self.L, self.L)
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.3)

        # Draw obstacles
        for obs in self.obstacles:
            circle = patches.Circle(obs, 2.0, color="red", alpha=0.7)
            self._ax.add_patch(circle)

        # Draw goal
        goal_circle = patches.Circle(self.goal, 1.5, color="green", alpha=0.5)
        self._ax.add_patch(goal_circle)

        # Draw agent
        agent_circle = patches.Circle(self.p, 0.5, color="blue", alpha=0.8)
        self._ax.add_patch(agent_circle)

        # Draw CBF safety boundary (approximate)
        cbf_val = self.h(self.p)  # Control Barrier Function
        if cbf_val <= 0:
            # Agent is in unsafe region
            safety_circle = patches.Circle(
                self.p, 1.0, fill=False, edgecolor="red", linewidth=3, linestyle="--"
            )
            self._ax.add_patch(safety_circle)

        # Add info text
        info_text = f"Position: ({self.p[0]:.2f}, {self.p[1]:.2f})\n"
        info_text += f"CBF: {cbf_val:.3f}\n"
        info_text += f"CLF: {self.l(self.p):.3f}"  # Control Lyapunov Function
        self._ax.text(
            -self.L + 1,
            self.L - 2,
            info_text,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.draw()
        plt.pause(0.01)  # Small pause for animation

        if self.render_mode == "human":
            return
