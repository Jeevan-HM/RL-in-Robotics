from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .state import VehicleState


class BaseVehicleModel:
    """Polymorphic dynamics handler so env logic stays clean."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def action_space(self) -> spaces.Box:
        raise NotImplementedError

    def reset_state(self) -> VehicleState:
        raise NotImplementedError

    def step(self, state: VehicleState, action: np.ndarray, dt: float) -> VehicleState:
        raise NotImplementedError


class PointVehicleModel(BaseVehicleModel):
    def action_space(self) -> spaces.Box:
        high = np.array([self.cfg.max_accel, self.cfg.max_accel], dtype=np.float32)
        return spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset_state(self) -> VehicleState:
        pos = np.array(self.cfg.start_pos, dtype=np.float32)
        vel = np.array(self.cfg.start_vel, dtype=np.float32)
        accel = np.zeros(2, dtype=np.float32)
        heading = float(self.cfg.start_heading)
        speed = float(np.linalg.norm(vel))
        if speed > 1e-6:
            heading = float(np.arctan2(vel[1], vel[0]))
        return VehicleState(
            position=pos,
            velocity=vel,
            accel=accel,
            heading=heading,
            steer=0.0,
            long_speed=speed,
            command_accel=np.zeros(2, dtype=np.float32),
        )

    def step(self, state: VehicleState, action: np.ndarray, dt: float) -> VehicleState:
        a_cmd = np.clip(action, -self.cfg.max_accel, self.cfg.max_accel)
        tau = max(self.cfg.engine_tau, 1e-6)
        lagged = state.command_accel + (dt / tau) * (a_cmd - state.command_accel)
        drag = - (self.cfg.drag_coeff / max(self.cfg.mass, 1e-6)) * state.velocity
        accel = lagged + drag
        velocity = state.velocity + accel * dt
        position = state.position + velocity * dt
        heading = state.heading
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-6:
            heading = float(np.arctan2(velocity[1], velocity[0]))
        return VehicleState(
            position=position.astype(np.float32),
            velocity=velocity.astype(np.float32),
            accel=accel.astype(np.float32),
            heading=heading,
            steer=0.0,
            long_speed=speed,
            command_accel=lagged.astype(np.float32),
        )


class BicycleVehicleModel(BaseVehicleModel):
    def action_space(self) -> spaces.Box:
        high = np.ones(2, dtype=np.float32)
        return spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset_state(self) -> VehicleState:
        heading = float(self.cfg.start_heading)
        steer = float(np.clip(self.cfg.start_steering, -self.cfg.max_steer, self.cfg.max_steer))
        if self.cfg.start_speed is not None:
            long_speed = float(self.cfg.start_speed)
        else:
            vx0, vy0 = self.cfg.start_vel
            if np.hypot(vx0, vy0) > 1e-6:
                heading = float(np.arctan2(vy0, vx0))
            long_speed = float(np.cos(heading) * vx0 + np.sin(heading) * vy0)
        direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        velocity = direction * long_speed
        pos = np.array(self.cfg.start_pos, dtype=np.float32)
        return VehicleState(
            position=pos,
            velocity=velocity.astype(np.float32),
            accel=np.zeros(2, dtype=np.float32),
            heading=heading,
            steer=steer,
            long_speed=long_speed,
            command_accel=np.zeros(2, dtype=np.float32),
            engine_accel=0.0,
            longitudinal_accel=0.0,
        )

    def step(self, state: VehicleState, action: np.ndarray, dt: float) -> VehicleState:
        throttle_cmd = float(np.clip(action[0], -1.0, 1.0))
        steer_cmd = float(np.clip(action[1], -1.0, 1.0))

        desired_long = throttle_cmd * (self.cfg.max_throttle if throttle_cmd >= 0.0 else self.cfg.max_brake)
        tau = max(self.cfg.engine_tau, 1e-6)
        engine_accel = state.engine_accel + (dt / tau) * (desired_long - state.engine_accel)

        drag_coeff = self.cfg.drag_coeff / max(self.cfg.mass, 1e-6)
        a_long = engine_accel - drag_coeff * state.long_speed

        long_speed_new = state.long_speed + a_long * dt
        if not self.cfg.allow_reverse:
            long_speed_new = max(0.0, long_speed_new)
        else:
            long_speed_new = float(np.clip(long_speed_new, -self.cfg.speed_limit, self.cfg.speed_limit))
        long_speed_mid = 0.5 * (state.long_speed + long_speed_new)

        steer_rate = steer_cmd * self.cfg.max_steer_rate
        steer_new = float(np.clip(state.steer + steer_rate * dt, -self.cfg.max_steer, self.cfg.max_steer))

        wheelbase = max(self.cfg.wheelbase, 1e-6)
        heading_rate = long_speed_mid * np.tan(steer_new) / wheelbase if abs(long_speed_mid) > 1e-6 else 0.0
        heading_new = float(
            np.arctan2(np.sin(state.heading + heading_rate * dt), np.cos(state.heading + heading_rate * dt))
        )
        heading_mid = state.heading + 0.5 * (heading_new - state.heading)

        direction_mid = np.array([np.cos(heading_mid), np.sin(heading_mid)], dtype=np.float32)
        position = state.position + direction_mid * (long_speed_mid * dt)

        direction_new = np.array([np.cos(heading_new), np.sin(heading_new)], dtype=np.float32)
        velocity = direction_new * long_speed_new
        accel_vec = direction_new * a_long

        return VehicleState(
            position=position.astype(np.float32),
            velocity=velocity.astype(np.float32),
            accel=accel_vec.astype(np.float32),
            heading=heading_new,
            steer=steer_new,
            long_speed=float(long_speed_new),
            command_accel=accel_vec.astype(np.float32),
            engine_accel=float(engine_accel),
            longitudinal_accel=float(a_long),
        )
