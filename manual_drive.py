#!/usr/bin/env python3
"""Manual keyboard driver for ModularCar2DEnv.

Run this script to try the car-like throttle/steer controls (or the point-mass fallback).
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from rl_main import EnvConfig, ModularCar2DEnv

try:
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "cairoagg", "figurecanvasagg"}:
        for candidate in ("QtAgg", "Qt5Agg", "TkAgg"):
            try:
                matplotlib.use(candidate, force=True)
                print(f"Switching Matplotlib backend to {candidate}")
                break
            except Exception:
                continue
    import matplotlib.pyplot as plt
    backend = plt.get_backend().lower()
    if backend in {"agg", "cairoagg", "figurecanvasagg"}:
        raise RuntimeError(
            "Matplotlib is using a non-interactive backend (Agg). "
            "Set MPLBACKEND=TkAgg (or QtAgg/Qt5Agg) before running, "
            "and ensure the corresponding GUI toolkit is installed."
        )
except Exception as exc:  # pragma: no cover - informative failure
    raise RuntimeError(
        "matplotlib with an interactive backend is required for manual control. "
        "Install it via `python -m pip install matplotlib pyqt5` (or tkinter)."
    ) from exc


def manual_drive(config: EnvConfig | None = None) -> None:
    """Launch a simple keyboard-controlled rollout."""
    plt.ion()
    env = ModularCar2DEnv(config or EnvConfig(vehicle_model="car"))

    obs, info = env.reset()
    del obs, info  # not used in manual mode

    action = np.zeros(2, dtype=np.float32)
    pressed = set()
    running = True
    car_mode = env.cfg.vehicle_model.lower() == "car"
    thrust = env.cfg.max_accel

    env.render()
    fig = env._fig  # type: ignore[attr-defined]
    if fig is None:
        raise RuntimeError(
            "Matplotlib did not create a window. "
            "Ensure you are running with a GUI backend (e.g., `python -m pip install pyqt5`)."
        )

    key_alias: Dict[str, str] = {
        "left": "left",
        "a": "left",
        "right": "right",
        "d": "right",
        "up": "up",
        "w": "up",
        "down": "down",
        "s": "down",
        "space": "space",
        " ": "space",
    }

    def compute_action() -> np.ndarray:
        if car_mode:
            throttle = 0.0
            if "up" in pressed:
                throttle += 1.0
            if "down" in pressed or "space" in pressed:
                throttle -= 1.0
            throttle = float(np.clip(throttle, -1.0, 1.0))

            steer_rate = 0.0
            if "left" in pressed:
                steer_rate += 1.0
            if "right" in pressed:
                steer_rate -= 1.0
            steer_rate = float(np.clip(steer_rate, -1.0, 1.0))
            return np.array([throttle, steer_rate], dtype=np.float32)

        ax = 0.0
        ay = 0.0
        if "left" in pressed:
            ax -= thrust
        if "right" in pressed:
            ax += thrust
        if "up" in pressed:
            ay += thrust
        if "down" in pressed:
            ay -= thrust
        return np.array([ax, ay], dtype=np.float32)

    def on_key_press(event) -> None:
        nonlocal action, running
        key = (event.key or "").lower()
        if key in {"escape", "q"}:
            running = False
            return
        mapped = key_alias.get(key)
        if mapped:
            pressed.add(mapped)
            action = compute_action()

    def on_key_release(event) -> None:
        nonlocal action
        key = (event.key or "").lower()
        mapped = key_alias.get(key)
        if mapped and mapped in pressed:
            pressed.discard(mapped)
            action = compute_action()

    def on_close(_event) -> None:
        nonlocal running
        running = False

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_connect("close_event", on_close)

    if car_mode:
        print(
            "Manual drive controls (car model):\n"
            "  - Arrow keys / WASD: throttle (up/W), brake (down/S), steer (left/right or A/D)\n"
            "  - Space: brake hard\n"
            "  - Q or Esc: quit\n"
            "Close the window or press Esc to exit."
        )
    else:
        print(
            "Manual drive controls (point mass):\n"
            "  - Arrow keys or WASD: accelerate along X/Y axes\n"
            "  - Q or Esc: quit\n"
            "Close the window or press Esc to exit."
        )

    try:
        while running:
            _, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                reason = info.get("terminated_reason") or ("truncated" if truncated else "terminated")
                print(f"Episode finished ({reason}). Resetting...")
                time.sleep(0.75)
                env.reset()
                action = np.zeros(2, dtype=np.float32)
                pressed.clear()
                env.render()
                continue

            time.sleep(env.cfg.dt)
    finally:
        env.close()
        plt.ioff()


if __name__ == "__main__":
    manual_drive()
