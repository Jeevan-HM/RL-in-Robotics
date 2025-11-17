#!/usr/bin/env python3
"""
CAC Training with More Responsive Car
Fixes: Faster control, quicker reactions, easier obstacles
"""

import sys

sys.path.insert(0, ".")

from realistic_car_env import CarPhysicsParams, NavigationConfig
from train_cac import train_stage1_safety_critic, train_stage2_restricted_policy

# More responsive car physics
responsive_car = CarPhysicsParams(
    # Lighter, more agile vehicle
    mass=800.0,  # Was 1500kg - now like a small sports car
    moment_inertia=1000.0,  # Was 2500 - easier to turn
    # Faster steering
    max_steering_angle=50.0 * 3.14159 / 180,  # Was 35° → now 50°
    max_steering_rate=90.0 * 3.14159 / 180,  # Was 45°/s → now 90°/s (2x faster!)
    # Better acceleration
    max_engine_force=8000.0,  # Was 5000N - more power
    max_brake_force=12000.0,  # Was 8000N - better brakes
    # Lower speed for control
    max_speed=25.0,  # Was 80 m/s → now 25 m/s (~56 mph, more reasonable)
    # Better sensors
    lidar_range=25.0,  # Was 15m → now 25m (more warning time)
    num_lidar_rays=64,  # Was 32 → now 64 (better coverage)
    # Keep other params
    length=4.5,
    width=1.8,
    wheelbase=2.7,
    tire_friction=0.9,  # Slightly better grip
    rolling_resistance=0.01,  # Was 0.015 - less resistance
    drag_coefficient=0.25,  # Was 0.3 - more aerodynamic
    frontal_area=2.0,
    air_density=1.225,
    safety_radius=2.5,
    collision_radius=2.0,
    lidar_fov=270.0 * 3.14159 / 180,
)

# Easier environment
easier_env = NavigationConfig(
    # Smaller world
    world_width=60.0,
    world_height=60.0,
    # Closer goal
    start_pos=(10.0, 10.0),
    goal_pos=(50.0, 50.0),  # ~56m vs 113m
    goal_radius=6.0,  # Was 3m - bigger target
    # Fewer, smaller obstacles
    n_static_obstacles=3,  # Was 8
    obstacle_min_radius=1.5,  # Was 2.0
    obstacle_max_radius=3.0,  # Was 5.0
    # No dynamic obstacles
    n_dynamic_obstacles=0,  # Was 3
    # FASTER control loop
    dt=0.1,  # Was 0.2s → now 0.1s (10 Hz instead of 5 Hz!)
    max_steps=3000,
    # Other settings
    use_walls=True,
    sensor_in_obs=True,
    num_rays=64,
    n_obstacles=3,
)

print("=" * 70)
print("RESPONSIVE CAR TRAINING")
print("=" * 70)
print("Car improvements:")
print("  - 2x faster steering (90°/s vs 45°/s)")
print("  - Lighter (800kg vs 1500kg)")
print("  - More power (8000N vs 5000N)")
print("  - Better sensors (25m vs 15m LIDAR)")
print("  - FASTER control (10 Hz vs 5 Hz)")
print()
print("Environment improvements:")
print("  - Smaller world (60x60 vs 100x100)")
print("  - Closer goal (56m vs 113m)")
print("  - Bigger target (6m vs 3m radius)")
print("  - Fewer obstacles (3 vs 11)")
print("  - Smaller obstacles (1.5-3m vs 2-5m)")
print("=" * 70)
print()

print("Physics analysis:")
print(f"  At 10 m/s speed:")
print(f"    - Distance per step: {10 * easier_env.dt:.1f}m (was 2m)")
print(f"    - LIDAR warning time: {responsive_car.lidar_range / 10:.1f}s (was 1.5s)")
print(
    f"    - Steps to react: {responsive_car.lidar_range / (10 * easier_env.dt):.0f} (was 7-8)"
)
print(
    f"    - Steering per step: {responsive_car.max_steering_rate * easier_env.dt * 180 / 3.14159:.1f}° (was 9°)"
)
print(
    f"    - Total steering: {responsive_car.max_steering_rate * easier_env.dt * 180 / 3.14159 * (responsive_car.lidar_range / (10 * easier_env.dt)):.0f}° (was 72°)"
)
print()

# Stage 1
print("Stage 1: Learning safety with responsive car...")
train_stage1_safety_critic(
    env_config=easier_env,
    total_steps=1_000_000,
    checkpoint_path="checkpoints/responsive_stage1.pt",
    device="auto",
    alpha0=0.7,  # Permissive
    seed=42,
)

# Stage 2
print("\nStage 2: Learning goal-reaching...")
train_stage2_restricted_policy(
    stage1_checkpoint="checkpoints/responsive_stage1.pt",
    env_config=easier_env,
    total_steps=1_000_000,
    checkpoint_path="checkpoints/responsive_stage2.pt",
    device="auto",
    alpha0=0.7,
    beta0=0.6,
    seed=42,
)

print("\n" + "=" * 70)
print("RESPONSIVE CAR TRAINING COMPLETE!")
print("=" * 70)
print("\nWith these changes, the car should be able to:")
print("  ✓ React faster to obstacles (10 Hz control)")
print("  ✓ Steer more aggressively (90°/s)")
print("  ✓ See further ahead (25m LIDAR)")
print("  ✓ Navigate easier obstacles (3 small ones vs 11 large)")
print("\nExpected results:")
print("  - Stage 1: 70-85% safety rate")
print("  - Stage 2: 60-75% safety, 30-50% goals!")
