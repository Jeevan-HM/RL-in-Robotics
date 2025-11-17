#!/usr/bin/env python3
"""
Continue Stage 2 training from existing checkpoint
"""

import sys
sys.path.insert(0, '.')

from train_cac import train_stage2_restricted_policy

# Continue Stage 2 training with more steps
print("Continuing Stage 2 training from checkpoint...")
print("This will improve goal-reaching performance.\n")

train_stage2_restricted_policy(
    stage1_checkpoint="checkpoints/stage1_safe_policy.pt",
    total_steps=500_000,  # Much longer
    checkpoint_path="checkpoints/stage2_final_policy.pt",
    device="auto",
    alpha0=0.5,
    beta0=0.8,
    seed=43  # Different seed for variety
)
