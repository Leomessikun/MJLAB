#!/bin/bash

# Test script to play the trained spinkick model
# This script uses the correct task name and parameters

echo "Testing play with trained model..."
echo "Using latest training run: logs/rsl_rl/g1_tracking/2025-10-23_16-34-47"
echo ""

# Set environment variables
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

# Run the play command with the correct task name
# Note: Use "Mjlab-Spinkick-Unitree-G1-Play" for playing (not training version)
# The checkpoint file should be the .pt file from your training run
uv run play.py Mjlab-Spinkick-Unitree-G1-Play \
  --motion-file OmniRetarget_Dataset/converted/motion_climb_00_1.0_fixed2.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/2025-10-23_16-34-47/model_2500.pt \
  --device cuda:0
