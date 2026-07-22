import argparse
import os
import sys
import numpy as np
import random
import csv
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env.HoverEnv import HoverEnv
from train.train_ppo import GymnasiumWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Test trained PPO drone.")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet visualizer")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per condition")
    parser.add_argument("--mode", type=str, choices=["none", "low", "medium", "high", "mixed", "benchmark"], 
                        default="high", help="Which wind mode to test. 'benchmark' tests ALL of them.")
    parser.add_argument("--target-x", type=float, default=2.0)
    parser.add_argument("--target-y", type=float, default=0.8)
    parser.add_argument("--target-z", type=float, default=1.0)
    parser.add_argument("--wind", type=float, default=0.006)
    parser.add_argument("--randomize-dynamics", action="store_true")
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--scenario", type=str, default="slalom", choices=["slalom", "forest", "racing", "tracking", "mixed"])
    return parser.parse_args()

def run_test_loop(env, model, condition_name, stage, episodes, disable_wind=False, is_mixed=False):
    successes = 0
    total_steps = 0
    min_distances = []

    # Apply specific overrides for this test loop
