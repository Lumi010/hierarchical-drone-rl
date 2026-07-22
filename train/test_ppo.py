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
    if disable_wind:
        env.env.wind_enabled = False
        env.env.current_curriculum_stage = 1
    else:
        env.env.wind_enabled = True

    for ep in range(episodes):
        if is_mixed:
            stage = random.choice([1, 2, 3])
            
        if not disable_wind:
            env.env.current_curriculum_stage = stage
            
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        ep_min_dist = float('inf')
        ep_success = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_steps += 1
            
            ep_min_dist = min(ep_min_dist, info.get("distance", float('inf')))
            if info.get("success", False):
                ep_success = True

        if ep_success:
            successes += 1
        
        min_distances.append(ep_min_dist)
        total_steps += ep_steps
        status = "SUCCESS" if ep_success else "FAILED"
        
