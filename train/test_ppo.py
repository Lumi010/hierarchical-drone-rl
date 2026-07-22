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
        
        # Display the specific stage if mixed, otherwise the general condition name
        display_name = f"Mixed (Stage {stage})" if is_mixed else condition_name
        print(f"Episode {ep+1:02d} | Condition: {display_name} | Status: {status} | Min Dist: {ep_min_dist:.3f}m")

    win_rate = (successes / episodes) * 100
    avg_min_dist = sum(min_distances) / len(min_distances)
    return win_rate, avg_min_dist

def test(args):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install stable-baselines3 to run PPO.") from exc

    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)
    
    # We initialize the env normally. The run_test_loop handles overriding wind settings.
    env = GymnasiumWrapper(HoverEnv(
        gui=args.gui,
        wind_enabled=True,
        wind_strength=args.wind,
        random_wind=True,
        target_xyz=target,
        log_every_steps=0,
        randomize_dynamics=args.randomize_dynamics,
        obstacles_enabled=not args.no_obstacles,
        curriculum_enabled=True,
        scenario=args.scenario,
    ))

    model_path = os.path.join("models", "drone_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first!")

    print(f"Loading model from {model_path}...\n")
    model = PPO.load(model_path)
    
    conditions = [
        {"name": "No Wind", "stage": 1, "disable_wind": True, "is_mixed": False},
        {"name": "Low Wind", "stage": 1, "disable_wind": False, "is_mixed": False},
        {"name": "Medium Wind", "stage": 2, "disable_wind": False, "is_mixed": False},
        {"name": "High Wind", "stage": 3, "disable_wind": False, "is_mixed": False},
        {"name": "Mixed Wind", "stage": 3, "disable_wind": False, "is_mixed": True},
    ]
    
    if args.mode == "benchmark":
        print("=== STARTING FULL BENCHMARK SUITE ===")
        results = []
        for c in conditions:
            print(f"\n--- Testing Condition: {c['name']} ---")
            win_rate, avg_dist = run_test_loop(
                env, model, c['name'], c['stage'], args.episodes, 
                disable_wind=c['disable_wind'], is_mixed=c['is_mixed']
            )
            results.append({
                "Condition": c['name'], 
                "Episodes": args.episodes, 
                "Win Rate (%)": win_rate, 
                "Avg Min Dist (m)": avg_dist
            })
        
        # Save results to CSV
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join("results", f"benchmark_{timestamp}.csv")
        
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Condition", "Episodes", "Win Rate (%)", "Avg Min Dist (m)"])
            writer.writeheader()
            for r in results:
