import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env.HoverEnv import HoverEnv


class GymnasiumWrapper(gym.Env):
    """Small adapter because the base drone simulator uses the older Gym API."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = spaces.Box(
            low=np.asarray(env.action_space.low, dtype=np.float32),
            high=np.asarray(env.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray(env.observation_space.low, dtype=np.float32),
            high=np.asarray(env.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward), bool(done), False, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_env(args):
    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)
    return GymnasiumWrapper(HoverEnv(
        gui=args.gui,
        wind_enabled=not args.no_wind,
        wind_strength=args.wind,
        random_wind=args.random_wind,
        target_xyz=target,
        log_every_steps=0,
        randomize_dynamics=args.randomize_dynamics,
        obstacles_enabled=not args.no_obstacles,
        scenario=args.scenario,
    ))


def train(args):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback
        from collections import deque
    except ImportError as exc:
        raise SystemExit("Install stable-baselines3 to train PPO.") from exc

    class CurriculumCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.success_history = deque(maxlen=50)
            
        def _on_step(self) -> bool:
            # Check if episode just ended
            if "dones" in self.locals and self.locals["dones"][0]:
                info = self.locals["infos"][0]
                if "success" in info:
                    self.success_history.append(float(info["success"]))
                    
                    # Log base metrics every time we have a full history
                    if len(self.success_history) == 50:
                        win_rate = sum(self.success_history) / 50.0
                        
                        # Access the underlying HoverEnv
                        base_env = self.training_env.envs[0].env.env
                        
                        # Log to TensorBoard!
                        self.logger.record("curriculum/win_rate", win_rate)
                        self.logger.record("curriculum/stage", getattr(base_env, "current_curriculum_stage", 1))
                        
                        # If win rate is > 85%, promote the curriculum!
                        if win_rate >= 0.85:
                            promoted = base_env.promote_curriculum_stage()
                            if promoted:
                                # Reset history so we don't spam promotions
                                self.success_history.clear()
            return True

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = Monitor(make_env(args), filename="logs/ppo_monitor.csv")
    model_path = os.path.join("models", "drone_model")

    if args.resume and os.path.exists(model_path + ".zip"):
        print(f"Loading existing PPO model: {model_path}.zip")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new PPO model.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=2e-4,
            n_steps=1024,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            tensorboard_log="logs/ppo_tensorboard/"
        )

    callback = CurriculumCallback()
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume, callback=callback)
    model.save(model_path)
    env.close()
    print(f"Saved model to {model_path}.zip")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for simple drone target navigation.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-wind", action="store_true")
    parser.add_argument("--wind", type=float, default=0.006)
    parser.add_argument("--random-wind", action="store_true")
    parser.add_argument("--target-x", type=float, default=2.0)
    parser.add_argument("--target-y", type=float, default=0.8)
    parser.add_argument("--target-z", type=float, default=1.0)
    parser.add_argument("--randomize-dynamics", action="store_true", help="Enable Domain Randomization (mass, inertia, motor kf/km)")
    parser.add_argument("--no-obstacles", action="store_true", help="Disable obstacle cylinders in environment")
    parser.add_argument("--scenario", type=str, default="mixed", choices=["slalom", "forest", "racing", "tracking", "mixed"], help="Which procedural scenario to train on")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
