import argparse
import csv
import os
import time

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def simple_velocity_policy(obs):
    """Fallback demo policy used when no PPO model is selected."""
    target_error = obs[0:3] * 3.0
    action = np.zeros(3, dtype=np.float32)
    action[0] = np.clip(0.55 * target_error[0], -0.8, 0.8)
    action[1] = np.clip(0.55 * target_error[1], -0.8, 0.8)
    action[2] = np.clip(0.85 * target_error[2], -0.9, 0.9)
    return action


def load_ppo_model(model_path):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Stable-Baselines3 is not installed, using the simple demo policy.")
        return None

    if not os.path.exists(model_path + ".zip"):
        print(f"No PPO model found at {model_path}.zip, using the simple demo policy.")
        return None

    print(f"Loading PPO model from {model_path}.zip")
    return PPO.load(model_path)


def run_demo(args):
    model = load_ppo_model(args.model_path) if args.use_ppo else None

    from env.HoverEnv import HoverEnv

    env = HoverEnv(
        gui=not args.no_gui,
        wind_enabled=not args.no_wind,
        wind_strength=args.wind,
        random_wind=args.random_wind,
        target_xyz=np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32),
        log_every_steps=0,
    )

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "fyp1_test_log.csv")

    fieldnames = [
        "episode", "step", "distance", "reward", "episode_reward",
        "x", "y", "z", "vx", "vy", "vz",
        "roll", "pitch", "yaw", "wind_x", "wind_y", "wind_z",
        "cmd_vx", "cmd_vy", "cmd_vz", "success",
    ]

    with open(log_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(args.episodes):
            obs = env.reset()
            episode_reward = 0.0
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

            for step in range(args.max_steps):
                if model is None:
                    action = simple_velocity_policy(obs)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, info = env.step(action)
                episode_reward += reward

                pos = info["position"]
                vel = info["velocity"]
                rpy = info["rpy"]
                wind = info["wind"]
                cmd = info["desired_velocity"]

                writer.writerow({
                    "episode": episode + 1,
                    "step": step,
                    "distance": info["distance"],
                    "reward": reward,
                    "episode_reward": episode_reward,
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "vx": vel[0],
                    "vy": vel[1],
                    "vz": vel[2],
                    "roll": rpy[0],
                    "pitch": rpy[1],
                    "yaw": rpy[2],
                    "wind_x": wind[0],
                    "wind_y": wind[1],
                    "wind_z": wind[2],
                    "cmd_vx": cmd[0],
                    "cmd_vy": cmd[1],
                    "cmd_vz": cmd[2],
                    "success": info["success"],
                })

                if step % args.print_every == 0:
                    print(
                        "step {:04d} | dist {:.3f} | reward {:7.2f} | "
                        "pos [{:.2f}, {:.2f}, {:.2f}] | "
                        "cmd [{:.2f}, {:.2f}, {:.2f}] | "
                        "wind [{:.3f}, {:.3f}, {:.3f}]".format(
                            step,
                            info["distance"],
                            episode_reward,
                            pos[0], pos[1], pos[2],
                            cmd[0], cmd[1], cmd[2],
                            wind[0], wind[1], wind[2],
                        )
                    )

                if done:
                    break

                if not args.no_gui:
                    time.sleep(1.0 / 240.0)

            print(
                "Finished: reward={:.2f}, min_distance={:.3f}, max_tilt={:.3f}, success={}".format(
                    episode_reward,
                    info["episode_min_distance"],
                    info["episode_max_tilt"],
                    info["success"],
                )
            )

    print(f"\nLog saved to: {os.path.abspath(log_path)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple FYP-1 drone navigation demo.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--target-x", type=float, default=1.0)
    parser.add_argument("--target-y", type=float, default=0.6)
    parser.add_argument("--target-z", type=float, default=1.0)
    parser.add_argument("--wind", type=float, default=0.006)
    parser.add_argument("--random-wind", action="store_true")
    parser.add_argument("--print-every", type=int, default=120)
    parser.add_argument("--use-ppo", action="store_true")
    parser.add_argument("--model-path", default="models/drone_model")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--no-wind", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
