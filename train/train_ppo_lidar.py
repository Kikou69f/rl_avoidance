import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Allow importing the environment from the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.avoidance_env import AvoidanceEnv
import rclpy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train PPO on the lidar avoidance environment")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total number of training steps")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save checkpoints and the final model")
    parser.add_argument("--checkpoint-freq", type=int, default=1000, help="How often to save checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create environment
    env = AvoidanceEnv()

    # Instantiate the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="ppo_lidar",
    )

    # Training loop
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    # Save the final model
    model.save(os.path.join(args.save_dir, "ppo_lidar_final"))

    rclpy.shutdown()


if __name__ == "__main__":
    main()
