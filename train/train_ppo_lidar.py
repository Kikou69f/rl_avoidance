import os
import sys
import argparse
import signal

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Permet d'importer avoidance_env.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.avoidance_env import AvoidanceEnv

import rclpy
from rclpy.executors import ExternalShutdownException

# Globals
global_model = None
args = None
should_exit = False  # Flag pour arrêt propre

def signal_handler(sig, frame):
    """Intercepte Ctrl+C pour marquer l'arrêt."""
    global should_exit
    print("\n Ctrl+C détecté : arrêt propre demandé.")
    should_exit = True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement PPO pour évasion d'obstacles (ROS2 + CoppeliaSim)"
    )
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Nombre total d'étapes d'entraînement")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Répertoire pour checkpoints et modèle final")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                        help="Fréquence de sauvegarde (en steps)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Checkpoint à charger (ex: ppo_lidar_30000_steps)")
    return parser.parse_args()

def main() -> None:
    global global_model, args, should_exit
    args = parse_args()
    signal.signal(signal.SIGINT, signal_handler)

    # Crée l'env (initialise ROS2 à l'intérieur)
    env = AvoidanceEnv()

    # Prépare dossiers
    os.makedirs(args.save_dir, exist_ok=True)
    # Ici on crée le dossier de logs TensorBoard sous train/ppo_lidar_tensorboard
    tb_log = os.path.join(os.path.dirname(__file__), "ppo_lidar_tensorboard")
    os.makedirs(tb_log, exist_ok=True)

    # Chargement ou création du modèle
    if args.load_checkpoint:
        ckpt = os.path.join(args.save_dir, args.load_checkpoint + ".zip")
        if os.path.isfile(ckpt):
            model = PPO.load(ckpt, env=env, device="auto")
            print(f" Modèle chargé depuis {ckpt}")
        else:
            print(f"  Checkpoint introuvable ({ckpt}), création d'un nouveau modèle.")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-4,
                clip_range=0.1,
                n_steps=2048,
                n_epochs=5,
                ent_coef=0.02,
                tensorboard_log=tb_log,
                device="auto"
            )
    else:
        print("  Création d'un nouveau modèle PPO.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            clip_range=0.1,
            n_steps=2048,
            n_epochs=5,
            ent_coef=0.02,
            tensorboard_log=tb_log,
            device="auto"
        )

    global_model = model

    # Callback pour checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="ppo_lidar"
    )

    # Entraînement
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=False,
            tb_log_name="run"
        )
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    # Sauvegarde en cas d'arrêt Ctrl+C
    if should_exit:
        path = os.path.join(args.save_dir, "ppo_lidar_interrupted")
        global_model.save(path)
        print(f" Modèle interrompu sauvegardé : {path}.zip")

    # Sauvegarde finale
    if rclpy.ok():
        final = os.path.join(args.save_dir, "ppo_lidar_final")
        model.save(final)
        print(f" Modèle final sauvegardé : {final}.zip")

    # Nettoyage ROS2
    env.close()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == "__main__":
    main()