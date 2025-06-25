# RL Obstacle Avoidance Project

Ce projet vise à entraîner un robot (Pioneer P3DX dans CoppeliaSim) à éviter des obstacles en utilisant l'apprentissage par renforcement (algorithmes PPO et SAC). 

Les observations utilisées seront soit les données brutes du Lidar soit une grille d'occupation (occupancy grid). Le projet repose sur ROS2, stable-baselines3, PyTorch, et Gym.

##  Structure du projet
- `envs/`: environnement Gym personnalisé
- `train/`: scripts d'entraînement
- `models/`: modèles entraînés
- `utils/`: fonctions utilitaires

## Lancer la simulation

-Ouvrir un nouveau terminal, exécuter successivement :
conda activate rl_avoidance
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp



## Pour entraîner l’agent PPO :

python3 train/train_ppo_lidar.py \
  --timesteps 500000 \
  --save-dir models \
  --checkpoint-freq 50000


Les logs sont écrits dans le dossier train/ppo_lidar_tensorboard/.

-Lancer TensorBoard avec :
tensorboard --logdir train/ppo_lidar_tensorboard --port 6006

-Puis ouvrir http://localhost:6006