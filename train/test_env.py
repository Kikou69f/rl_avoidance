import sys
import os
import time
from geometry_msgs.msg import Twist

# Ajouter le chemin du projet à PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from envs.avoidance_env import AvoidanceEnv
import rclpy

def main():
    # Ne PAS faire rclpy.init() ici : c'est déjà géré dans AvoidanceEnv
    env = AvoidanceEnv()

    # unpack properly reset()
    obs, info = env.reset()
    print("Observation shape:", obs.shape, "— Goal:", info["goal"])

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")

        time.sleep(0.2)  # pour visualiser dans CoppeliaSim

        if terminated or truncated:
            print(f"Fin d’épisode (terminated={terminated}, truncated={truncated}).")
            break

    # Stopper le robot proprement
    stop_msg = Twist()
    env.cmd_pub.publish(stop_msg)

    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
