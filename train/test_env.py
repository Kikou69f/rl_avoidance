import sys
import os
import time
import rclpy
from geometry_msgs.msg import Twist

# Ajouter le chemin du projet à PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from envs.avoidance_env import AvoidanceEnv

def main():
    env = AvoidanceEnv()
    obs = env.reset()
    print("Observation shape:", obs.shape)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {step}: reward={reward:.2f}, done={done}")
        time.sleep(0.2)  # Attente pour voir le robot bouger

        if done:
            print("Fin d’épisode : collision détectée.")
            break

    # Stop le robot proprement à la fin
    stop_msg = Twist()
    env.cmd_pub.publish(stop_msg)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
