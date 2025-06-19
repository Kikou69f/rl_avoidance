import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

class AvoidanceEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Initialise ROS2
        rclpy.init(args=None)
        self.node = rclpy.create_node('avoidance_env_node')

        # Observation : 1D array des distances Lidar
        self.scan = None
        self.scan_sub = self.node.create_subscription(LaserScan,'/scan',self.scan_callback,10)

        # Pose du robot 
        self.pose = None
        self.pose_sub = self.node.create_subscription(Pose,'/pose',self.pose_callback,10)

        # Action : cmd_vel (vitesse linéaire et angulaire)
        self.cmd_pub = self.node.create_publisher(Twist,'/cmd_vel',10)

        # Définition des espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(360,),  # 360 valeurs Lidar
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),   # [vitesse, rotation]
            high=np.array([0.5, 1.0]),
            dtype=np.float32
        )

        # Variables internes
        self.collision = False
        self.rate = self.node.create_rate(10)

    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges, dtype=np.float32)

    def pose_callback(self, msg):
        self.pose = msg.position  # On simplifiera pour l’instant

    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment.

        Parameters
        ----------
        seed : Optional[int]
            Seed for the environment RNG. It is ignored as the simulation
            handles its own randomness but is accepted for compatibility with
            the Gymnasium API.
        options : Optional[dict]
            Additional options, unused but kept for API compatibility.
        """

        super().reset(seed=seed)

        # Reset de la simulation manuellement (CoppeliaSim bouton Reset)

        # Attente active jusqu'à avoir un scan valide
        while self.scan is None:
            rclpy.spin_once(self.node)

        # Option : clear cmd_vel
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)

        obs = np.clip(self.scan, 0.0, 10.0)  # nettoyage des valeurs aberrantes
        info = {}
        return obs, info

    def step(self, action):
        # 1. Extraire vitesse linéaire et angulaire
        linear_vel = float(action[0])
        angular_vel = float(action[1])

        # 2. Publier la commande
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_vel
        cmd_msg.angular.z = angular_vel
        self.cmd_pub.publish(cmd_msg)

        # 3. Attendre nouvelle observation
        rclpy.spin_once(self.node)

        # 4. Calculer l'observation
        obs = np.clip(self.scan, 0.0, 10.0)

        # 5. Détecter collision (si un rayon < seuil)
        collision_threshold = 0.3  # mètres
        self.collision = np.any(obs < collision_threshold)

        # 6. Calculer le reward
        if self.collision:
            reward = -10.0
            terminated = True
        else:
            reward = +1.0
            terminated = False
        truncated = False

        # 7. Optionnel : info log/debug
        info = {}

        return obs, reward, terminated, truncated, info
