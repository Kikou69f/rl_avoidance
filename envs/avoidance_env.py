import rclpy
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose

class AvoidanceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Initialisation ROS2
        rclpy.init(args=None)
        self.node = rclpy.create_node("avoidance_env_node")

        # États internes
        self.scan = None
        self.range_max = None
        self.x = 0.0
        self.y = 0.0
        self.robot_orientation = 0.0
        self.prev_distance = None
        self.stuck_counter = 0
        self.scan_time = None
        self.prev_scan_time = None

        # Goal fixe
        self.goal = (6.0, -3.0, 0.57)

        # Subscriptions & publisher
        self.pose_sub = self.node.create_subscription(
            Pose, "/pose", self.pose_callback, 10
        )
        self.scan_sub = self.node.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.cmd_pub = self.node.create_publisher(
            Twist, "/cmd_vel", 10
        )

        # Attente des premières données
        while rclpy.ok() and self.scan is None:
            rclpy.spin_once(self.node)

        # Espaces Gym
        N = len(self.scan)
        self.observation_space = spaces.Box(
            low=0.0,
            high=self.range_max,
            shape=(N + 3,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-0.2, -1.0], dtype=np.float32),
            high=np.array([ 0.5,  1.0], dtype=np.float32),
            dtype=np.float32
        )

    def pose_callback(self, msg: Pose):
       
        self.x = msg.position.x
        self.y = msg.position.y
        self.robot_orientation = msg.orientation.z

    def scan_callback(self, msg: LaserScan):
        # Conversion et filtrage des ranges
        ranges = np.array(msg.ranges, dtype=np.float32)
        self.range_max = msg.range_max
        ranges[np.isinf(ranges) | (ranges <= 0.0)] = self.range_max

        # Calcul des angles 
        self.num_beams = len(ranges)
        self.angles = msg.angle_min + np.arange(self.num_beams) * msg.angle_increment

        # Ignorer les scans latéraux
        mask = (
            (self.angles >= -3*math.pi/4) & (self.angles < -math.pi/2)
        ) | (
            (self.angles >  math.pi/2) & (self.angles <= 3*math.pi/4)
        )
        ranges[mask] = self.range_max

        # Gestion des timestamps pour Δt réel
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.scan_time is not None:
            self.prev_scan_time = self.scan_time
        self.scan_time = t

        self.scan = ranges

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_distance = None
        self.stuck_counter = 0

        # Attente d'un nouveau scan
        self.scan = None
        while rclpy.ok() and self.scan is None:
            rclpy.spin_once(self.node)
        if not rclpy.ok():
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        # Stop du robot au reset
        self.cmd_pub.publish(Twist())

        # Calcul des deltas initiaux
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        target_heading = math.atan2(dy, dx)
        heading_error = abs((target_heading - self.robot_orientation + math.pi)
                            % (2*math.pi) - math.pi)

        # Observation initiale
        obs = np.concatenate([
            np.clip(self.scan, 0.0, self.range_max),
            [dx, dy, heading_error]
        ]).astype(np.float32)
        return obs, {"goal": self.goal}

    def step(self, action):
        if not rclpy.ok():
            raise RuntimeError("ROS context closed")

        # Publication de la commande
        cmd = Twist()
        cmd.linear.x  = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)

        # Spin et récupération du scan
        rclpy.spin_once(self.node)
        scan_vals = np.clip(self.scan, 0.0, self.range_max)

        # 1) Deltas positionnels
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        dist = math.hypot(dx, dy)

        # 2) Erreur de cap
        target_heading = math.atan2(dy, dx)
        heading_error = abs((target_heading - self.robot_orientation + math.pi)
                            % (2*math.pi) - math.pi)

        # 3) Progression
        if self.prev_distance is None:
            delta_dist = 0.0
        else:
            delta_dist = self.prev_distance - dist
        self.prev_distance = dist

        # Détection de stuck
        if abs(delta_dist) < 0.01 and abs(action[0]) > 0.05:
            self.stuck_counter += 1
        elif abs(delta_dist) >= 0.01:
            self.stuck_counter = max(0, self.stuck_counter - 2)
        if abs(action[0]) < 0.05 and abs(action[1]) < 0.05:
            self.stuck_counter += 2

        # Calcul de Δt réel
        if self.prev_scan_time is None:
            dt = 0.05
        else:
            dt = self.scan_time - self.prev_scan_time
            if dt <= 0.0:
                dt = 0.05

        # Reward
        reward = 0.0
        # a) bonus vitesse effective
        effective_speed = delta_dist / dt
        reward += effective_speed * 15

        # b) récompense d’alignement si avance
        if action[0] > 0.05:
            heading_reward = (1.0 - heading_error / math.pi) * 7
            if heading_error > 0.9:
                penalty = min(3.0, 2.0 + 2.0 * (heading_error - 0.9))
                heading_reward /= penalty
            reward += heading_reward

        # c) pénalité par pas renforcée
        reward -= 0.02

        # d) bonus proximité
        if dist < 2.0:
            reward += (2.0 - dist) * 25

        # Conditions de fin & collision
        terminated = False
        success = False
        lidar_collision = any(
            (-math.pi/2 <= ang <= math.pi/2 and 0.01 < d < 0.4)
            for d, ang in zip(scan_vals, self.angles)
        )
        if lidar_collision:
            reward = -10.0
            terminated = True

        # Succès en deux étapes
        if not terminated and dist < 0.4:
            reward += 50.0
            terminated = True
            success = True
            if heading_error < 0.3:
                reward += 50.0
                print("SUCCESS: Goal reached with correct orientation")
            else:
                reward -= 10.0
                print("PARTIAL SUCCESS: Goal reached but bad orientation")

        # Pénalité stuck prolongé
        if not terminated and self.stuck_counter > 30:
            reward -= 5.0

        truncated = False

        # Observation & info
        obs = np.concatenate([scan_vals, [dx, dy, heading_error]]).astype(np.float32)
        info = {
            "goal": self.goal,
            "distance": dist,
            "heading_error": heading_error,
            "stuck_counter": self.stuck_counter,
            "lidar_collision": lidar_collision,
            "success": success
        }

        # Debug print par étape
        print(f"[STEP] reward={reward:.2f}, dist={dist:.2f}, heading_err={heading_error:.2f}, "
              f"stuck={self.stuck_counter}, lidar_coll={lidar_collision}, success={success}")

        return obs, reward, terminated, truncated, info
