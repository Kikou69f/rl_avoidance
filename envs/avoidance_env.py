import rclpy
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty  # pour les services de reset et start

class AvoidanceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # --- Constantes du robot ---
        self.WHEEL_RADIUS = 0.195 / 2       # 19.5 cm
        self.ROBOT_WIDTH  = 0.33            # 33 cm hors roues

        # Initialisation ROS2
        rclpy.init(args=None)
        self.node = rclpy.create_node("avoidance_env_node")

        # Clients pour les services /reset_simulation et /start_simulation
        self.reset_cli = self.node.create_client(Empty, '/reset_simulation')
        self.start_cli = self.node.create_client(Empty, '/start_simulation')
        if not self.reset_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Le service /reset_simulation n'est pas disponible")
        if not self.start_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Le service /start_simulation n'est pas disponible")

        # Drapeau pour ignorer le premier reset
        self._first_reset = True

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
        self.initial_distance = None  # Nouveau: distance initiale au but

        # Goal fixe
        self.goal = (6.0, -3.0, 0.0)

        # Subs & pub
        self.pose_sub = self.node.create_subscription(Pose, "/pose", self.pose_callback, 10)
        self.scan_sub = self.node.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)

        # Attente du 1er scan
        while rclpy.ok() and self.scan is None:
            rclpy.spin_once(self.node)

        # Espaces Gym
        N = len(self.scan)
        self.observation_space = spaces.Box(
            low=0.0, high=self.range_max,
            shape=(N + 3,), dtype=np.float32
        )
        # action_space avec vx ≥ 0
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def pose_callback(self, msg: Pose):
        self.x = msg.position.x
        self.y = msg.position.y
        self.robot_orientation = msg.orientation.z
        
        # Calculer la distance initiale au premier appel
        if self.initial_distance is None:
            dx = self.goal[0] - self.x
            dy = self.goal[1] - self.y
            self.initial_distance = math.hypot(dx, dy)

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        self.range_max = msg.range_max
        ranges[np.isinf(ranges) | (ranges <= 0.0)] = self.range_max
        self.num_beams = len(ranges)
        self.angles = msg.angle_min + np.arange(self.num_beams) * msg.angle_increment

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.prev_scan_time = self.scan_time
        self.scan_time = t

        self.scan = ranges

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Stop & restart de la simulation après le premier épisode
        if not self._first_reset:
            req = Empty.Request()
            # stop
            fut = self.reset_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut)
            # start
            fut2 = self.start_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut2)
        else:
            self._first_reset = False

        # Réinitialisation des variables internes
        self.prev_distance = None
        self.stuck_counter = 0

        # Attente d'un nouveau scan après reset
        self.scan = None
        while rclpy.ok() and self.scan is None:
            rclpy.spin_once(self.node)

        # Stoppe le robot
        self.cmd_pub.publish(Twist())

        # Construction de l'observation initiale
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        target_heading = math.atan2(dy, dx)
        heading_error = abs((target_heading - self.robot_orientation + math.pi) % (2*math.pi) - math.pi)

        # Calcul de la distance initiale si nécessaire
        if self.initial_distance is None:
            self.initial_distance = math.hypot(dx, dy)

        obs = np.concatenate([
            np.clip(self.scan, 0.0, self.range_max),
            [dx, dy, heading_error]
        ]).astype(np.float32)
        return obs, {"goal": self.goal}

    def step(self, action):
        if not rclpy.ok():
            raise RuntimeError("ROS context closed")

        # Envoi de la commande
        cmd = Twist()
        cmd.linear.x  = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)

        # Lecture du scan
        rclpy.spin_once(self.node)
        scan_vals = np.clip(self.scan, 0.0, self.range_max)

        # 1) position et heading
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        dist = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_error = abs((target_heading - self.robot_orientation + math.pi) % (2*math.pi) - math.pi)

        # 2) progression
        if self.prev_distance is None:
            delta_dist = 0.0
        else:
            delta_dist = self.prev_distance - dist
        self.prev_distance = dist

        # Gestion du stuck counter - réinitialisation immédiate au moindre mouvement
        if abs(delta_dist) > 0.01:  # Seuil très bas pour détecter tout mouvement
            self.stuck_counter = 0  # Réinitialisation immédiate
        else:
            self.stuck_counter += 1

        # Calcul de la progression globale
        progress = (self.initial_distance - dist) / self.initial_distance

        # Reward principal (nouveau système)
        reward = 0.0
        
        # 1. RÉCOMPENSE PRINCIPALE FORTEMENT AUGMENTÉE
        # Récompense immédiate pour la réduction de distance
        reward += delta_dist * 50  # Très forte récompense pour chaque mètre gagné
        
        # 2. Récompense basée sur la progression globale
        reward += progress * 10  # Récompense supplémentaire importante
        
        # 3. Orientation
        reward += math.cos(heading_error) * 10  # Récompense pour être bien orienté
        
        # Pénalité temporelle minimale
        reward -= 0.01
        
        # 4. Récompense de proximité finale (boost quand proche du but)
        if dist < 2.0:
            reward += (2.0 - dist) * 20  # Bonus progressif plus important

        # 5. PÉNALITÉ D'OBSTACLE PROGRESSIVE (à partir de 2.5m)
        min_distance = np.min(scan_vals)
        if min_distance < 2.5:
            # Pénalité progressive : plus on est proche, plus la pénalité est forte
            penalty_factor = (2.5 - min_distance) / 2.5
            # Pénalité quadratique pour renforcer l'effet quand très proche
            penalty = -30 * (penalty_factor ** 2)
            reward += penalty
            # Ajout d'une pénalité supplémentaire pour les très courtes distances
            if min_distance < 0.5:
                reward -= 15 * (0.5 - min_distance)

        # 6. Pénalité d'immobilisme (seulement si vraiment bloqué)
        #if self.stuck_counter > 500:  # Seuil élevé (3 secondes d'immobilité)
        #    reward -= 1.0 * (self.stuck_counter - 500)  # Pénalité progressive

        terminated = False
        success = False

        # Détection de collision
        lidar_collision = any(
            (-math.pi/2 <= ang <= math.pi/2 and 0.01 < d < 0.4)
            for d, ang in zip(scan_vals, self.angles)
        )
        if lidar_collision:
            terminated = True
            reward -= 30  # Pénalité de collision

        # Succès d'arrivée au but
        if not terminated and dist < 0.4:
            reward += 30.0
            terminated = True
            success = True
            if heading_error < 0.3:
                reward += 20.0
            else:
                reward -= 10.0

        truncated = False
        obs = np.concatenate([scan_vals, [dx, dy, heading_error]]).astype(np.float32)
        info = {
            "goal": self.goal,
            "distance": dist,
            "heading_error": heading_error,
            "stuck_counter": self.stuck_counter,
            "lidar_collision": lidar_collision,
            "success": success
        }

        print(f"[STEP] reward={reward:.2f}, dist={dist:.2f}, "
              f"heading_err={heading_error:.2f}, stuck={self.stuck_counter}, "
              f"lidar_coll={lidar_collision}, success={success}")
        return obs, reward, terminated, truncated, info