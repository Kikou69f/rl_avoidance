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

        # -Initialisation de ROS2 et du Node 
        rclpy.init(args=None)
        self.node = rclpy.create_node("avoidance_env_node")

        # -Variables internes
        self.scan = None                # dernières mesures Lidar
        self.range_max = None           # portée max du capteur
        self.pose = None                # position du robot
        self.robot_orientation = 0.0    # cap (yaw) du robot
        self.prev_distance = None       # distance au goal à l'étape précédente
        self.stuck_counter = 0          # compteur de blocage (immobilité)

        # -Goal fixe (x, y, θ) 
        self.goal = (6.0, -3.0, 1.57)

        # -Subscriptions et publication ROS 
        self.scan_sub = self.node.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.pose_sub = self.node.create_subscription(
            Pose, "/pose", self.pose_callback, 10
        )
        self.cmd_pub = self.node.create_publisher(
            Twist, "/cmd_vel", 10
        )

        # -Attente de la première mesure Lidar et pose 
        while rclpy.ok() and (self.scan is None or self.pose is None):
            rclpy.spin_once(self.node)

        # -Définition des espaces Gym 
        # Observation : N mesures Lidar + dx, dy, heading_error
        N = len(self.scan)
        self.observation_space = spaces.Box(
            low=0.0,
            high=self.range_max,
            shape=(N + 3,),
            dtype=np.float32
        )
        # Action : vitesse linéaire x et vitesse angulaire z
        self.action_space = spaces.Box(
            low=np.array([-0.2, -1.0]),
            high=np.array([ 0.5,  1.0]),
            dtype=np.float32
        )

    def scan_callback(self, msg: LaserScan):
        """Callback pour mettre à jour et filtrer les lectures Lidar."""
        ranges = np.array(msg.ranges, dtype=np.float32)
        self.range_max = msg.range_max

        # Remplace inf et valeurs ≤0 par la portée max
        ranges[np.isinf(ranges)] = self.range_max
        ranges[ranges <= 0.0] = self.range_max

        # Calcul des angles associé à chaque rayon (une seule fois)
        if not hasattr(self, 'angles'):
            self.num_beams = len(ranges)
            self.angles = np.linspace(
                msg.angle_min, msg.angle_max,
                self.num_beams, dtype=np.float32
            )

        # Filtrage des angles latéraux [-135°, -90°) & (90°, 135°]
        ignore_mask = (
            (self.angles >= -3*math.pi/4) & (self.angles < -math.pi/2)
        ) | (
            (self.angles >  math.pi/2) & (self.angles <= 3*math.pi/4)
        )
        ranges[ignore_mask] = self.range_max

        # Stocke les données de scan filtrées
        self.scan = ranges

    def pose_callback(self, msg: Pose):
        """Callback pour mettre à jour la position et l orientation du robot."""
        self.pose = msg.position
        x, y, z, w = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        # Calcul du yaw (cap) à partir du quaternion
        self.robot_orientation = math.atan2(
            2.0 * (w*z + x*y),
            1.0 - 2.0 * (y*y + z*z)
        )

    def reset(self, *, seed=None, options=None):
        """Réinitialise l environnement et renvoie l observation initiale."""
        super().reset(seed=seed)
        self.prev_distance = None
        self.stuck_counter = 0

        # Attente de nouvelles mesures
        self.scan = None
        self.pose = None
        while rclpy.ok() and (self.scan is None or self.pose is None):
            rclpy.spin_once(self.node)
        if not rclpy.ok():
            # En cas de shutdown ROS
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        # Arrête le robot au départ
        self.cmd_pub.publish(Twist())

        # Calcul des deltas initiaux
        dx = self.goal[0] - self.pose.x
        dy = self.goal[1] - self.pose.y
        target_heading = math.atan2(dy, dx)
        heading_error = abs((target_heading - self.robot_orientation + math.pi) % (2*math.pi) - math.pi)

        # Observation initiale
        obs = np.concatenate([
            np.clip(self.scan, 0.0, self.range_max),
            [dx, dy, heading_error]
        ]).astype(np.float32)
        return obs, {"goal": self.goal}

    def step(self, action):
        """Applique l action, calcule le reward, détecte collision/succès et renvoie (obs, reward, done, truncated, info)."""
        # Publication de la commande de vitesse
        if rclpy.ok():
            cmd = Twist()
            cmd.linear.x  = float(action[0])
            cmd.angular.z = float(action[1])
            self.cmd_pub.publish(cmd)
        else:
            raise RuntimeError("ROS context closed")

        # Récupération des nouvelles mesures
        rclpy.spin_once(self.node)
        scan_vals = np.clip(self.scan, 0.0, self.range_max)

        # 1) Calcul des deltas positionnels
        dx = self.goal[0] - self.pose.x
        dy = self.goal[1] - self.pose.y
        dist = math.hypot(dx, dy)

        # 2) Calcul de l’erreur de cap (heading_error)
        target_heading = math.atan2(dy, dx)
        heading_error = abs((target_heading - self.robot_orientation + math.pi) % (2*math.pi) - math.pi)

        # 3) Calcul de la progression
        if self.prev_distance is None:
            delta_dist = 0.0
        else:
            delta_dist = self.prev_distance - dist
        self.prev_distance = dist

        # Détection améliorée de stuck (blocage) 
        if abs(delta_dist) < 0.01 and abs(action[0]) > 0.05:
            # Le robot essaie d’avancer mais ne progresse pas
            self.stuck_counter += 1
        elif abs(delta_dist) >= 0.01:
            # Le robot progresse, on décrémente le compteur
            self.stuck_counter = max(0, self.stuck_counter - 2)
        # Pénalité supplémentaire si le robot est totalement arrêté
        if abs(action[0]) < 0.05 and abs(action[1]) < 0.05:
            self.stuck_counter += 2

        # Reward 
        reward = 0.0

        # a) Bonus proportionnel à la vitesse effective vers le goal
        effective_speed = delta_dist / 0.05  # ~0.05s par step
        reward += effective_speed * 8

        # b) Récompense d’alignement 
        heading_reward = (1.0 - heading_error / math.pi) * 10
        # Pénalité si l’erreur très grande
        if heading_error > 0.9:
            penalty = min(3.0, 2.0 + 2.0 * (heading_error - 0.9))
            heading_reward /= penalty
        reward += heading_reward

        # c) Petite pénalité constante pour encourager l’efficacité
        reward -= 0.005

        # d) Bonus intermédiaire lorsque proche du goal (<2m)
        if dist < 2.0:
            reward += (2.0 - dist) * 10

        #  Conditions de fin & détection collision / succès 
        terminated = False
        success = False

        # i) Détection de collision Lidar frontale fiable
        d_inf = 0.4
        lidar_collision = False
        for i, d in enumerate(scan_vals):
            angle = self.angles[i]
            if -math.pi/2 <= angle <= math.pi/2 and 0.01 < d < d_inf:
                lidar_collision = True
                break
        if lidar_collision:
            reward = -10.0
            terminated = True

        # ii) Nouvelle condition de succès en deux étapes
        #   - Étape 1 : position atteinte
        if not terminated and dist < 0.4:
            reward += 50.0
            terminated = True
            success = True
            #   - Étape 2 : orientation 
            if heading_error < 0.3:
                reward += 50.0
                print("SUCCESS: Goal reached with correct orientation")
            else:
                reward -= 10.0
                print("PARTIAL SUCCESS: Goal reached but bad orientation")

        # iii) Pénalité pour blocage prolongé
        if not terminated and self.stuck_counter > 50:
            reward -= 5.0

        truncated = False

        # Observation 
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
