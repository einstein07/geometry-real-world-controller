"""ROS2 IMPORTS"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
from mocap4r2_msgs.msg import RigidBodies
from ament_index_python.packages import get_package_share_directory

"""PYTHON IMPORTS"""
import math
import threading
import os
import json
import random

#======================define global pose variable=========================
pos_message_g = {}
pos_lock = threading.Lock()
#==========================================================================

#======================define parameters=============================
package_name = 'ringattractor'

param_file = os.path.join(
    get_package_share_directory(package_name),
    'parameters.json'
)

with open(param_file, 'r') as f:
    parameters = json.load(f)

class Options():
    def __init__(self):
        
        self.id = parameters["id"]
        
        self.linear_speed = float(parameters["linear_speed"])
        self.angular_speed = float(parameters["angular_speed"]) # radians (~5.73°)

        self.targets = parameters.get("targets", [])
        self.hard_turn_threshold = float(parameters.get("hard_turn_threshold", 0.5)) # radians (~28.65°)
        self.goal_tolerance = float(parameters.get("goal_tolerance", 0.1))

opt = Options()
#==================================================================

                
#====================== Controller Node ======================

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        self.id = opt.id
        # Pick a random target commitment from the list (if not empty)
        if opt.targets:
            self.target_commitment = random.choice(opt.targets)
        else:
            self.target_commitment = None  # or handle default
            print("No targets available in parameters.")
            return
        
        self.linear_speed = opt.linear_speed
        self.angular_speed = opt.angular_speed
        self.goal_tolerance = opt.goal_tolerance
        self.hard_turn_threshold = opt.hard_turn_threshold

        # --- State ---
        self.yaw = 0.0
        self.pos_message = {}

        # --- ROS Interfaces ---
        self.odom_sub = self.create_subscription(
            RigidBodies,
            "/rigid_bodies",
            self.odom_callback,
            10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Moves 0.022 meters (2.2 cm) per update at 10 Hz 
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"Controller node started. Target commitment: {self.target_commitment}")

    def odom_callback(self, msg):
        """Update robot pose from odometry."""
        for i in range(msg.n):
            if msg.rigidbodies[i].rigid_body_name == self.id:
                # Update the current position with the received data
                self.pos_message['self'] = msg.rigidbodies[i].pose
            else:
                self.pos_message[msg.rigidbodies[i].rigid_body_name] = msg.rigidbodies[i].pose


        # Quaternion -> yaw
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def control_loop(self):
        """Control loop: hard-turn if needed, else drive straight."""
        if not self.pos_message or 'self' not in self.pos_message or self.target_commitment not in self.pos_message:
            self.get_logger().warn("Waiting for valid position data...")
            return
        dx = self.pos_message[self.target_commitment].pose.position.x - self.pos_message['self'].pose.position.x 
        dy = self.pos_message[self.target_commitment].pose.position.y - self.pos_message['self'].pose.position.y 
        distance = math.hypot(dx, dy)

        # Stop if goal reached
        if distance < self.goal_tolerance:
            self.stop_robot()
            self.get_logger().info("Goal reached!")
            return

        # Desired heading
        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.yaw)

        twist = Twist()

        if abs(angle_error) > self.hard_turn_threshold:
            # Hard turn in place
            twist.angular.z = self.angular_speed * (1 if angle_error > 0 else -1)
            self.get_logger().debug("Hard turning...")
        else:
            # Go straight towards target
            twist.linear.x = self.linear_speed
            self.get_logger().debug("Moving straight...")

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """Publish zero velocities."""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def normalize_angle(angle):
        """Keep angle between -pi and pi."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init()
    controller_node = ControllerNode()
    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
