"""ROS2 IMPORTS"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from mocap4r2_msgs.msg import RigidBodies
from geometry_msgs.msg import TwistStamped
from tf_transformations import euler_from_quaternion
from controller_real_world.msg import CommitmentState
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


"""PYTHON IMPORTS"""
import os
import math
import json
import random
import threading
import numpy as np

#======================define global pose variable=========================
pos_message_g = {}
pos_lock = threading.Lock()
#==========================================================================

#======================define parameters=============================
package_name = 'controller_real_world'

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
        self.soft_turn_threshold = float(parameters.get("soft_turn_threshold", 0.08727)) # radians (~5°)
        self.hard_turn_threshold = float(parameters.get("hard_turn_threshold", 0.17453)) # radians (~10°)
        self.goal_tolerance = float(parameters.get("goal_tolerance", 0.5))
        self.kp_angle = float(parameters.get("kp_angle", 0.5)) # Proportional gain for angle correction # radians (~28.65°)

opt = Options()
#==================================================================

#====================== QoS Profile for Commitment States ======================

commitment_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
#===============================================================================
                
#====================== Controller Node ======================

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node_' + opt.id)

        # --------- Parameters ---------
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
        self.soft_turn_threshold = opt.soft_turn_threshold
        self.kp_angle = opt.kp_angle  # Proportional gain for angle correction
        #------------------------------

        # --- State ---
        self.yaw = 0.0
        self.pos_message = {}

        # Commitment quality (0 to 1)
        self.quality = 1.0
        # -------------

        # --- ROS Interfaces ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub = self.create_publisher(CommitmentState, 'commitments', qos)
        self.robot_id = robot_id
        self.seq = 0
        # Publish commitment state every second (heartbeat)
        self.timer_pub = self.create_timer(1.0, self.publish_commitment_state)
        # Listen to other robots' commitments/opinions
        self.sub = self.create_subscription(
            CommitmentState,
            'commitments',
            self.listener_cb,
            qos
        )


        self.odom_sub = self.create_subscription(
            RigidBodies,
            "/rigid_bodies",
            self.odom_callback,
            10)
        
        self.cmd_pub = self.create_publisher(TwistStamped, '/turtlebot4_3/cmd_vel', 10)
        # Moves 0.022 meters (2.2 cm) per update at 10 Hz 
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"Controller node started. Target commitment: {self.target_commitment}")

    def publish_state(self):
        msg = CommitmentState()
        msg.robot_id = self.id
        msg.stamp = self.get_clock().now().to_msg()
        msg.seq = self.seq
        msg.commitment = self.commitment
        msg.quality = self.quality
        self.pub.publish(msg)
        self.get_logger().debug(f'Published commitment {msg.commitment}')
        self.seq += 1

    def listener_cb(self, msg: CommitmentState):
        self.commitments[msg.robot_id] = msg
        self.get_logger().info(
            f'[{self.get_name()}] {msg.robot_id} committed to {msg.commitment}'
        )
        
    def odom_callback(self, msg: RigidBodies):
        """Update robot pose from odometry."""
        for rb in msg.rigidbodies:
            if rb.rigid_body_name == self.id:
                # Update the current position with the received data
                self.pos_message['self'] = rb.pose
            else:
                self.pos_message[rb.rigid_body_name] = rb.pose


        # Quaternion -> yaw
        q = self.pos_message['self'].orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def control_loop(self):
        """Control loop: hard-turn if needed, else drive straight."""
        if not self.pos_message or 'self' not in self.pos_message or self.target_commitment not in self.pos_message:
            self.get_logger().warn("Waiting for valid position data...")
            self.get_logger().warn(f"Current pos_message keys: {list(self.pos_message.keys())}")
            return
        dx = self.pos_message[self.target_commitment].position.x - self.pos_message['self'].position.x 
        dy = self.pos_message[self.target_commitment].position.y - self.pos_message['self'].position.y 
        distance = math.hypot(dx, dy)

        # Stop if goal reached
        if distance < self.goal_tolerance:
            self.stop_robot()
            self.get_logger().info("Goal reached!")
            return

        # Desired heading
        target_angle = math.atan2(dy, dx) - math.pi/2
        angle_error = self.wrap_angle(target_angle - self.yaw)
        self.get_logger().info(f"Current target commitment: {self.target_commitment}, Current yaw: {math.degrees(self.yaw):.2f}° Distance to target: {distance:.2f}, Angle to target: {math.degrees(target_angle):.2f}°, Angle error: {math.degrees(angle_error):.2f}°")

        msg = TwistStamped()
        msg.header = Header()
        msg.header.frame_id = 'base_link'  # Set the frame_id
        msg.header.stamp = self.get_clock().now().to_msg()  # Set the timestamp
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0

        if abs(angle_error) > self.hard_turn_threshold:
            # Turn in place for very large errors
            msg.twist.angular.z = self.angular_speed * (1 if angle_error > 0 else -1)
            msg.twist.linear.x = 0.0
        elif abs(angle_error) < self.hard_turn_threshold and abs(angle_error) > self.soft_turn_threshold:
            # Curve while moving
            msg.twist.linear.x = self.linear_speed
            # Proportional controller for angular velocity
            msg.twist.angular.z = msg.twist.angular.z = max(-self.angular_speed,
                                                            min(self.kp_angle * angle_error, self.angular_speed))

        else:
            # Go mostly straight
            msg.twist.linear.x = self.linear_speed
            msg.twist.angular.z = 0.0


        self.cmd_pub.publish(msg)
        self.get_logger().info(f"Published cmd_vel: linear.x={msg.twist.linear.x}, angular.z={msg.twist.angular.z}")

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

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


def main(args=None):
    rclpy.init()
    controller_node = ControllerNode()
    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
