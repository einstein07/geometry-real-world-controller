"""ROS2 IMPORTS"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TwistStamped, Twist, Pose
from tf_transformations import euler_from_quaternion, quaternion_from_matrix
from controller_msgs.msg import CommitmentState
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

"""PYTHON IMPORTS"""
import os
import csv
import math
import json
import random
import datetime
import threading
import numpy as np
import asyncio
import qtm_rt as qtm
import xml.etree.ElementTree as ET


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
        self.robot_namespace = parameters.get("robot_namespace")
        
        self.linear_speed = float(parameters["linear_speed"])
        self.angular_speed = float(parameters["angular_speed"]) # radians (~5.73°)

        self.targets = parameters.get("targets", [])
        self.soft_turn_threshold = float(parameters.get("soft_turn_threshold", 0.08727)) # radians (~5°)
        self.hard_turn_threshold = float(parameters.get("hard_turn_threshold", 0.17453)) # radians (~10°)
        self.goal_tolerance = float(parameters.get("goal_tolerance", 0.5))
        self.kp_angle = float(parameters.get("kp_angle", 0.5)) # Proportional gain for angle correction # radians (~28.65°)
        self.qtm_ip = parameters.get("qtm_ip", "134.34.231.207")  # Add QTM server IP to parameters

        self.update_rate = int(parameters.get("update_rate", 10)) # time steps
        self.eta = float(parameters.get("eta", 0.1)) # weight for neighbor influence

        self.base_log_dir = os.path.expanduser(parameters.get('log_directory', '~/geometry-logs'))
        #self.base_log_dir = parameters.get('log_directory', '~/geometry-logs')
        os.makedirs(self.base_log_dir, exist_ok=True)         # If directory does not exist, create it
        self.experiment_name = parameters.get('experiment_name', 'experiment')

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
        self.update_rate = opt.update_rate   # time steps
        self.counter = random.randint(0, self.update_rate)
        self.eta = opt.eta # weight for neighbor influence
        self.id = opt.id
        # Pick a random target commitment from the list (if not empty)
        if opt.targets:
            self.target_commitment = random.randrange(len(opt.targets)) +1  # +1 to make sure not starting with 0
        else:
            self.target_commitment = 0  # or handle default
            print("No targets available in parameters.")
            return
        self.publishable_commitment = self.target_commitment
        
        self.linear_speed = opt.linear_speed
        self.angular_speed = opt.angular_speed
        self.goal_tolerance = opt.goal_tolerance
        self.hard_turn_threshold = opt.hard_turn_threshold
        self.soft_turn_threshold = opt.soft_turn_threshold
        self.kp_angle = opt.kp_angle  # Proportional gain for angle correction
        self.qtm_ip = opt.qtm_ip  # QTM server IP
        #------------------------------

        # --- State ---
        self.pos_message = {}
        self.pos_lock = threading.Lock()
        self.rb_names = []
        self.commitments = {}
        self.commitment = self.target_commitment
        self.my_opinions = []
        self.quality = 1.0
        # -------------

        # ----- Logging -----
        self.base_log_dir = opt.base_log_dir
        self.experiment_name = opt.experiment_name
        self.get_logger().info(f"Logging data to: {self.base_log_dir}, Experiment name: {self.experiment_name}")
        # -------------------

        # --- ROS Interfaces ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub = self.create_publisher(CommitmentState, 'commitments', qos)
        self.robot_id = self.id
        self.seq = 0
        
        # Listen to other robots' commitments/opinions
        self.sub = self.create_subscription(
            CommitmentState,
            'commitments',
            self.listener_cb,
            qos
        )
        self.robot_namespace = opt.robot_namespace

        ns = (self.robot_namespace or "").strip()

        # remove all leading/trailing slashes so we control formatting
        ns = ns.strip("/")

        # build a prefix: "" (no namespace) or "/<ns>"
        prefix = f"/{ns}" if ns else ""

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            f"{prefix}/cmd_vel",
            10
        )
        # Moves 0.022 meters (2.2 cm) per update at 10 Hz 
        self.timer = self.create_timer(0.1, self.control_loop)

        # Setup QTM connection in a separate thread
        self._loop = asyncio.new_event_loop()
        self._connection = None
        self._thread = threading.Thread(target=self._run_rt, daemon=True)
        self._thread.start()
        # ----------------------

        # ----------- Initialize log files -----------
        run_folder = os.path.join(self.base_log_dir, self.experiment_name)
        self.initialize_opinions_log()
        self.initialize_position_log()    
        # ---------------------------------------------

        self.get_logger().info(f"Controller node started. Target commitment: {opt.targets[self.target_commitment-1]}")

    def _run_rt(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._rt_protocol())
        self._loop.run_forever()

    async def _rt_protocol(self):
        try:
            self._connection = await qtm.connect(self.qtm_ip, version="1.22")
            if self._connection is None:
                self.get_logger().error(f"Failed to connect to QTM server at {self.qtm_ip}")
                return

            xml_string = await self._connection.get_parameters(parameters=["6d"])
            if not xml_string:
                self.get_logger().error("Failed to retrieve 6D parameters from QTM")
                return

            root = ET.fromstring(xml_string)
            self.rb_names = [body.find("Name").text for body in root.iter("Body")]
            self.get_logger().info(f"Found rigid bodies: {self.rb_names}")

            await self._connection.stream_frames(components=["6d"], on_packet=self._on_packet)
        except Exception as e:
            self.get_logger().error(f"Error in QTM connection: {str(e)}")

    def _on_packet(self, packet):
        try:
            header, rbs = packet.get_6d()
            if not rbs:
                self.get_logger().warn("No 6D rigid body data received")
                return

            with self.pos_lock:
                temp = {}
                for i, ((x, y, z), rotation) in enumerate(rbs):
                    if i >= len(self.rb_names):
                        continue
                    name = self.rb_names[i]
                    pose = Pose()
                    pose.position.x = x / 1000.0  # Convert mm to m
                    pose.position.y = y / 1000.0
                    pose.position.z = z / 1000.0

                    # Convert rotation matrix to quaternion
                    matrix = np.reshape(np.array(rotation.matrix), (3, 3))
                    homogeneous = np.eye(4)
                    homogeneous[:3, :3] = matrix
                    q = quaternion_from_matrix(homogeneous)
                    pose.orientation.x = float(q[0])
                    pose.orientation.y = float(q[1])
                    pose.orientation.z = float(q[2])
                    pose.orientation.w = float(q[3])

                    if name == self.id:
                        temp['self'] = pose
                    else:
                        temp[name] = pose

                self.pos_message = temp
        except Exception as e:
            self.get_logger().error(f"Error processing QTM packet: {str(e)}")

    def publish_commitment_state(self):
        msg = CommitmentState()
        msg.robot_id = self.id
        msg.stamp = self.get_clock().now().to_msg()
        msg.seq = self.seq
        msg.commitment = self.publishable_commitment
        msg.quality = self.quality
        self.pub.publish(msg)
        self.get_logger().debug(f'Published commitment {msg.commitment}')
        self.seq += 1

    def listener_cb(self, msg: CommitmentState):
        if msg.robot_id == self.id:
            return  # Ignore own messages
        self.commitments[msg.robot_id] = msg.commitment
        #self.get_logger().info(
        #    f'[{self.get_name()}] {msg.robot_id} committed to {msg.commitment}'
        #)

    def update_target_commitment(self):
        if self.counter % self.update_rate == 0:
            if random.random() < self.eta:
                self.target_commitment = random.randrange(len(opt.targets))
            else:
                # Pick random neighbor's commitment
                if self.commitments:
                    neighbor = random.choice(list(self.commitments.values()))
                    # Check if value is not 0, if 0 keep own commitment
                    if neighbor != 0:    
                        self.target_commitment = neighbor
                # Clear commitments to avoid bias
                self.commitments = {}
            
    def update_robot_movement(self):
        """Control loop: hard-turn if needed, else drive straight."""
        with self.pos_lock:
            if not self.pos_message or 'self' not in self.pos_message or opt.targets[self.target_commitment-1] not in self.pos_message:
                self.get_logger().warn("Waiting for valid position data...")
                self.get_logger().warn(f"Current pos_message keys: {list(self.pos_message.keys())}")
                return
            pos_message_copy = self.pos_message.copy()

        dx = pos_message_copy[opt.targets[self.target_commitment-1]].position.x - pos_message_copy['self'].position.x 
        dy = pos_message_copy[opt.targets[self.target_commitment-1]].position.y - pos_message_copy['self'].position.y 
        distance = math.hypot(dx, dy)

        # Stop if goal reached
        if distance < self.goal_tolerance:
            self.stop_robot()
            self.get_logger().info("Goal reached!")
            # Destroy node and shutdown ROS cleanly
            self.destroy_node()
            rclpy.shutdown()
            return

        # Compute current yaw
        q = pos_message_copy['self'].orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        """CCW rotations give negative yaw → Qualisys yaw is 
        opposite sign to the atan2 convention), convert by negating:"""
        yaw = -yaw

        # Desired heading
        target_angle = math.atan2(dy, dx)
        angle_error = self.wrap_angle(target_angle - yaw)
        # Break ties at ±180° by slightly preferring one direction
        if abs(abs(angle_error) - np.pi) < 0.05:
            angle_error = -np.pi + 0.1  # Always turn right when a,biguous
        #self.get_logger().info(f"Current target commitment: {opt.targets[self.target_commitment]}, Current yaw: {math.degrees(yaw):.2f}° Distance to target: {distance:.2f}, Angle to target: {math.degrees(target_angle):.2f}°, Angle error: {math.degrees(angle_error):.2f}°")

        msg = TwistStamped()
        msg.header = Header()
        msg.header.frame_id = 'base_link'  # Set the frame_id
        msg.header.stamp = self.get_clock().now().to_msg()  # Set the timestamp
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0

        if abs(angle_error) > self.hard_turn_threshold:
            #self.get_logger().info("Hard turn needed")
            # Turn in place for very large errors
            msg.twist.angular.z = self.angular_speed * (1 if angle_error > 0 else -1)
            msg.twist.linear.x = 0.0
            # during a hard turn, publish commitment 0
            self.publishable_commitment = 0

            self.publish_commitment_state()  # Immediately publish the change

            self.my_opinions.append(self.publishable_commitment)

        elif abs(angle_error) < self.hard_turn_threshold and abs(angle_error) > self.soft_turn_threshold:
            #self.get_logger().info("Soft turn needed")
            # Curve while moving
            msg.twist.linear.x = self.linear_speed
            # Proportional controller for angular velocity
            msg.twist.angular.z = max(-self.angular_speed,
                                      min(self.kp_angle * angle_error, self.angular_speed))
            # during a soft turn, publish target commitment
            self.publishable_commitment = self.target_commitment 

            self.publish_commitment_state()  # Immediately publish the change

            self.my_opinions.append(self.publishable_commitment)
        else:
            #self.get_logger().info("Going straight")
            # Go mostly straight
            msg.twist.linear.x = self.linear_speed
            msg.twist.angular.z = 0.0
            # when going straight, publish target commitment
            self.publishable_commitment = self.target_commitment

            self.publish_commitment_state()  # Immediately publish the change

            self.my_opinions.append(self.publishable_commitment)

        self.cmd_pub.publish(msg)
        #self.get_logger().info(f"Published cmd_vel: linear.x={msg.twist.linear.x}, angular.z={msg.twist.angular.z}")

    def control_loop(self):
        """Update target commitment and execute movement."""
        self.update_target_commitment()
        self.update_robot_movement()
        self.log_opinions_data(self.counter)
        self.log_positions_data(self.counter)
        self.counter += 1

    def stop_robot(self):
        """Publish zero velocities."""
        msg = TwistStamped()
        msg.header = Header()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def initialize_position_log(self):
        """Initialize the position log file."""
        time_stamp = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}" # Default experiment name with timestamp

        filename = os.path.join(self.base_log_dir, f"{self.experiment_name}_positions_{time_stamp}.csv")
        self.position_log = open(filename, "w", newline="")
        writer = csv.writer(self.position_log)
        writer.writerow(["Time", "ID", "x", "y"])
        self.position_writer = writer

    def log_positions_data(self, time_step):
        """Log all agents' positions for the current timestep."""
        with self.pos_lock:
            if not self.pos_message or 'self' not in self.pos_message or opt.targets[self.target_commitment-1] not in self.pos_message:
                self.get_logger().warn("Waiting for valid position data...")
                self.get_logger().warn(f"Current pos_message keys: {list(self.pos_message.keys())}")
                return
              
            self.position_writer.writerow([time_step, self.id, self.pos_message['self'].position.x, self.pos_message['self'].position.y])
            self.position_log.flush()   # <- critical to ensure data is actually written

    def close_positions_log_file(self):
        """Close the agent's log file."""
        if self.position_log:
            self.position_log.close()

    def initialize_opinions_log(self):
        """Initialize the agent's log file."""
        time_stamp = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}" # Default experiment name with timestamp

        filename = os.path.join(self.base_log_dir, f"{self.experiment_name}_{self.robot_namespace}_{time_stamp}.csv")
        self.opinions_log = open(filename, "w", newline="")
        writer = csv.writer(self.opinions_log)
        writer.writerow(["Time", "Commitment", "Opinion", "Received Opinions"])
        self.csv_writer = writer

    def log_opinions_data(self, time_step):
        """Log the agent's current state."""
        opinions = ";".join(map(str, self.my_opinions))
        received_opinions = ";".join(f"{k}:{v}" for k, v in self.commitments.items())        # Log the data to the CSV file
        
        self.csv_writer.writerow([time_step, self.commitment, opinions, received_opinions])
        self.opinions_log.flush()   # <- critical to ensure data is actually written
        self.my_opinions.clear()

    def close_opinions_log_file(self):
        """Close the agent's log file."""
        if self.opinions_log:
            self.opinions_log.close()

    def logging_cleanup(self):
        """Close all open resources."""
        # Close position log
        self.close_positions_log_file()

        # Close opnions log
        self.close_opinions_log_file()

    def destroy_node(self):
        """Override destroy_node to clean up logs before shutdown."""
        self.logging_cleanup()
        super().destroy_node()

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