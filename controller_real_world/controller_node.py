"""ROS2 IMPORTS"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion
from controller_msgs.msg import CommitmentState
from argos3_ros2_bridge.msg import Position
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.task import Future

"""PYTHON IMPORTS"""
import os
import csv
import math
import json
import random
import datetime
import threading
import numpy as np
import zlib


#==========================================================================

#======================define parameters=============================
package_name = 'controller_real_world'

param_file = (
    os.environ.get('PARAMS_FILE')
    or os.path.join(get_package_share_directory(package_name), 'parameters.json')
)

with open(param_file, 'r') as f:
    parameters = json.load(f)

class Options():
    def __init__(self):
        
        #self.id = parameters["robot_namespace"].strip("/")
        #self.robot_namespace = parameters.get("robot_namespace")
        
        self.linear_speed = float(parameters["linear_speed"])
        self.angular_speed = float(np.radians(parameters["angular_speed"])) # Convert from degrees/s to radians/s

        self.targets = parameters.get("targets", [])
        self.soft_turn_threshold = float(np.radians(parameters.get("soft_turn_threshold", 5.0))) # radians (~5°)
        self.hard_turn_threshold = float(np.radians(parameters.get("hard_turn_threshold", 10.0))) # radians (~10°)
        self.formation_radius = float(parameters.get("formation_radius", 2.5))
        self.kp_angle = float(parameters.get("kp_angle", 0.5)) # Proportional gain for angle correction # radians (~28.65°)
        #self.qtm_ip = parameters.get("qtm_ip", "134.34.231.207")  # Add QTM server IP to parameters

        self.update_rate = int(parameters.get("update_rate", 10)) # time steps
        self.eta = float(parameters.get("eta", 0.1)) # weight for neighbor influence
        self.commitment_topic = parameters.get("commitment_topic", "/commitments")

        self.termination_epsilon = float(parameters.get("termination_epsilon", 0.05))
        self.patience_threshold = int(parameters.get("patience_threshold", 50))
        self.improvement_epsilon = float(parameters.get("improvement_epsilon", 0.01))
        self.position_stale_timeout = float(parameters.get("position_stale_timeout", 3.0))  # seconds

        self.base_log_dir = os.path.expanduser(parameters.get('log_directory', '~/geometry-logs'))
        os.makedirs(self.base_log_dir, exist_ok=True)         # If directory does not exist, create it
        self.experiment_name = parameters.get('experiment_name', 'experiment')

        # "shared"    — all robots publish/subscribe to one shared topic (default)
        # "per_robot" — each robot publishes on its own topic; others subscribe individually
        self.commitment_mode = parameters.get('commitment_mode', 'shared')
        self.robots = parameters.get('robots', [])  # required when commitment_mode == "per_robot"

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
        super().__init__('controller_node')

        # --------- Parameters ---------
        self.id = self.get_namespace().strip("/") # opt.id
        self.robot_namespace = self.get_namespace() #opt.robot_namespace
        self.update_rate = opt.update_rate   # time steps
        self.counter = random.randint(0, self.update_rate)
        self.eta = opt.eta # weight for neighbor influence
        
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
        self.formation_radius = opt.formation_radius
        self.hard_turn_threshold = opt.hard_turn_threshold
        self.soft_turn_threshold = opt.soft_turn_threshold
        self.kp_angle = opt.kp_angle  # Proportional gain for angle correction
        #self.qtm_ip = opt.qtm_ip  # QTM server IP
        commitment_topic = (opt.commitment_topic or "/commitments").strip()
        if not commitment_topic.startswith("/"):
            commitment_topic = f"/{commitment_topic}"
        self.commitment_topic = commitment_topic
        #------------------------------

        # --- State ---
        self.pos_message = {}
        self.pos_lock = threading.Lock()
        self.rb_names = []
        self.rb_indices = []
        self.commitments = {}
        # Logging-only mirror of target commitment used in CSV output.
        self.logged_target_commitment = self.target_commitment
        self.my_opinions = []
        self.quality = 1.0
        self.arrived_at_goal = False
        self.hold_commitment = False

        # --- Termination state ---
        self.min_distance_achieved = float('inf')
        self.patience_counter = 0
        self.terminated = False
        self.last_position_time = None  # set on first position message
        self.shutdown_future = Future()
        # -------------------------
        # -------------

        # ----- Logging -----
        self.base_log_dir = opt.base_log_dir
        self.experiment_name = opt.experiment_name
        self.get_logger().info(f"Logging data to: {self.base_log_dir}, Experiment name: {self.experiment_name}")
        # -------------------

       
        self.robot_id = self.id
        self.seq = 0

        if opt.commitment_mode == 'per_robot':
            # Each robot publishes on its own topic and subscribes to each neighbour individually.
            # TRANSIENT_LOCAL + KEEP_LAST(1) means the middleware caches the last value,
            # so a subscriber always gets the latest even if it missed the publish tick.
            per_robot_topic = f"/{self.id}/commitment"
            self.pub = self.create_publisher(CommitmentState, per_robot_topic, commitment_qos)
            self.neighbor_subs = []
            for robot_id in opt.robots:
                if robot_id == self.id:
                    continue
                self.neighbor_subs.append(
                    self.create_subscription(
                        CommitmentState,
                        f"/{robot_id}/commitment",
                        self.listener_cb,
                        commitment_qos
                    )
                )
            self.get_logger().info(
                f"commitment_mode=per_robot: publishing on {per_robot_topic}, "
                f"subscribed to {[f'/{r}/commitment' for r in opt.robots if r != self.id]}"
            )
        else:
            # Shared topic: all robots publish and subscribe to the same topic.
            self.pub = self.create_publisher(CommitmentState, self.commitment_topic, commitment_qos)
            self.sub = self.create_subscription(
                CommitmentState,
                self.commitment_topic,
                self.listener_cb,
                commitment_qos
            )
            self.get_logger().info(f"per-robot mode off, commitment_mode=shared: using topic {self.commitment_topic}")

        # -------------Listen to ARGoS messages --------------
        # position subscriber to listen to ARGoS position updates
        self.position_topic = f"/{self.robot_id}/position"  
        self.pos_sub = self.create_subscription(
            Position,
            self.position_topic,
            self.position_listener_cb,
            1
        )

        ns = (self.robot_namespace or "").strip()

        # remove all leading/trailing slashes so we control formatting
        ns = ns.strip("/")

        # build a prefix: "" (no namespace) or "/<ns>"
        prefix = f"/{ns}" if ns else ""

        self.get_logger().info(f"Using namespace prefix: {prefix}/cmd_vel")

        self.cmd_pub = self.create_publisher(
            Twist,
            f"{prefix}/cmd_vel",
            1
        )
        self.termination_pub = self.create_publisher(String, '/robot_terminated', 10)
        # Moves 0.022 meters (2.2 cm) per update at 10 Hz 
        self.timer = self.create_timer(0.1, self.control_loop)

        # Setup QTM connection in a separate thread
        """self._loop = asyncio.new_event_loop()
        self._connection = None
        self._thread = threading.Thread(target=self._run_rt, daemon=True)
        self._thread.start()"""
        # ----------------------

        # ----------- Initialize log files -----------
        run_folder = os.path.join(self.base_log_dir, self.experiment_name)
        self.initialize_opinions_log()
        self.initialize_position_log()    
        # ---------------------------------------------

        self.get_logger().debug(f"Controller node started. Target commitment: {opt.targets[self.target_commitment-1]}")

    """def _run_rt(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._rt_protocol())
        self._loop.run_forever()

    async def _rt_protocol(self):
        try:
            self._connection = await qtm.connect(self.qtm_ip, version="1.22")
            if self._connection is None:
                self.get_logger().error(f"Failed to connect to QTM server at {self.qtm_ip}")
                return

            self.get_logger().info("Query 6DOF settings XML")
            xml_string = await self._connection.get_parameters(parameters=["6d"])
            if not xml_string:
                self.get_logger().error("Failed to retrieve 6D parameters from QTM")
                return

            root = ET.fromstring(xml_string)
            enabled_names = []
            enabled_indices = []
            for idx, body in enumerate(root.iter("Body")):
                name_el = body.find("Name")
                if name_el is None or not name_el.text:
                    continue
                enabled_el = body.find("Enabled")
                if enabled_el is None or enabled_el.text is None:
                    continue
                enabled = enabled_el.text.strip().lower() in {"true", "1", "yes"}
                if enabled:
                    enabled_names.append(name_el.text)
                    enabled_indices.append(idx)
            self.rb_names = enabled_names
            self.rb_indices = enabled_indices
            self.get_logger().info(f"Enabled rigid bodies: {self.rb_names}")

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
                for list_idx, body_idx in enumerate(self.rb_indices):
                    if list_idx >= len(self.rb_names):
                        continue
                    if body_idx >= len(rbs):
                        continue
                    ((x, y, z), rotation) = rbs[body_idx]
                    name = self.rb_names[list_idx]
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
            self.get_logger().error(f"Error processing QTM packet: {str(e)}")"""

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
        self.get_logger().debug(
            f'[{self.get_name()}] {msg.robot_id} committed to {msg.commitment}'
        )

    def position_listener_cb(self, msg: Position):
        self.pos_message = msg
        self.last_position_time = self.get_clock().now()
        self.get_logger().debug(f"Current pos_message: {(self.pos_message)}")


    def update_target_commitment(self):
        if self.arrived_at_goal:
            return
        if self.counter % self.update_rate == 0:
            if random.random() < self.eta:
                self.target_commitment = random.randrange(len(opt.targets)) + 1
            else:
                # Pick random neighbor's commitment
                if self.commitments:
                    neighbor = random.choice(list(self.commitments.values()))
                    # Check if value is not 0, if 0 keep own commitment
                    if neighbor != 0:    
                        self.target_commitment = neighbor
                # Clear commitments to avoid bias
                self.commitments = {}
            self.logged_target_commitment = self.target_commitment
            
    def update_robot_movement(self):
        """Control loop: hard-turn if needed, else drive straight."""
        if not self.pos_message:
            self.get_logger().debug("Waiting for valid position data...")
            return

        target_idx = self.target_commitment - 1
        if target_idx < 0 or target_idx >= len(opt.targets):
            self.get_logger().warn(f"Invalid target commitment: {self.target_commitment}")
            return

        

        target_pose = opt.targets[target_idx]
        self_pose = self.pos_message

        dx = target_pose[0] - self_pose.position.x
        dy = target_pose[1] - self_pose.position.y
        distance_to_target = math.hypot(dx, dy)

        # Update best distance and patience counter
        if distance_to_target < self.min_distance_achieved - opt.improvement_epsilon:
            self.min_distance_achieved = distance_to_target
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Condition 1: essentially at target
        direct_arrival = distance_to_target < opt.termination_epsilon

        # Condition 2: stuck (no improvement for patience_threshold steps) and within formation_radius
        stuck_within_radius = (
            self.patience_counter >= opt.patience_threshold
            and distance_to_target < self.formation_radius
        )

        if direct_arrival or stuck_within_radius:
            self.arrived_at_goal = True
            self.stop_robot()
            self.publishable_commitment = self.target_commitment
            self.publish_commitment_state()
            self.my_opinions.append(self.publishable_commitment)
            reason = "direct arrival" if direct_arrival else "stuck within radius"
            self.get_logger().info(
                f"[{self.id}] Terminating ({reason}): "
                f"distance={distance_to_target:.3f}, "
                f"formation_radius={self.formation_radius}, "
                f"patience={self.patience_counter}"
            )
            self.publish_terminated()
            self.terminated = True
            return

        if self.arrived_at_goal:
            self.stop_robot()
            return
    

        # Compute current yaw
        q = self_pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        """CCW rotations give negative yaw → Qualisys yaw is 
        opposite sign to the atan2 convention), convert by negating:"""
        #yaw = -yaw # do not negate rn

        # Desired heading
        target_angle = math.atan2(dy, dx)
        angle_error = self.wrap_angle(target_angle - yaw)
        # Break ties at ±180° by slightly preferring one direction
        if abs(abs(angle_error) - np.pi) < 0.005:
            angle_error = -np.pi + 0.1  # Always turn right when a,biguous

        
        #self.get_logger().info(f"Current target commitment: {opt.targets[self.target_commitment]}, Current yaw: {math.degrees(yaw):.2f}° Distance to target: {distance:.2f}, Angle to target: {math.degrees(target_angle):.2f}°, Angle error: {math.degrees(angle_error):.2f}°")

        msg = Twist()
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0

        if abs(angle_error) > self.hard_turn_threshold:
            #self.get_logger().info("Hard turn needed")
            # Turn in place for very large errors
            msg.angular.z = self.angular_speed * (1 if angle_error > 0 else -1)
            msg.linear.x = 0.0
            # during a hard turn, publish commitment 0
            self.publishable_commitment = 0

        elif abs(angle_error) < self.hard_turn_threshold and abs(angle_error) > self.soft_turn_threshold:
            #self.get_logger().info("Soft turn needed")
            # Curve while moving
            msg.linear.x = self.linear_speed
            # Proportional controller for angular velocity
            msg.angular.z = max(-self.angular_speed,
                                min(self.kp_angle * angle_error, self.angular_speed))
            # during a soft turn, publish target commitment
            self.publishable_commitment = self.target_commitment
        else:
            #self.get_logger().info("Going straight")
            # Go mostly straight
            msg.linear.x = self.linear_speed
            msg.angular.z = 0.0
            # when going straight, publish target commitment
            self.publishable_commitment = self.target_commitment

        if self.arrived_at_goal:
            self.publishable_commitment = self.target_commitment

        self.publish_commitment_state()
        self.my_opinions.append(self.publishable_commitment)

        self.cmd_pub.publish(msg)
        #self.get_logger().info(f"Published cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}")

    def control_loop(self):
        """Update target commitment and execute movement."""
        if self.terminated:
            return

        # Staleness check: if position data has gone silent (e.g. ARGoS/bridge
        # terminated before us) and our last known distance was within
        # formation_radius, self-terminate rather than hanging indefinitely.
        if (self.last_position_time is not None
                and not self.arrived_at_goal
                and self.min_distance_achieved < self.formation_radius):
            elapsed = (self.get_clock().now() - self.last_position_time).nanoseconds * 1e-9
            if elapsed > opt.position_stale_timeout:
                self.get_logger().warn(
                    f"[{self.id}] Position data stale for {elapsed:.1f}s "
                    f"(last known distance={self.min_distance_achieved:.3f}). Terminating."
                )
                self.arrived_at_goal = True
                self.terminated = True
                self.stop_robot()
                self.publish_terminated()
                self.timer.cancel()
                self.shutdown_future.set_result(True)
                return

        received_snapshot = dict(self.commitments)
        self.update_target_commitment()
        self.update_robot_movement()
        if self.my_opinions:
            self.log_opinions_data(self.counter, received_snapshot)
        if not self.arrived_at_goal:
            self.log_positions_data(self.counter)
        self.counter += 1

        if self.terminated:
            self.timer.cancel()
            self.get_logger().info(f"[{self.id}] Shutting down node.")
            self.shutdown_future.set_result(True)

    def publish_terminated(self):
        """Signal to the termination monitor that this robot is done."""
        msg = String()
        msg.data = self.id
        self.termination_pub.publish(msg)

    def stop_robot(self):
        """Publish zero velocities."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def initialize_position_log(self):
        """Initialize the position log file."""
        time_stamp = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}" # Default experiment name with timestamp

        filename = os.path.join(self.base_log_dir, f"{self.experiment_name}_{self.id}_positions_{time_stamp}.csv")
        self.position_log = open(filename, "w", newline="")
        writer = csv.writer(self.position_log)
        header = ["Time", "ID", "x", "y"]
        """for target in opt.targets:
            header.extend([f"{target}_x", f"{target}_y"])"""
        writer.writerow(header)
        self.position_writer = writer

    def log_positions_data(self, time_step):
        """Log all agents' positions for the current timestep."""
        with self.pos_lock:
            if not self.pos_message:
                return
              
            row = [
                time_step,
                self.id,
                self.pos_message.position.x,
                self.pos_message.position.y,
            ]
            """for target in opt.targets:
                x, y, z  = opt.targets[opt.targets.index(target)]
                row.extend([x, y])"""
            self.position_writer.writerow(row)
            self.position_log.flush()   # <- critical to ensure data is actually written

    def close_positions_log_file(self):
        """Close the agent's log file."""
        if self.position_log:
            self.position_log.close()

    def initialize_opinions_log(self):
        """Initialize the agent's log file."""
        time_stamp = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}" # Default experiment name with timestamp

        filename = os.path.join(self.base_log_dir, f"{self.experiment_name}_{self.id}_opinions_{time_stamp}.csv")
        self.opinions_log = open(filename, "w", newline="")
        writer = csv.writer(self.opinions_log)
        writer.writerow(["Time", "Commitment", "Opinion", "Received Opinions"])
        self.csv_writer = writer

    def log_opinions_data(self, time_step, received_snapshot):
        """Log the agent's current state."""
        opinions = ";".join(map(str, self.my_opinions))
        received_opinions = ";".join(f"{k}:{v}" for k, v in received_snapshot.items())
        self.csv_writer.writerow([time_step, self.publishable_commitment, opinions, received_opinions])
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
    rclpy.spin_until_future_complete(controller_node, controller_node.shutdown_future)
    controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
