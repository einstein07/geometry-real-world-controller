"""ROS2 IMPORTS"""
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

"""PYTHON IMPORTS"""
import os
import json
import signal
import subprocess


package_name = 'controller_real_world'

param_file = os.path.join(
    get_package_share_directory(package_name),
    'parameters.json'
)

with open(param_file, 'r') as f:
    parameters = json.load(f)


class TerminationMonitorNode(Node):
    def __init__(self):
        super().__init__('termination_monitor')

        self.num_robots = int(parameters.get('num_robots', 1))
        self.terminated_robots = set()
        self.done = Future()

        self.sub = self.create_subscription(
            String,
            '/robot_terminated',
            self._on_robot_terminated,
            10
        )

        self.get_logger().info(
            f"Termination monitor started. Waiting for {self.num_robots} robot(s) to terminate."
        )

    def _on_robot_terminated(self, msg: String):
        robot_id = msg.data
        if robot_id in self.terminated_robots:
            return

        self.terminated_robots.add(robot_id)
        self.get_logger().info(
            f"Robot '{robot_id}' terminated "
            f"({len(self.terminated_robots)}/{self.num_robots})."
        )

        if len(self.terminated_robots) >= self.num_robots:
            self.get_logger().info("All robots terminated. Shutting down.")
            self._kill_argos()
            self._kill_controllers()
            self.done.set_result(True)

    def _kill_argos(self):
        result = subprocess.run(
            ['pkill', '-SIGINT', '-f', 'argos3'],
            capture_output=True
        )
        if result.returncode == 0:
            self.get_logger().info("ARGoS process terminated (SIGINT).")
        else:
            self.get_logger().warn(
                "pkill found no matching argos3 process "
                f"(returncode={result.returncode}). "
                "ARGoS may have already exited."
            )

    def _kill_controllers(self):
        result = subprocess.run(
            ['pkill', '-SIGINT', '-f', 'controller_node'],
            capture_output=True
        )
        if result.returncode == 0:
            self.get_logger().info("Controller processes terminated (SIGINT).")
        else:
            self.get_logger().warn(
                "pkill found no matching controller_node processes "
                f"(returncode={result.returncode}). "
                "Controllers may have already exited."
            )


def main(args=None):
    rclpy.init(args=args)
    node = TerminationMonitorNode()
    rclpy.spin_until_future_complete(node, node.done)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
