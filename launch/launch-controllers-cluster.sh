#!/bin/bash

set -euo pipefail

# The purpose of this script is to create a launch file for a multi-robot
# system by replicating a "group" tag "n" times, then executing the resulting
# launch file.

# The number of robots.  This should match the 'quantity' value in the argos world file (e.g. argos_worlds/demo.argos).

n=20

LAUNCH_FILE=/tmp/argos_interface.launch.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat > "${LAUNCH_FILE}" <<EOF
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
EOF

# If a writable params file was provided by the caller (e.g. run_experiments_cluster.sh),
# pass it as a command-line argument to each controller node.
# The container filesystem at /opt/ is read-only so we must never rely on
# the installed share copy being up-to-date.
NODE_ARGS_FRAGMENT=""
if [ -n "${PARAMS_FILE:-}" ]; then
    NODE_ARGS_FRAGMENT=", arguments=[\"${PARAMS_FILE}\"]"
fi

for ((i=0; i<n; i++)); do
    namespace="bot$i"
    domain_id="${ROS_DOMAIN_ID:-0}"  # Inherited from SLURM task; isolates this task's graph.
    echo "    ${namespace} = Node(package=\"controller_real_world\", executable=\"controller_node\", name=\"controller_real_world\", output=\"screen\", namespace=\"${namespace}\"${NODE_ARGS_FRAGMENT}, additional_env={\"ROS_DOMAIN_ID\": \"${domain_id}\"})" >> "${LAUNCH_FILE}"
    echo "    ld.add_action(${namespace})" >> "${LAUNCH_FILE}"
done

# Termination monitor: kills ARGoS once all n robots have reported done.
# on_exit=Shutdown() causes ros2 launch to exit cleanly when this node finishes.
echo "    termination_monitor = Node(package=\"controller_real_world\", executable=\"termination_monitor\", name=\"termination_monitor\", output=\"screen\", on_exit=Shutdown())" >> "${LAUNCH_FILE}"
echo "    ld.add_action(termination_monitor)" >> "${LAUNCH_FILE}"

echo "    return ld" >> "${LAUNCH_FILE}"


# keep topics local to computer
export ROS_LOCALHOST_ONLY=0

#argos3 -c $ARGOS_CONFIG_DIR

RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch "${LAUNCH_FILE}"
