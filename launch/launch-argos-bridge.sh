#!/bin/bash

set -euo pipefail

# The purpose of this script is to create a launch file for a multi-robot
# system by replicating a "group" tag "n" times, then executing the resulting
# launch file.

# The number of robots.  This should match the 'quantity' value in the argos world file (e.g. argos_worlds/demo.argos).


LAUNCH_FILE=/tmp/argos_interface.launch.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGOS_CONFIG_DIR="${SCRIPT_DIR}/world_tb.argos"


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/argos3:/home/sindiso/ros2_ws/install/argos3_ros2_bridge/lib
export ARGOS_PLUGIN_PATH=/home/sindiso/ros2_ws/install/argos3_ros2_bridge/lib/

export ROS_LOCALHOST_ONLY=0

RMW_IMPLEMENTATION=rmw_cyclonedds_cpp argos3 -c $ARGOS_CONFIG_DIR -z