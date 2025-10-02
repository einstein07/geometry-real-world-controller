from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        # Declare parameter file argument (default to parameters.json in package)
        DeclareLaunchArgument(
            'param_file',
            default_value=os.path.join(
                os.getenv('COLCON_PREFIX_PATH').split(':')[0],
                'share', 'controller_real_world', 'parameters.json'
            ),
            description='Full path to the parameters.json file'
        ),

        # Start control.py node
        Node(
            package='controller_real_world',
            executable='controller_node',
            name='controller_real_world',
            output='screen',
            arguments=[LaunchConfiguration('param_file')]
        )
    ])
