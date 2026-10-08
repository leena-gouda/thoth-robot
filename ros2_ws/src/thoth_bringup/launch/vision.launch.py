"""Camera + visitor perception. Run inside ~/thoth_venv:  ros2 launch thoth_bringup vision.launch.py"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('thoth_bringup')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'camera.launch.py'))),
        Node(
            package='thoth_vision',
            executable='visitor_perception_node',
            name='visitor_perception',
            parameters=[os.path.join(bringup, 'config', 'vision.yaml')],
            output='screen',
        ),
    ])
