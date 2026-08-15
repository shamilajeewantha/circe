"""RViz check of the rover model (no Gazebo). Verifies chassis + camera/IMU/ToF frames."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("circe_rover_description")
    xacro_file = os.path.join(pkg, "urdf", "circe_rover.urdf.xacro")
    robot_desc = xacro.process_file(xacro_file).toxml()
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[{"robot_description": robot_desc}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2", output="screen"),
    ])
