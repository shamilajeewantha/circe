"""Bring up the circe rover BRAIN (everything except Gazebo + the SLAM host).

Prereqs, in order:
  1. On the SLAM laptop:  python slam_host/slam_server.py --port 8000 --vis_map
  2. On this (sim) box:   ros2 launch circe_sim_gazebo sim.launch.py world:=.../depot.sdf
  3. Then:                ros2 launch circe_bringup brain.launch.py slam_url:=http://<slam-ip>:8000

Starts: vggt_client, localization, mapping, coverage, explore, driver, viz — all
reading circe_bringup/config/params.yaml. Set include_viz:=false to skip the Gradio app.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    cfg = os.path.join(get_package_share_directory("circe_bringup"), "config", "params.yaml")
    slam_url = LaunchConfiguration("slam_url")

    def node(pkg, exe, name, extra=None):
        params = [cfg] + ([extra] if extra else [])
        return Node(package=pkg, executable=exe, name=name, output="screen", parameters=params)

    return LaunchDescription([
        DeclareLaunchArgument("slam_url", default_value="http://127.0.0.1:8000"),
        DeclareLaunchArgument("include_viz", default_value="true"),
        node("circe_vggt_client", "client_node", "circe_vggt_client", {"slam_url": slam_url}),
        node("circe_localization", "localization_node", "circe_localization"),
        node("circe_mapping", "mapping_node", "circe_mapping"),
        node("circe_coverage", "coverage_node", "circe_coverage"),
        node("circe_explore", "explore_node", "circe_explore"),
        node("circe_driver", "driver_node", "circe_driver"),
        # circe_viz is a Gradio app (has its own web server); launch separately if preferred:
        #   python -m circe_viz.app --port 7860
    ])
