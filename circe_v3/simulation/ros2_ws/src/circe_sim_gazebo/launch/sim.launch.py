"""Bring up Gazebo Harmonic with an indoor world + spawn the circe rover +
start the ros_gz bridge.

[SIM-BOX: choose + document the world.] No world is hardcoded — pass any indoor
SDF via the `world` arg. Pick one you can actually get running on your install and
that gives the coverage-aware NBV loop something to chew on: enough occluded/
non-convex structure that fog frontier + detection gaps aren't trivial, sized so a
full run finishes in reasonable time, and no missing-asset/model-path headaches.
TurtleBot4's `depot.sdf` is one evidence-backed option (BUILD_GUIDE.md §3) — but the
stock world (ros-jazzy-turtlebot4-simulator, at .../turtlebot4_gz_bringup/worlds/depot.sdf)
ships its Sensors system plugin COMMENTED OUT, so camera/IMU/gpu_lidar sensors advertise
gz topics but never publish data. Use this package's local fork instead, which is
byte-identical except that plugin is enabled (see worlds/depot_sensors.sdf's own comment):
  ros2 launch circe_sim_gazebo sim.launch.py \
      world:=$(ros2 pkg prefix circe_sim_gazebo)/share/circe_sim_gazebo/worlds/depot_sensors.sdf
but it's an example, not a mandate. Record whatever you land on (name + source +
why) in BUILD_GUIDE.md §3/§8 so it's not a mystery later.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import xacro


def generate_launch_description() -> LaunchDescription:
    desc_pkg = get_package_share_directory("circe_rover_description")
    sim_pkg = get_package_share_directory("circe_sim_gazebo")
    ros_gz_sim = get_package_share_directory("ros_gz_sim")

    robot_desc = xacro.process_file(
        os.path.join(desc_pkg, "urdf", "circe_rover.urdf.xacro")).toxml()
    bridge_cfg = os.path.join(sim_pkg, "config", "bridge.yaml")
    world = LaunchConfiguration("world")

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": [world, " -r"]}.items(),
    )

    spawn = Node(
        package="ros_gz_sim", executable="create", output="screen",
        arguments=["-name", "circe_rover", "-string", robot_desc,
                   "-x", "0", "-y", "0", "-z", "0.12"],
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        parameters=[{"config_file": bridge_cfg}],
    )

    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        output="screen", parameters=[{"robot_description": robot_desc}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", description="Path to an indoor world SDF [SIM-BOX: your choice — see this file's docstring]"),
        gz, spawn, bridge, rsp,
    ])
