"""Launch the VGGT-SLAM client. Point --slam_url at the off-board SLAM host
(the WSL server on the SLAM laptop in sim; a native-Linux box on the real robot).
Networking to reach the SLAM laptop is set up + verified there (WSL mirrored
networking + two scoped Hyper-V/Windows Firewall rules for TCP/8000 — see
slam_host/README.md "Networking"). Get its CURRENT LAN IP from whoever runs it
(DHCP can change it, don't assume the example below stays valid):

  ros2 launch circe_vggt_client client.launch.py slam_url:=http://192.168.1.8:8000
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    slam_url = LaunchConfiguration("slam_url")
    map_poll_period = LaunchConfiguration("map_poll_period")
    return LaunchDescription([
        DeclareLaunchArgument("slam_url", default_value="http://127.0.0.1:8000",
                              description="Base URL of the off-board VGGT-SLAM host"),
        DeclareLaunchArgument("map_poll_period", default_value="1.0",
                              description="Seconds between GET /map polls (stop-and-map cadence)"),
        Node(
            package="circe_vggt_client",
            executable="client_node",
            name="circe_vggt_client",
            output="screen",
            parameters=[{
                "slam_url": slam_url,
                "map_poll_period": map_poll_period,
            }],
        ),
    ])
