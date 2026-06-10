#!/usr/bin/env bash
# Opens rqt_image_view showing /rover_camera/image.
# Run this after launch_default.sh or launch_baylands.sh is up.
# Usage: bash ~/github_desktop/circe/gazebo/view_camera.sh

source /opt/ros/jazzy/setup.bash
source "$HOME/ws_px4/install/local_setup.bash"
ros2 run rqt_image_view rqt_image_view /rover_camera/image
