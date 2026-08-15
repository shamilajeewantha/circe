"""circe_localization — motion self-calibration (§5) + two-rate IMU↔SLAM fusion (§6).

No wheel encoders and the VGGT map is RELATIVE scale, so we self-calibrate:
  scale = ||Δt_vggt||  /  (commanded velocity integrated over time)   [map-units per command-second]
maintained as an EMA (translation only — skid-steer turn slip makes rotation noisy, §5).

Two-rate pose (§6): dead-reckon at IMU rate between VGGT updates (gyro for heading,
commanded forward × scale for translation), HARD-RESET to the VGGT pose on each arrival.

Publishes:  /circe/pose_fused (Odometry, relative scale)   /circe/scale (Float64)

[SIM-BOX] Everything here is skeleton-grade; tune EMA gain + validate drift on hardware.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Header

from circe_common.geom import T_from_odom, R_to_quat, quat_to_R


class Localization(Node):
    def __init__(self):
        super().__init__("circe_localization")
        self.declare_parameter("scale_ema", 0.3)
        self.declare_parameter("frame_id", "vggt_map")
        self.ema = float(self.get_parameter("scale_ema").value)
        self.frame = self.get_parameter("frame_id").value

        self.scale = None                  # map-units per command-second (unknown until first move)
        self.cmd_integral = 0.0            # commanded forward distance since last VGGT pose (command-units)
        self.last_cmd_v = 0.0
        self.last_cmd_t = None

        self.T_vggt = None                 # last VGGT pose (4x4)
        self.p_last_vggt = None            # its translation
        self.fused_T = np.eye(4)           # dead-reckoned pose
        self.yaw_rate = 0.0
        self.last_imu_t = None

        self.create_subscription(Twist, "/rover/cmd_vel", self._cmd, 10)
        self.create_subscription(Odometry, "/vggt/pose", self._vggt, 10)
        self.create_subscription(Imu, "/rover/imu", self._imu, 50)
        self.pub_pose = self.create_publisher(Odometry, "/circe/pose_fused", 10)
        self.pub_scale = self.create_publisher(Float64, "/circe/scale", 10)

    def _cmd(self, msg: Twist):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_cmd_t is not None:
            self.cmd_integral += abs(self.last_cmd_v) * (now - self.last_cmd_t)
        self.last_cmd_v = msg.linear.x
        self.yaw_rate = msg.angular.z          # commanded; overridden by IMU below if available
        self.last_cmd_t = now

    def _vggt(self, msg: Odometry):
        T = T_from_odom(msg)
        p = T[:3, 3]
        if self.p_last_vggt is not None and self.cmd_integral > 1e-3:
            disp = float(np.linalg.norm(p - self.p_last_vggt))
            sample = disp / self.cmd_integral            # map-units per command-second
            self.scale = sample if self.scale is None else (1 - self.ema) * self.scale + self.ema * sample
            self.pub_scale.publish(Float64(data=float(self.scale)))
        self.cmd_integral = 0.0
        self.p_last_vggt = p.copy()
        self.T_vggt = T
        self.fused_T = T.copy()              # HARD RESET (§6)

    def _imu(self, msg: Imu):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_imu_t is None:
            self.last_imu_t = now
            return
        dt = now - self.last_imu_t
        self.last_imu_t = now
        # heading from gyro-z; translation forward by commanded v × scale (no encoders)
        wz = msg.angular_velocity.z
        R = self.fused_T[:3, :3]
        dR = _rotz(wz * dt)
        self.fused_T[:3, :3] = R @ dR
        if self.scale is not None:
            fwd = self.fused_T[:3, :3] @ np.array([1.0, 0.0, 0.0])
            self.fused_T[:3, 3] += fwd * (self.last_cmd_v * self.scale * dt)
        self._publish()

    def _publish(self):
        od = Odometry()
        od.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
        od.child_frame_id = "rover"
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = \
            (float(v) for v in self.fused_T[:3, 3])
        w, x, y, z = R_to_quat(self.fused_T[:3, :3])
        od.pose.pose.orientation.w = float(w); od.pose.pose.orientation.x = float(x)
        od.pose.pose.orientation.y = float(y); od.pose.pose.orientation.z = float(z)
        self.pub_pose.publish(od)


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
