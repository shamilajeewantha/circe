"""circe_driver — the rover's motor behavior (project.md §4, §5, §7.1).

State machine, closed on the fused pose (relative scale):
  IDLE → ROTATE_TO_HEADING (face the goal xy) → DRIVE (forward until at goal xy)
       → ROTATE_TO_YAW (goal heading) → RING (rotate 45°, settle, capture ×8) → IDLE
The command vocabulary is only rotate + drive-forward — no arcs. At each ring shot it
publishes the camera OPTICAL pose to /circe/capture_views for circe_coverage.

**ToF hard-interrupt reflex** (always on): if forward clearance < threshold, publish
neutral cmd_vel and refuse to drive forward — overrides any goal, independent of the map.

Subscribes: /circe/goal_station, /circe/pose_fused, /rover/tof (LaserScan), /circe/done
Publishes:  /rover/cmd_vel, /circe/capture_views (PoseArray), /circe/state (String)

[SIM-BOX] Gains (kp_ang/kp_lin), tolerances, ring settle time, ToF threshold are skeleton
defaults — tune on hardware. Camera mount offset must match the URDF (H_CAM, forward x).
"""
from __future__ import annotations

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose, PoseArray
from std_msgs.msg import String, Bool, Header

from circe_common.geom import R_to_quat, quat_to_R

# optical axes expressed in the rover base frame (z=forward=base x, y=down=-base z)
R_MOUNT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float64)


class Driver(Node):
    def __init__(self):
        super().__init__("circe_driver")
        p = self.declare_parameter
        p("kp_ang", 1.5); p("kp_lin", 0.6)
        p("max_ang", 0.8); p("max_lin", 0.25)
        p("yaw_tol", 0.05); p("pos_tol", 0.10)
        p("tof_stop", 0.35)                        # forward clearance (map units) to halt
        p("cam_forward", 0.18); p("cam_height", 0.18)
        p("ring_steps", 8); p("settle_ticks", 10)
        p("frame_id", "vggt_map")
        g = self.get_parameter
        self.kp_ang, self.kp_lin = g("kp_ang").value, g("kp_lin").value
        self.max_ang, self.max_lin = g("max_ang").value, g("max_lin").value
        self.yaw_tol, self.pos_tol = g("yaw_tol").value, g("pos_tol").value
        self.tof_stop = g("tof_stop").value
        self.cam_fwd, self.cam_h = g("cam_forward").value, g("cam_height").value
        self.ring_steps, self.settle_ticks = int(g("ring_steps").value), int(g("settle_ticks").value)
        self.frame = g("frame_id").value

        self.pose = None                           # (xy, yaw)
        self.goal = None                           # (xy, yaw)
        self.state = "IDLE"
        self.tof_clear = math.inf
        self.done = False
        self.ring_i = 0; self.ring_wait = 0; self.ring_start_yaw = 0.0
        self.ring_views = []

        self.create_subscription(Odometry, "/circe/pose_fused", self._pose, 10)
        self.create_subscription(Odometry, "/circe/goal_station", self._goal, 10)
        self.create_subscription(LaserScan, "/rover/tof", self._tof, 10)
        self.create_subscription(Bool, "/circe/done", self._done, 1)
        self.pub_cmd = self.create_publisher(Twist, "/rover/cmd_vel", 10)
        self.pub_views = self.create_publisher(PoseArray, "/circe/capture_views", 10)
        self.pub_state = self.create_publisher(String, "/circe/state", 10)
        self.create_timer(0.05, self._tick)        # 20 Hz control loop

    def _pose(self, m):
        pos = m.pose.pose.position; o = m.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        self.pose = (np.array([pos.x, pos.y]), yaw)

    def _goal(self, m):
        pos = m.pose.pose.position; o = m.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        self.goal = (np.array([pos.x, pos.y]), yaw)
        if self.state == "IDLE":
            self.state = "ROTATE_TO_HEADING"

    def _tof(self, m: LaserScan):
        r = [x for x in m.ranges if not math.isinf(x) and not math.isnan(x) and x > m.range_min]
        self.tof_clear = min(r) if r else math.inf

    def _done(self, m):
        self.done = m.data

    # ------------------------------------------------------------------ #
    def _tick(self):
        if self.pose is None:
            return
        cmd = Twist()

        # ToF hard-interrupt: overrides everything, refuses forward motion.
        blocked = self.tof_clear < self.tof_stop

        if self.done:
            self.state = "DONE"
        elif self.goal is not None and self.state != "RING":
            xy, yaw = self.pose
            gxy, gyaw = self.goal
            to_goal = gxy - xy
            dist = float(np.linalg.norm(to_goal))
            head = math.atan2(to_goal[1], to_goal[0])

            if self.state == "ROTATE_TO_HEADING":
                e = _wrap(head - yaw)
                if abs(e) < self.yaw_tol:
                    self.state = "DRIVE"
                else:
                    cmd.angular.z = _clip(self.kp_ang * e, self.max_ang)
            elif self.state == "DRIVE":
                if dist < self.pos_tol:
                    self.state = "ROTATE_TO_YAW"
                elif blocked:
                    pass                            # halt; wait for clearance
                else:
                    cmd.angular.z = _clip(self.kp_ang * _wrap(head - yaw), self.max_ang)
                    cmd.linear.x = _clip(self.kp_lin * dist, self.max_lin)
            elif self.state == "ROTATE_TO_YAW":
                e = _wrap(gyaw - yaw)
                if abs(e) < self.yaw_tol:
                    self.state = "RING"; self.ring_i = 0; self.ring_wait = 0
                    self.ring_start_yaw = yaw; self.ring_views = []
                else:
                    cmd.angular.z = _clip(self.kp_ang * e, self.max_ang)

        elif self.state == "RING":
            self._ring_step(cmd)

        if blocked:                                 # safety backstop, always last word
            cmd.linear.x = min(cmd.linear.x, 0.0)

        self.pub_cmd.publish(cmd)
        self.pub_state.publish(String(data=self.state))

    def _ring_step(self, cmd: Twist):
        xy, yaw = self.pose
        target = _wrap(self.ring_start_yaw + self.ring_i * (2 * math.pi / self.ring_steps))
        e = _wrap(target - yaw)
        if abs(e) > self.yaw_tol:
            cmd.angular.z = _clip(self.kp_ang * e, self.max_ang)
            self.ring_wait = 0
            return
        # settled at this shot heading — never capture while moving (§7.1)
        self.ring_wait += 1
        if self.ring_wait >= self.settle_ticks:
            self.ring_views.append(self._camera_pose(xy, yaw))
            self.ring_i += 1
            self.ring_wait = 0
            if self.ring_i >= self.ring_steps:
                self._publish_views()
                self.goal = None
                self.state = "IDLE"

    def _camera_pose(self, xy, yaw) -> Pose:
        Rr = _rotz(yaw)
        c = np.array([xy[0], xy[1], 0.0]) + Rr @ np.array([self.cam_fwd, 0.0, self.cam_h])
        Ropt = Rr @ R_MOUNT                         # camera optical axes in map
        w, x, y, z = R_to_quat(Ropt)
        ps = Pose()
        ps.position.x, ps.position.y, ps.position.z = float(c[0]), float(c[1]), float(c[2])
        ps.orientation.w, ps.orientation.x, ps.orientation.y, ps.orientation.z = \
            float(w), float(x), float(y), float(z)
        return ps

    def _publish_views(self):
        pa = PoseArray()
        pa.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
        pa.poses = self.ring_views
        self.pub_views.publish(pa)
        self.get_logger().info(f"[driver] published 8-shot ring ({len(self.ring_views)} views)")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clip(v, m):
    return max(-m, min(m, v))


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
