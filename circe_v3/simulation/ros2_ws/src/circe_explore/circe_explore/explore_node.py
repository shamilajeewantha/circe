"""circe_explore — coverage-aware Next-Best-View station selection (project.md §7A.5-6).

Per detection-gap cluster: estimate centroid C + mean normal n̄ from nearby surfels;
plan the ideal standoff view and PROJECT it onto the rover's reachable manifold
(z = H_CAM, xy on traversable floor along n̄, yaw facing C — §7A.5). Score every
candidate station by argmax(gain / cost), gain = #surfels made adequate from that view,
cost = path_len + W_TURN·|Δyaw| (§7A.6). Fallback: nearest fog frontier; else DONE.

Subscribes: /circe/detection_gaps, /circe/surfels, /circe/frontiers, /circe/pose_fused
Publishes:  /circe/goal_station (Odometry: position + yaw)   /circe/done (Bool)

[SIM-BOX] snap_to_traversable is a TODO (assumes free floor); add a costmap/height check
and re-verify adequacy after the snap. Tune d_h, R_CLUSTER, W_TURN, F_MIN, COS_MIN.
"""
from __future__ import annotations

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Header

from circe_common.geom import read_points, R_to_quat


def _look_at_optical(c, target):
    """Camera optical rotation (z toward target, y roughly down)."""
    z = target - c
    z = z / (np.linalg.norm(z) + 1e-9)
    up = np.array([0, 0, -1.0])                 # optical y points down
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _cluster(points, radius):
    """Cheap greedy euclidean clustering → list of index arrays."""
    if len(points) == 0:
        return []
    unused = np.ones(len(points), bool)
    clusters = []
    r2 = radius * radius
    for i in range(len(points)):
        if not unused[i]:
            continue
        seed = points[i]
        d2 = ((points - seed) ** 2).sum(1)
        members = np.where(unused & (d2 < r2 * 9))[0]   # coarse ball
        unused[members] = False
        clusters.append(members)
    return clusters


class Explore(Node):
    def __init__(self):
        super().__init__("circe_explore")
        self.declare_parameter("H_CAM", 0.18)
        self.declare_parameter("d_h", 0.6)             # horizontal standoff (map units)
        self.declare_parameter("R_CLUSTER", 0.25)
        self.declare_parameter("W_TURN", 0.3)
        self.declare_parameter("fx", 900.0)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("F_MIN", 6.0)
        self.declare_parameter("COS_INCIDENCE_MIN", 0.5)
        self.declare_parameter("period", 2.0)
        self.declare_parameter("frame_id", "vggt_map")
        g = self.get_parameter
        self.H_CAM = float(g("H_CAM").value); self.d_h = float(g("d_h").value)
        self.R_CLUSTER = float(g("R_CLUSTER").value); self.W_TURN = float(g("W_TURN").value)
        self.fx = float(g("fx").value); self.cx = float(g("cx").value); self.cy = float(g("cy").value)
        self.W = int(g("width").value); self.H = int(g("height").value)
        self.F_MIN = float(g("F_MIN").value); self.COS_MIN = float(g("COS_INCIDENCE_MIN").value)
        self.frame = g("frame_id").value

        self.gaps = np.empty((0, 3), np.float32)
        self.frontier = np.empty((0, 3), np.float32)
        self.surf_p = np.empty((0, 3), np.float32)
        self.surf_n = np.empty((0, 3), np.float32)
        self.rover = np.zeros(3); self.rover_yaw = 0.0
        self._map_seen = False   # has mapping produced ANY data yet? (cold-start guard, see _plan)

        self.create_subscription(PointCloud2, "/circe/detection_gaps", self._gaps, 1)
        self.create_subscription(PointCloud2, "/circe/frontiers", self._frontier, 1)
        self.create_subscription(PointCloud2, "/circe/surfels", self._surfels, 1)
        self.create_subscription(Odometry, "/circe/pose_fused", self._pose, 10)
        self.pub_goal = self.create_publisher(Odometry, "/circe/goal_station", 10)
        self.pub_done = self.create_publisher(Bool, "/circe/done", 1)
        self.create_timer(float(g("period").value), self._plan)

    def _gaps(self, m): self.gaps = read_points(m, ("x", "y", "z"))
    def _frontier(self, m):
        self.frontier = read_points(m, ("x", "y", "z"))
        if len(self.frontier) > 0:
            self._map_seen = True
    def _surfels(self, m):
        d = read_points(m, ("x", "y", "z", "nx", "ny", "nz"))
        self.surf_p, self.surf_n = d[:, :3], d[:, 3:6]
        if len(self.surf_p) > 0:
            self._map_seen = True
    def _pose(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        self.rover = np.array([p.x, p.y, p.z])
        self.rover_yaw = math.atan2(2 * (o.w * o.z + o.x * o.y),
                                    1 - 2 * (o.y * o.y + o.z * o.z))

    # ------------------------------------------------------------------ #
    def _adequate_count(self, c, R):
        """gain: how many surfels this candidate view would make adequate (§7A.6)."""
        if len(self.surf_p) == 0:
            return 0
        rel = self.surf_p - c
        depth = np.linalg.norm(rel, axis=1) + 1e-9
        pc = (R.T @ rel.T).T                       # camera frame
        infront = pc[:, 2] > 0
        u = self.fx * pc[:, 0] / np.where(infront, pc[:, 2], 1) + self.cx
        v = self.fx * pc[:, 1] / np.where(infront, pc[:, 2], 1) + self.cy
        infov = infront & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        footprint = self.fx * 0.05 / depth        # voxel≈0.05; footprint gate
        ray = rel / depth[:, None]
        cos_inc = -(ray * self.surf_n).sum(1)
        ok = infov & (footprint >= self.F_MIN) & (cos_inc >= self.COS_MIN)
        return int(ok.sum())

    def _plan(self):
        best = None                                # (score, c_xy, yaw)
        # --- detection-gap viewpoints (§7A.5/6) ---
        for members in _cluster(self.gaps, self.R_CLUSTER):
            pts = self.gaps[members]
            C = pts.mean(0)
            nbar = self._mean_normal(C)
            if nbar is None:
                continue
            nh = np.array([nbar[0], nbar[1], 0.0])
            if np.linalg.norm(nh) < 1e-3:          # near-vertical normal → floor, unreachable
                continue
            nh /= np.linalg.norm(nh)
            c = np.array([C[0] + nh[0] * self.d_h, C[1] + nh[1] * self.d_h, self.H_CAM])
            R = _look_at_optical(c, C)
            yaw = math.atan2(C[1] - c[1], C[0] - c[0])
            gain = self._adequate_count(c, R)
            if gain <= 0:
                continue
            cost = np.linalg.norm(c[:2] - self.rover[:2]) + self.W_TURN * abs(_wrap(yaw - self.rover_yaw))
            score = gain / max(cost, 1e-3)
            if best is None or score > best[0]:
                best = (score, c, yaw)

        if best is not None:
            self._emit(best[1], best[2]); return

        # --- fallback: nearest fog frontier ---
        if len(self.frontier) > 0:
            d = np.linalg.norm(self.frontier[:, :2] - self.rover[:2], axis=1)
            C = self.frontier[int(np.argmin(d))]
            c = np.array([C[0], C[1], self.H_CAM])
            yaw = math.atan2(C[1] - self.rover[1], C[0] - self.rover[0])
            self._emit(c, yaw); return

        # --- both layers closed --- but only once mapping has actually produced
        # something at least once: at cold start gaps/frontier are BOTH trivially
        # empty (nothing mapped yet), which is indistinguishable from "fully explored"
        # unless guarded — without this, DONE latches permanently on the very first
        # tick (circe_driver has no un-latch path) and the rover never moves.
        if not self._map_seen:
            self.get_logger().info("waiting for first map data (cold start)...",
                                   throttle_duration_sec=5.0)
            return
        self.pub_done.publish(Bool(data=True))
        self.get_logger().info("DONE — no reachable detection gap or frontier", throttle_duration_sec=5.0)

    def _mean_normal(self, C):
        if len(self.surf_p) == 0:
            return None
        d = np.linalg.norm(self.surf_p - C, axis=1)
        near = d < self.R_CLUSTER * 2
        if near.sum() == 0:
            return None
        n = self.surf_n[near].mean(0)
        nn = np.linalg.norm(n)
        return n / nn if nn > 1e-6 else None

    def _emit(self, c, yaw):
        od = Odometry()
        od.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = \
            float(c[0]), float(c[1]), float(c[2])
        R = _rotz(yaw)
        w, x, y, z = R_to_quat(R)
        od.pose.pose.orientation.w = float(w); od.pose.pose.orientation.x = float(x)
        od.pose.pose.orientation.y = float(y); od.pose.pose.orientation.z = float(z)
        self.pub_goal.publish(od)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = Explore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
