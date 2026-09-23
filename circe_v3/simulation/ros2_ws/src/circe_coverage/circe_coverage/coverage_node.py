"""circe_coverage — the geometric adequacy test (project.md §7A.3). NO detector.

Decides whether each surfel has been imaged at detection-grade quality, using only
geometry: FOV projection, footprint (f·voxel/depth ≥ F_MIN), incidence
(dot(−ray, n) ≥ COS_INCIDENCE_MIN). (Occlusion z-buffer is a TODO — see note.) The
word "detection" refers to the quality a detector WOULD need; evaluating it needs no
detector, which is exactly why the full §7A loop survives with the classifier removed.

Owns the persistent covered/q_best state, keyed by surfel voxel (survives map growth).

Subscribes:
  /circe/surfels        PointCloud2 (x,y,z,nx,ny,nz)   geometry from circe_mapping
  /circe/capture_views  geometry_msgs/PoseArray        camera OPTICAL poses of the 8-shot ring
Publishes:
  /circe/detection_gaps PointCloud2 (xyz)              uncovered surfel clusters
  /circe/coverage_viz   PointCloud2 (xyz+rgb)          green=covered / red=gap

[SIM-BOX] Calibrate F_MIN / COS_INCIDENCE_MIN (§7A.8). Add z-buffer occlusion if
partial-map false positives bite (they self-heal next cycle, §7A.7). Floor surfels
(normal≈+z) are excluded from gaps — a horizontal camera can never satisfy them (§7A.5).
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64

from circe_common.geom import read_points, make_cloud, pack_rgb, quat_to_R


class Coverage(Node):
    def __init__(self):
        super().__init__("circe_coverage")
        self.declare_parameter("voxel_size", 0.05)
        self.declare_parameter("fx", 900.0)
        self.declare_parameter("fy", 900.0)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("F_MIN", 6.0)              # pixels the surfel must span
        self.declare_parameter("COS_INCIDENCE_MIN", 0.5)  # ~cos 60°
        self.declare_parameter("floor_normal_cos", 0.8)   # |n·z|>this → floor, excluded
        self.declare_parameter("frame_id", "vggt_map")
        g = self.get_parameter
        self.voxel = float(g("voxel_size").value)
        self.K = (float(g("fx").value), float(g("fy").value), float(g("cx").value), float(g("cy").value))
        self.W, self.H = int(g("width").value), int(g("height").value)
        self.F_MIN = float(g("F_MIN").value)
        self.COS_MIN = float(g("COS_INCIDENCE_MIN").value)
        self.floor_cos = float(g("floor_normal_cos").value)
        self.frame = g("frame_id").value

        # persistent per-surfel state keyed by voxel tuple
        self.pos = {}      # key -> position (3,)
        self.nrm = {}      # key -> normal (3,)
        self.cov = {}      # key -> covered bool
        self.q = {}        # key -> best quality

        # circe_mapping sizes the voxel grid from the map extent (relative scale),
        # so this MUST track it — self.voxel keys every surfel's persistent
        # covered/q_best state and feeds the footprint gate. A stale voxel here
        # silently re-keys the whole detection layer against a different grid.
        self.create_subscription(Float64, "/circe/voxel_size", self._voxel, 10)
        self.create_subscription(PointCloud2, "/circe/surfels", self._surfels, 1)
        self.create_subscription(PoseArray, "/circe/capture_views", self._views, 10)
        self.pub_gaps = self.create_publisher(PointCloud2, "/circe/detection_gaps", 1)
        self.pub_viz = self.create_publisher(PointCloud2, "/circe/coverage_viz", 1)
        self.create_timer(1.0, self._publish)

    def _key(self, p):
        return (int(np.floor(p[0] / self.voxel)), int(np.floor(p[1] / self.voxel)),
                int(np.floor(p[2] / self.voxel)))

    def _voxel(self, m: Float64):
        v = float(m.data)
        if v > 0 and abs(v - self.voxel) > 1e-9:
            self.get_logger().info(f"[coverage] voxel {self.voxel:.5f} -> {v:.5f}; "
                                   f"re-keying {len(self.pos)} surfels")
            self.voxel = v
            # re-key persistent state onto the new grid rather than dropping it
            old = (self.pos, self.nrm, self.cov, self.q)
            self.pos, self.nrm, self.cov, self.q = {}, {}, {}, {}
            for k, p in old[0].items():
                nk = self._key(p)
                self.pos[nk] = p
                self.nrm[nk] = old[1][k]
                self.cov[nk] = self.cov.get(nk, False) or old[2][k]
                self.q[nk] = max(self.q.get(nk, 0.0), old[3][k])

    def _surfels(self, msg: PointCloud2):
        d = read_points(msg, ("x", "y", "z", "nx", "ny", "nz"))
        for row in d:
            p, n = row[:3], row[3:6]
            k = self._key(p)
            self.pos[k] = p
            self.nrm[k] = n
            self.cov.setdefault(k, False)
            self.q.setdefault(k, 0.0)

    def _is_floor(self, n):
        return abs(n[2]) > self.floor_cos

    def _views(self, msg: PoseArray):
        # Per-gate rejection tally. Without it, "covered stayed 0 after 4 rings"
        # is indistinguishable between "views never arrived", "everything was
        # behind the camera / out of frame" and "the footprint or incidence
        # threshold is simply unreachable at this map's scale" — and the
        # footprint gate in particular is scale-sensitive, since it is
        # fx * voxel / depth against a pixel threshold while voxel and depth are
        # both RELATIVE-scale map units.
        rej = {"behind": 0, "fov": 0, "footprint": 0, "incidence": 0,
               "already": 0, "degenerate": 0}
        newly = 0
        fx, fy, cx, cy = self.K
        fp_max = 0.0
        for pose in msg.poses:
            c = np.array([pose.position.x, pose.position.y, pose.position.z])
            o = pose.orientation
            R = quat_to_R(o.w, o.x, o.y, o.z)      # camera optical axes in map frame (z fwd)
            for k, p in self.pos.items():
                if self.cov[k]:
                    rej["already"] += 1
                    continue
                rel = p - c
                depth = float(np.linalg.norm(rel))
                if depth < 1e-4:
                    rej["degenerate"] += 1
                    continue
                pc = R.T @ rel                     # into camera optical frame
                if pc[2] <= 0:                     # behind camera
                    rej["behind"] += 1
                    continue
                u = fx * pc[0] / pc[2] + cx
                vv = fy * pc[1] / pc[2] + cy
                if not (0 <= u < self.W and 0 <= vv < self.H):     # gate (1) FOV
                    rej["fov"] += 1
                    continue
                footprint = fx * self.voxel / depth                # gate (3) close/sharp
                fp_max = max(fp_max, footprint)
                if footprint < self.F_MIN:
                    rej["footprint"] += 1
                    continue
                ray = rel / depth
                cos_inc = float(np.dot(-ray, self.nrm[k]))         # gate (4) head-on
                if cos_inc < self.COS_MIN:
                    rej["incidence"] += 1
                    continue
                # (gate 2 occlusion: TODO z-buffer over surfels)
                self.cov[k] = True
                self.q[k] = max(self.q[k], footprint * cos_inc)
                newly += 1
        self.get_logger().info(
            f"[coverage] ring: {len(msg.poses)} views x {len(self.pos)} surfels -> "
            f"+{newly} covered | rejected {rej} | best_footprint={fp_max:.2f}px "
            f"(F_MIN={self.F_MIN}, voxel={self.voxel:.5f})")

    def _publish(self):
        if not self.pos:
            return
        stamp = self.get_clock().now().to_msg()
        gaps, viz_pts, viz_rgb = [], [], []
        for k, p in self.pos.items():
            covered = self.cov[k]
            floor = self._is_floor(self.nrm[k])
            viz_pts.append(p)
            viz_rgb.append([60, 200, 60] if covered else ([90, 90, 90] if floor else [230, 60, 60]))
            if not covered and not floor:          # floor excluded from completion (§7A.5)
                gaps.append(p)
        viz_pts = np.asarray(viz_pts, np.float32)
        self.pub_viz.publish(make_cloud(viz_pts, self.frame, stamp,
                                        extra={"rgb": pack_rgb(np.asarray(viz_rgb, np.uint8))}))
        if gaps:
            self.pub_gaps.publish(make_cloud(np.asarray(gaps, np.float32), self.frame, stamp))
        n = len(self.pos); nc = sum(self.cov.values())
        self.get_logger().info(f"[coverage] surfels={n} covered={nc} gaps={len(gaps)}",
                               throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = Coverage()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
