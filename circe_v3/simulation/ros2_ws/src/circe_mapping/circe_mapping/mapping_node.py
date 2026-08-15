"""circe_mapping — builds BOTH coverage layers from the VGGT cloud (project.md §7A.1-2).

  (a) FOG layer: a voxel occupancy grid → frontier = boundary of the observed region.
  (b) DETECTION layer: one surfel per occupied voxel {position p, PCA normal n},
      published for circe_coverage to test adequacy against.

All relative-scale-safe (voxel_size + depth are both map units, §7A.3). Recomputed on a
timer (not every cloud) to bound cost.

Publishes:
  /circe/frontiers  PointCloud2 (xyz)                     — fog frontier centroids
  /circe/surfels    PointCloud2 (x,y,z,nx,ny,nz)          — detection-layer surfels
  /circe/map_viz    PointCloud2 (xyz+rgb)                 — for circe_viz / RViz

[SIM-BOX] Frontier here is a boundary heuristic (no free/unknown raycasting yet); the
proper §7A.1 free/unknown split from camera rays is a TODO. Normals are oriented toward
the latest rover pose. Tune voxel_size, neighborhood radius, frontier thresholds.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

from circe_common.geom import read_points, make_cloud, pack_rgb, voxel_downsample


class Mapping(Node):
    def __init__(self):
        super().__init__("circe_mapping")
        self.declare_parameter("voxel_size", 0.05)
        self.declare_parameter("max_points", 400000)
        self.declare_parameter("recompute_period", 1.0)
        self.declare_parameter("frame_id", "vggt_map")
        self.voxel = float(self.get_parameter("voxel_size").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.frame = self.get_parameter("frame_id").value

        self.points = np.empty((0, 3), np.float32)
        self.colors = np.empty((0, 3), np.uint8)
        self.rover_pos = np.zeros(3, np.float32)

        self.create_subscription(PointCloud2, "/vggt/cloud", self._cloud, 1)
        self.create_subscription(Odometry, "/circe/pose_fused", self._pose, 10)
        self.pub_frontier = self.create_publisher(PointCloud2, "/circe/frontiers", 1)
        self.pub_surfels = self.create_publisher(PointCloud2, "/circe/surfels", 1)
        self.pub_viz = self.create_publisher(PointCloud2, "/circe/map_viz", 1)
        self.create_timer(float(self.get_parameter("recompute_period").value), self._recompute)

    def _pose(self, m: Odometry):
        p = m.pose.pose.position
        self.rover_pos = np.array([p.x, p.y, p.z], np.float32)

    def _cloud(self, msg: PointCloud2):
        xyz = read_points(msg, ("x", "y", "z"))
        if len(xyz) == 0:
            return
        self.points = np.vstack([self.points, xyz]).astype(np.float32)
        # keep bounded + deduped on the fog grid
        self.points, _ = voxel_downsample(self.points, self.voxel)
        if len(self.points) > self.max_points:
            sel = np.linspace(0, len(self.points) - 1, self.max_points).astype(int)
            self.points = self.points[sel]

    # ------------------------------------------------------------------ #
    def _recompute(self):
        if len(self.points) < 50:
            return
        stamp = self.get_clock().now().to_msg()
        v = self.voxel
        keys = np.floor(self.points / v).astype(np.int64)
        occ = set(map(tuple, keys))                        # occupied voxel keys

        # --- surfels: one per occupied voxel, PCA normal from k-NN points ---
        centroids, normals = [], []
        # bucket points by voxel for local PCA
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        ks = keys[order]; ps = self.points[order]
        starts = np.concatenate([[0], np.where(np.any(np.diff(ks, axis=0) != 0, axis=1))[0] + 1, [len(ks)]])
        for a, b in zip(starts[:-1], starts[1:]):
            pts = ps[a:b]
            c = pts.mean(0)
            if len(pts) >= 3:
                cov = np.cov((pts - c).T)
                w, V = np.linalg.eigh(cov)
                n = V[:, 0]                                 # smallest eigenvector
            else:
                n = np.array([0, 0, 1], np.float32)
            # orient toward the rover (observed side, §7A.2)
            if np.dot(self.rover_pos - c, n) < 0:
                n = -n
            centroids.append(c); normals.append(n)
        centroids = np.asarray(centroids, np.float32)
        normals = np.asarray(normals, np.float32)

        if len(centroids):
            self.pub_surfels.publish(make_cloud(
                centroids, self.frame, stamp,
                extra={"nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2]}))
            rgb = np.tile(np.array([170, 170, 170], np.uint8), (len(centroids), 1))
            self.pub_viz.publish(make_cloud(centroids, self.frame, stamp,
                                            extra={"rgb": pack_rgb(rgb)}))

        # --- frontier: empty voxels adjacent to occupied at the map boundary ---
        frontier = []
        nbrs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        checked = set()
        for k in occ:
            for d in nbrs:
                e = (k[0] + d[0], k[1] + d[1], k[2] + d[2])
                if e in occ or e in checked:
                    continue
                checked.add(e)
                # boundary heuristic: an empty voxel touching occupied but mostly
                # surrounded by empty (few occupied neighbors) is a frontier
                occn = sum((e[0] + dd[0], e[1] + dd[1], e[2] + dd[2]) in occ for dd in nbrs)
                if occn <= 2:
                    frontier.append([(e[0] + 0.5) * v, (e[1] + 0.5) * v, (e[2] + 0.5) * v])
        if frontier:
            self.pub_frontier.publish(make_cloud(np.asarray(frontier, np.float32), self.frame, stamp))
        self.get_logger().info(
            f"[map] pts={len(self.points)} surfels={len(centroids)} frontier={len(frontier)}",
            throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = Mapping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
