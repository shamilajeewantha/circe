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
from std_msgs.msg import Float64

from circe_common.geom import read_points, make_cloud, pack_rgb, voxel_downsample


class Mapping(Node):
    def __init__(self):
        super().__init__("circe_mapping")
        self.declare_parameter("voxel_size", 0.05)
        self.declare_parameter("max_points", 400000)
        self.declare_parameter("recompute_period", 1.0)
        # PCA neighbourhood half-width in VOXELS: r=1 -> 3x3x3 block (0.15 m at the
        # default 5 cm voxel). Raise for smoother normals on sparse maps, at cost.
        self.declare_parameter("normal_radius", 1)
        # frontier heuristic: empty voxel is frontier if <= this many occupied nbrs
        self.declare_parameter("max_occ_nbrs", 2)
        # cap on published frontier points (a depot run produced 1,077,860 of them,
        # which is a pointless PointCloud2 to build and ship every cycle)
        self.declare_parameter("max_frontier", 60000)
        # Relative-scale safety: size the grid from the map extent instead of a
        # fixed absolute value (see _adapt_voxel). target_cells is the real knob.
        self.declare_parameter("adaptive_voxel", True)
        self.declare_parameter("target_cells", 120)
        self.declare_parameter("voxel_hysteresis", 0.25)
        # percentile trimmed off EACH end when measuring map extent (outlier guard)
        self.declare_parameter("extent_pct", 2.0)
        self.declare_parameter("frame_id", "vggt_map")
        self.voxel = float(self.get_parameter("voxel_size").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.normal_radius = int(self.get_parameter("normal_radius").value)
        self.max_occ_nbrs = int(self.get_parameter("max_occ_nbrs").value)
        self.max_frontier = int(self.get_parameter("max_frontier").value)
        self.adaptive_voxel = bool(self.get_parameter("adaptive_voxel").value)
        self.target_cells = int(self.get_parameter("target_cells").value)
        self.voxel_hysteresis = float(self.get_parameter("voxel_hysteresis").value)
        self.extent_pct = float(self.get_parameter("extent_pct").value)
        self.frame = self.get_parameter("frame_id").value

        self.points = np.empty((0, 3), np.float32)
        self.colors = np.empty((0, 3), np.uint8)
        self.rover_pos = np.zeros(3, np.float32)

        self.create_subscription(PointCloud2, "/vggt/cloud", self._cloud, 1)
        self.create_subscription(Odometry, "/circe/pose_fused", self._pose, 10)
        self.pub_frontier = self.create_publisher(PointCloud2, "/circe/frontiers", 1)
        self.pub_surfels = self.create_publisher(PointCloud2, "/circe/surfels", 1)
        self.pub_viz = self.create_publisher(PointCloud2, "/circe/map_viz", 1)
        # latched-ish: coverage/explore must share this exact grid
        self.pub_voxel = self.create_publisher(Float64, "/circe/voxel_size", 10)
        self.create_timer(float(self.get_parameter("recompute_period").value), self._recompute)

    def _pose(self, m: Odometry):
        p = m.pose.pose.position
        self.rover_pos = np.array([p.x, p.y, p.z], np.float32)

    def _adapt_voxel(self, xyz: np.ndarray) -> None:
        """Size the fog/surfel grid from the map's OWN extent.

        VGGT-SLAM is monocular, so the map is RELATIVE scale (project.md §3/§5) —
        a fixed voxel_size is not a length, it is an arbitrary fraction of
        whatever scale this particular run happened to land on. Measured on two
        runs with the identical 0.05 setting: 25,674 surfels on one, 183 on the
        next (cloud extent 0.155 x 0.016 x 0.201 units = 3 x 0.3 x 4 cells).
        A 140x density swing, so map quality was a per-run lottery and planning
        with it was meaningless.

        Deriving it from the longest axis instead makes the grid resolution
        scale-invariant: target_cells across the map, whatever the units are.
        Published on /circe/voxel_size so coverage and explore use the SAME grid
        (both key surfel state / the footprint gate off it).
        """
        if not self.adaptive_voxel or len(xyz) == 0:
            return
        # Percentile extent, NOT min/max: VGGT clouds carry occasional far-flung
        # outliers, and absolute min/max lets a single stray point set the grid for
        # the whole map. Seen live: one outlier stretched the span to ~7.5e3 units,
        # so voxel became 62.7 and the entire real map collapsed into ~1 cell
        # (60 surfels, every adequacy view then "behind camera"/out of frame).
        lo = np.percentile(xyz, self.extent_pct, axis=0)
        hi = np.percentile(xyz, 100.0 - self.extent_pct, axis=0)
        span = float(np.max(hi - lo))
        if span <= 1e-9:
            return
        v = span / max(self.target_cells, 1)
        # Only react to real changes: rewriting the grid every cloud would
        # re-bucket the whole map constantly for no benefit.
        if self.voxel <= 0 or abs(v - self.voxel) / self.voxel > self.voxel_hysteresis:
            self.voxel = v
            self.pub_voxel.publish(Float64(data=float(v)))
            self.get_logger().info(
                f"[map] voxel -> {v:.5f} (map span {span:.3f} rel-units / "
                f"{self.target_cells} cells)")

    def _cloud(self, msg: PointCloud2):
        xyz = read_points(msg, ("x", "y", "z"))
        if len(xyz) == 0:
            return
        self.points = np.vstack([self.points, xyz]).astype(np.float32)
        self._adapt_voxel(self.points)
        # keep bounded + deduped on the fog grid
        self.points, _ = voxel_downsample(self.points, self.voxel)
        if len(self.points) > self.max_points:
            sel = np.linspace(0, len(self.points) - 1, self.max_points).astype(int)
            self.points = self.points[sel]

    # ------------------------------------------------------------------ #
    def _neighbour_normals(self, pts: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Per-point PCA normal fitted over the surrounding (2r+1)^3 voxel block.

        The cloud is one point per voxel, so a single voxel can never define a plane
        — the neighbourhood is what carries the surface orientation. Fully vectorised
        (one pass per neighbour offset + a batched 3x3 eigh) because this runs on the
        whole map every recompute_period; the previous per-voxel Python loop was
        already the slowest thing in this node.

        Points whose neighbourhood is still too small to define a plane keep the
        original (0,0,1) fallback — now a rare isolated-speck case rather than,
        as it was, every single surfel.
        """
        n = len(pts)
        r = self.normal_radius
        kk = (keys - keys.min(0)).astype(np.int64)
        dims = kk.max(0) + 1
        # pack the 3-D voxel index into one int64 so neighbours are a sorted lookup
        code = (kk[:, 0] * dims[1] + kk[:, 1]) * dims[2] + kk[:, 2]
        order = np.argsort(code)
        code_sorted = code[order]

        s1 = np.zeros((n, 3))       # sum of neighbour coords
        s2 = np.zeros((n, 6))       # sum of products: xx, yy, zz, xy, xz, yz
        cnt = np.zeros(n)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nk = kk + np.array([dx, dy, dz], np.int64)
                    inside = np.all((nk >= 0) & (nk < dims), axis=1)
                    ncode = (nk[:, 0] * dims[1] + nk[:, 1]) * dims[2] + nk[:, 2]
                    pos = np.clip(np.searchsorted(code_sorted, ncode), 0, n - 1)
                    hit = inside & (code_sorted[pos] == ncode)
                    centers = np.nonzero(hit)[0]
                    if len(centers) == 0:
                        continue
                    q = pts[order[pos[hit]]].astype(np.float64)
                    # centers are unique within one offset pass, so += is safe
                    # (no np.add.at needed, which would be far slower)
                    s1[centers] += q
                    s2[centers, 0] += q[:, 0] * q[:, 0]
                    s2[centers, 1] += q[:, 1] * q[:, 1]
                    s2[centers, 2] += q[:, 2] * q[:, 2]
                    s2[centers, 3] += q[:, 0] * q[:, 1]
                    s2[centers, 4] += q[:, 0] * q[:, 2]
                    s2[centers, 5] += q[:, 1] * q[:, 2]
                    cnt[centers] += 1.0

        ok = cnt >= 3
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        if not np.any(ok):
            return normals

        c = cnt[ok][:, None]
        mu = s1[ok] / c
        e = s2[ok] / c
        cov = np.empty((int(ok.sum()), 3, 3))
        cov[:, 0, 0] = e[:, 0] - mu[:, 0] * mu[:, 0]
        cov[:, 1, 1] = e[:, 1] - mu[:, 1] * mu[:, 1]
        cov[:, 2, 2] = e[:, 2] - mu[:, 2] * mu[:, 2]
        cov[:, 0, 1] = cov[:, 1, 0] = e[:, 3] - mu[:, 0] * mu[:, 1]
        cov[:, 0, 2] = cov[:, 2, 0] = e[:, 4] - mu[:, 0] * mu[:, 2]
        cov[:, 1, 2] = cov[:, 2, 1] = e[:, 5] - mu[:, 1] * mu[:, 2]

        _, V = np.linalg.eigh(cov)          # batched; ascending eigenvalues
        nn = V[:, :, 0]                     # smallest eigenvector = plane normal
        ln = np.linalg.norm(nn, axis=1, keepdims=True)
        normals[ok] = np.divide(nn, ln, out=np.zeros_like(nn), where=ln > 1e-9)
        # a degenerate fit (collinear neighbours) yields a zero row — keep it upright
        degenerate = np.linalg.norm(normals, axis=1) < 1e-6
        normals[degenerate] = np.array([0.0, 0.0, 1.0])
        return normals

    # ------------------------------------------------------------------ #
    def _recompute(self):
        if len(self.points) < 50:
            return
        stamp = self.get_clock().now().to_msg()
        v = self.voxel
        keys = np.floor(self.points / v).astype(np.int64)
        # (no python set of voxel keys any more — _frontier_cells works on the
        # int64-coded arrays directly; building a 400k-tuple set every cycle was
        # pure overhead once the frontier scan stopped needing it)

        # --- surfels: one per occupied voxel, PCA normal over a voxel NEIGHBOURHOOD ---
        # This used to bucket points by voxel and PCA within the bucket. That could
        # never work: _cloud() voxel_downsample()s to exactly ONE point per voxel, so
        # every bucket had len(pts)==1, the `len(pts) >= 3` test was never true, and
        # EVERY surfel fell through to the (0,0,1) fallback. Measured on a live 25,674
        # surfel map: 100% had |n_z| == 1.0000, so coverage_node classified all of them
        # as floor (|n_z| > floor_normal_cos) and published gaps=0 forever — the whole
        # detection layer (§7A.3/7A.5/7A.6) was silently dead and explore only ever hit
        # its frontier fallback. Fit the plane over the surrounding voxels instead.
        centroids = self.points
        normals = self._neighbour_normals(centroids, keys)
        # orient toward the rover (observed side, §7A.2)
        flip = ((self.rover_pos - centroids) * normals).sum(1) < 0
        normals[flip] *= -1.0
        normals = normals.astype(np.float32)

        if len(centroids):
            self.pub_surfels.publish(make_cloud(
                centroids, self.frame, stamp,
                extra={"nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2]}))
            rgb = np.tile(np.array([170, 170, 170], np.uint8), (len(centroids), 1))
            self.pub_viz.publish(make_cloud(centroids, self.frame, stamp,
                                            extra={"rgb": pack_rgb(rgb)}))

        # --- frontier: empty voxels adjacent to occupied at the map boundary ---
        # Same boundary heuristic as before (an empty voxel touching occupied but
        # with <=2 occupied neighbours), but vectorised. The original nested Python
        # loop did ~#voxels x 6 tuple-builds and set lookups, plus 6 more per
        # candidate: at 400k occupied voxels that is >10M set operations and this
        # recompute measured ~70s against a 1.0s timer, which starved coverage and
        # explore of fresh data for over a minute at a time.
        frontier = self._frontier_cells(keys, v)
        if len(frontier):
            self.pub_frontier.publish(make_cloud(frontier, self.frame, stamp))
        self.get_logger().info(
            f"[map] pts={len(self.points)} surfels={len(centroids)} frontier={len(frontier)}",
            throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ #
    def _frontier_cells(self, keys: np.ndarray, v: float) -> np.ndarray:
        """Empty voxels touching occupied space with <= max_occ_nbrs occupied
        neighbours (the map's outer boundary), as world-frame centres."""
        nbr = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                        (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.int64)
        occ_keys = np.unique(keys, axis=0)
        lo = occ_keys.min(0) - 1
        kk = occ_keys - lo
        dims = kk.max(0) + 3                     # +2 margin for the -1/+1 shells

        def code(a):
            return (a[:, 0] * dims[1] + a[:, 1]) * dims[2] + a[:, 2]

        occ_code = np.sort(code(kk))

        def is_occ(a):
            inside = np.all((a >= 0) & (a < dims), axis=1)
            c = code(a)
            pos = np.clip(np.searchsorted(occ_code, c), 0, len(occ_code) - 1)
            return inside & (occ_code[pos] == c)

        # candidate empty neighbours of occupied voxels
        cand = np.unique((kk[:, None, :] + nbr[None, :, :]).reshape(-1, 3), axis=0)
        cand = cand[~is_occ(cand)]
        if len(cand) == 0:
            return np.empty((0, 3), np.float32)

        occ_n = np.zeros(len(cand), np.int32)
        for d in nbr:
            occ_n += is_occ(cand + d).astype(np.int32)
        cand = cand[occ_n <= self.max_occ_nbrs]
        if len(cand) == 0:
            return np.empty((0, 3), np.float32)

        if self.max_frontier and len(cand) > self.max_frontier:
            sel = np.linspace(0, len(cand) - 1, self.max_frontier).astype(int)
            cand = cand[sel]
        return ((cand + lo).astype(np.float32) + 0.5) * v


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
