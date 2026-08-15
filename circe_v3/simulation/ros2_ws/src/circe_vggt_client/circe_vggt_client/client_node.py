"""circe_vggt_client — bridges the rover (ROS 2) to the off-board VGGT-SLAM host.

Runs on the SIM laptop (and, unchanged, on the real robot's companion compute).
It is a **thin HTTP client** — no torch / gtsam / vggt_slam here. It:

  * subscribes to the rover camera (`/rover/camera/image`) and POSTs JPEG
    keyframes to the SLAM host's `POST /frames`,
  * polls `GET /map` at the stop-and-map cadence and republishes the SLAM
    output into ROS as `/vggt/pose` (nav_msgs/Odometry) and `/vggt/cloud`
    (sensor_msgs/PointCloud2), both **relative scale**,
  * honours the host's `full_refresh` flag on loop closure (clears its cache so
    downstream mapping/localization rebuild + re-run motion self-cal, §5).

The SLAM host URL is a parameter, so the same node points at a WSL server (sim)
or a native-Linux box (real robot) with no code change — the whole sim→real
invariant of the design.
"""

from __future__ import annotations

import base64
import io
import threading

import numpy as np
import requests

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

try:
    from cv_bridge import CvBridge
    import cv2
except Exception:  # pragma: no cover - only available on the ROS box
    CvBridge = None


class VggtClient(Node):
    def __init__(self) -> None:
        super().__init__("circe_vggt_client")
        self.declare_parameter("slam_url", "http://127.0.0.1:8000")
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("map_poll_period", 1.0)     # s; stop-and-map cadence
        self.declare_parameter("voxel", 0.0)               # server-side cloud dedupe
        self.declare_parameter("max_points", 400000)
        self.declare_parameter("frame_id", "vggt_map")

        self.url = self.get_parameter("slam_url").value.rstrip("/")
        self.quality = int(self.get_parameter("jpeg_quality").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.bridge = CvBridge() if CvBridge else None

        # last map marker we've consumed, for delta requests
        self._after_submap = -1
        self._known_loops = 0
        self._server_session_id = None     # SLAM-side run id; see _poll_map
        self._session = requests.Session()
        self._start_slam_session()

        self.create_subscription(Image, "/rover/camera/image", self._on_image, 10)
        self.pub_pose = self.create_publisher(Odometry, "/vggt/pose", 10)
        self.pub_cloud = self.create_publisher(PointCloud2, "/vggt/cloud", 1)

        period = float(self.get_parameter("map_poll_period").value)
        self.create_timer(period, self._poll_map)
        self.get_logger().info(f"circe_vggt_client → SLAM host {self.url}")

    # --- session ---------------------------------------------------------
    def _start_slam_session(self) -> None:
        """Tell the SLAM host a new run is starting, so it archives the previous
        map instead of fusing this sim's scene into it. This node starts once per
        sim launch, so exactly one new session per run — which is the intent."""
        try:
            r = self._session.post(f"{self.url}/session",
                                   json={"label": "circe_sim", "reset": True},
                                   timeout=10.0)
            r.raise_for_status()
            info = r.json()
            self._server_session_id = info.get("session_id")
            self.get_logger().info(
                f"SLAM session {self._server_session_id} started "
                f"(archived run: {info.get('archived_run', {}).get('session_id')})")
        except requests.RequestException as e:
            # Non-fatal: the rover can still map, it just shares whatever run the
            # server already had. Loud, because that silently means a stale map.
            self.get_logger().error(
                f"POST /session failed ({e}) — SLAM may append to a PREVIOUS run's map")

    # --- frames up -------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        if self.bridge is None:
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        files = {"files": ("frame.jpg", io.BytesIO(jpg.tobytes()), "image/jpeg")}
        try:
            self._session.post(f"{self.url}/frames", files=files, timeout=2.0)
        except requests.RequestException as e:
            self.get_logger().warn(f"POST /frames failed: {e}", throttle_duration_sec=5.0)

    # --- map down --------------------------------------------------------
    def _poll_map(self) -> None:
        try:
            r = self._session.get(
                f"{self.url}/map",
                params={"after_submap": self._after_submap,
                        "known_loops": self._known_loops,
                        "voxel": float(self.get_parameter("voxel").value),
                        "max_points": int(self.get_parameter("max_points").value)},
                timeout=5.0)
            r.raise_for_status()
            m = r.json()
        except requests.RequestException as e:
            self.get_logger().warn(f"GET /map failed: {e}", throttle_duration_sec=5.0)
            return

        # The SLAM host restarted its run (server restart, or another client
        # opened a session). Submap ids restart at 0 there, so keeping our old
        # cursor would filter out the whole new map and we'd sit on a stale one
        # forever, seeing "no new submaps" while SLAM is actually mapping.
        sid = m.get("session_id")
        if sid is not None and sid != self._server_session_id:
            self.get_logger().warn(
                f"SLAM session changed {self._server_session_id} → {sid}; resetting map cache")
            self._server_session_id = sid
            self._after_submap = -1
            self._known_loops = 0
            return      # refetch from scratch next tick rather than trust this delta

        if m.get("full_refresh"):
            # loop closure re-optimised all poses → downstream must rebuild
            self._after_submap = -1
            self.get_logger().info("full_refresh (loop closure) — rebuilding map cache")

        self._known_loops = int(m.get("num_loops", self._known_loops))
        stamp = self.get_clock().now().to_msg()

        # newest camera pose → /vggt/pose
        submaps = m.get("submaps", [])
        if submaps:
            last = submaps[-1]
            self._after_submap = max(self._after_submap, int(last["submap_id"]))
            T = np.asarray(last["poses"][-1], dtype=float)   # 4x4, relative scale
            self.pub_pose.publish(self._odom_from_T(T, stamp))

        cloud = m.get("cloud", {})
        if cloud.get("n", 0) > 0:
            self.pub_cloud.publish(self._cloud_from_b64(cloud, stamp))

    # --- ROS message builders -------------------------------------------
    def _odom_from_T(self, T: np.ndarray, stamp) -> Odometry:
        od = Odometry()
        od.header = Header(stamp=stamp, frame_id=self.frame_id)
        od.child_frame_id = "rover"
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = \
            (float(v) for v in T[:3, 3])
        qw, qx, qy, qz = _quat_from_R(T[:3, :3])
        od.pose.pose.orientation.w = qw
        od.pose.pose.orientation.x = qx
        od.pose.pose.orientation.y = qy
        od.pose.pose.orientation.z = qz
        return od

    def _cloud_from_b64(self, cloud: dict, stamp) -> PointCloud2:
        xyz = np.frombuffer(base64.b64decode(cloud["xyz_f32_b64"]), np.float32).reshape(-1, 3)
        rgb = np.frombuffer(base64.b64decode(cloud["rgb_u8_b64"]), np.uint8).reshape(-1, 3)
        n = xyz.shape[0]
        rgb_f = np.zeros(n, np.float32)
        packed = (rgb[:, 0].astype(np.uint32) << 16 |
                  rgb[:, 1].astype(np.uint32) << 8 |
                  rgb[:, 2].astype(np.uint32))
        rgb_f = packed.view(np.float32)
        data = np.zeros(n, dtype=[("x", np.float32), ("y", np.float32),
                                  ("z", np.float32), ("rgb", np.float32)])
        data["x"], data["y"], data["z"], data["rgb"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb_f

        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        msg.height, msg.width = 1, n
        msg.is_dense = False
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = data.tobytes()
        return msg


def _quat_from_R(R: np.ndarray):
    """Rotation matrix → (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VggtClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
