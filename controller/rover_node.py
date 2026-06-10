#!/usr/bin/env python3
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class RoverNode(Node):
    def __init__(self) -> None:
        super().__init__('rover_controller')

        self._pub = self.create_publisher(Twist, '/rover/cmd_vel', 10)

        self._braked = True   # start braked — user must explicitly release
        self._lock = threading.Lock()
        self._stop_timer: threading.Timer | None = None

        self.get_logger().info('Rover controller node started (braked)')

    # ── public API (called from FastAPI thread) ──────────────────────────────────

    def move_step(self, linear: float, angular: float, duration: float) -> None:
        with self._lock:
            if self._braked:
                return
            self._cancel_pending_timer()
            self._publish(linear, angular)
            t = threading.Timer(duration, self._stop)
            t.daemon = True
            t.start()
            self._stop_timer = t

    def brake(self) -> None:
        with self._lock:
            self._cancel_pending_timer()
            self._braked = True
        self._publish(0.0, 0.0)
        self.get_logger().info('Rover braked')

    def release(self) -> None:
        with self._lock:
            self._braked = False
        self.get_logger().info('Rover brake released')

    def get_status(self) -> dict:
        with self._lock:
            return {'braked': self._braked}

    # ── internal ─────────────────────────────────────────────────────────────────

    def _stop(self) -> None:
        self._publish(0.0, 0.0)

    def _cancel_pending_timer(self) -> None:
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

    def _publish(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._pub.publish(msg)
