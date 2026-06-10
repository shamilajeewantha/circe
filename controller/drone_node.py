#!/usr/bin/env python3
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint,
    VehicleCommand, VehicleLocalPosition, VehicleStatus,
)

# Initial hover position: 5m in front of rover camera in NED frame.
# Rover at (-6, 0) yaw=30°; camera faces NED ~north-northeast.
# Drone at NED (2.782, -1.181, -2.0) = 2m altitude directly in camera view.
INITIAL_X = 2.782
INITIAL_Y = -1.181
INITIAL_Z = -2.0
INITIAL_YAW = 0.0


class DroneNode(Node):
    def __init__(self) -> None:
        super().__init__('drone_controller')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._ocm_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self._sp_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self._cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self._pos_cb, qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4',
            self._status_cb, qos)

        self._lock = threading.Lock()
        self._sp = [INITIAL_X, INITIAL_Y, INITIAL_Z, INITIAL_YAW]
        self._local_pos = VehicleLocalPosition()
        self._status = VehicleStatus()
        self._counter = 0
        self._arm_sent = False

        self.create_timer(0.1, self._tick)
        self.get_logger().info('Drone controller node started')

    # ── callbacks ───────────────────────────────────────────────────────────────

    def _pos_cb(self, msg):
        self._local_pos = msg

    def _status_cb(self, msg):
        self._status = msg

    # ── public API (called from FastAPI thread) ──────────────────────────────────

    def apply_increment(self, direction: str, step: float) -> None:
        with self._lock:
            sp = self._sp
            if direction == 'fwd':
                sp[0] += step
            elif direction == 'back':
                sp[0] -= step
            elif direction == 'right':
                sp[1] += step
            elif direction == 'left':
                sp[1] -= step
            elif direction == 'up':
                sp[2] -= step   # NED: z is down, so decrease = go higher
            elif direction == 'down':
                sp[2] += step
            elif direction == 'yaw_right':
                sp[3] += math.radians(step)
            elif direction == 'yaw_left':
                sp[3] -= math.radians(step)

    def get_status(self) -> dict:
        return {
            'armed': bool(self._status.arming_state == VehicleStatus.ARMING_STATE_ARMED),
            'preflight_ok': bool(self._status.pre_flight_checks_pass),
            'x_ned': float(self._local_pos.x),
            'y_ned': float(self._local_pos.y),
            'z_ned': float(self._local_pos.z),
            'nav_state': int(self._status.nav_state),
        }

    # ── internal ─────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._publish_heartbeat()

        with self._lock:
            x, y, z, yaw = self._sp

        self._publish_setpoint(x, y, z, yaw)

        if not self._arm_sent and self._counter >= 10:
            if self._status.pre_flight_checks_pass:
                self._engage_offboard()
                self._arm()
                self._arm_sent = True
            elif self._counter % 50 == 0:
                self.get_logger().info('Waiting for pre-flight checks...')

        if self._status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if self._counter % 100 == 0:
                self.get_logger().info(
                    f'Hovering z={self._local_pos.z:.2f}m NED  sp=({x:.2f},{y:.2f},{z:.2f})')

        self._counter += 1

    def _publish_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._ocm_pub.publish(msg)

    def _publish_setpoint(self, x, y, z, yaw) -> None:
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._sp_pub.publish(msg)

    def _arm(self) -> None:
        self._publish_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def _engage_offboard(self) -> None:
        self._publish_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Switching to offboard mode')

    def _publish_cmd(self, command, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._cmd_pub.publish(msg)
