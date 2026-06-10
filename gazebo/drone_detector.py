#!/usr/bin/env python3
"""
Subscribes to /rover_camera/image, runs YOLOv8, publishes drone pixel position.
Falls back to HSV bright-object detection if YOLO misses the drone.

Install: pip3 install ultralytics --break-system-packages
"""

from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np


class DroneDetector(Node):
    def __init__(self):
        super().__init__('drone_detector')

        from ultralytics import YOLO
        model_path = str(Path(__file__).parent.parent / 'drone_detection' / 'best.pt')
        self.model = YOLO(model_path)
        self.get_logger().info(f'Drone detection model loaded: {model_path}')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/rover_camera/image', self.image_callback, 10)
        self.pixel_pub = self.create_publisher(Point, '/drone_pixel', 10)
        self.debug_pub = self.create_publisher(Image, '/rover_camera/debug', 10)

        self.frame_count = 0
        self.get_logger().info('Waiting for images on /rover_camera/image...')

    def image_callback(self, msg):
        self.frame_count += 1
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        debug_frame = frame.copy()
        u, v, conf = None, None, 0.0

        # YOLO detection
        results = self.model(frame, verbose=False)
        best_box, best_conf = None, 0.0
        for r in results:
            for box in r.boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    best_conf = c
                    best_box = box

        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            conf = best_conf
            cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(debug_frame, f'YOLO ({u:.0f},{v:.0f}) {conf:.2f}',
                        (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # HSV fallback if YOLO found nothing
        if u is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 50:
                    M = cv2.moments(largest)
                    if M['m00'] > 0:
                        u = M['m10'] / M['m00']
                        v = M['m01'] / M['m00']
                        conf = 0.5
                        cv2.circle(debug_frame, (int(u), int(v)), 8, (255, 0, 0), 2)
                        cv2.putText(debug_frame, f'HSV ({u:.0f},{v:.0f})',
                                    (int(u) + 10, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if u is not None:
            pt = Point()
            pt.x = u
            pt.y = v
            pt.z = conf
            self.pixel_pub.publish(pt)
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'Drone at pixel ({u:.1f}, {v:.1f}) conf={conf:.2f}')
        else:
            if self.frame_count % 30 == 0:
                self.get_logger().info('No detection this frame')

        if self.frame_count % 30 == 0:
            cv2.imwrite('/tmp/rover_cam_debug.jpg', debug_frame)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8'))
        except Exception:
            pass


def main():
    rclpy.init()
    rclpy.spin(DroneDetector())


if __name__ == '__main__':
    main()
