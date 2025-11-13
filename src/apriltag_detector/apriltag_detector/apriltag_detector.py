import argparse

import cv2  # pip install opencv-python
import numpy as np
import rclpy
from apriltag import apriltag  # pip install apriltag
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import Point32, Polygon
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ApriltagDetector(Node):
    def __init__(self):
        super().__init__("apriltag_detector")

        self.image_subscriber = self.create_subscription(
            CompressedImage, "/image_rect/compressed", self.listener_callback, 10
        )
        self.detections_publisher = self.create_publisher(
            AprilTagDetectionArray, "/apriltag_detections", 10
        )

        self.apriltagdetector = apriltag("tagStandard41h12")

    def listener_callback(self, image_buffer):
        np_arr = np.frombuffer(image_buffer.data, np.uint8)
        gray_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        results = self.apriltagdetector.detect(gray_image)

        detection_array = AprilTagDetectionArray()

        for r in results:
            polygon = Polygon()
            for x, y in r.corners:
                polygon.points.append(Point32(x=float(x), y=float(y), z=0.0))
            detection_array.detections.append(polygon)

        self.detections_publisher.publish(detection_array)
        self.get_logger().info(
            f"Published {len(detection_array.detections)} apriltag detections."
        )


def main(args=None):
    rclpy.init(args=args)
    node = ApriltagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
