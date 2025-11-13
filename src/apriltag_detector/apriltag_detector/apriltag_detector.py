import cv2  # pip install opencv-python
import numpy as np
import rclpy
from apriltag import apriltag
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray, Point
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ApriltagDetector(Node):
    def __init__(self):
        super().__init__("apriltag_detector")

        self.image_subscriber = self.create_subscription(
            Image, "/image_rect", self.listener_callback, 10
        )
        self.detections_publisher = self.create_publisher(
            AprilTagDetectionArray, "/apriltag_detections", 10
        )

        self.bridge = CvBridge()

        self.family = "tagStandard41h12"
        self.apriltagdetector = apriltag(self.family)
        self.get_logger().info(f"Apriltag Detector Initialized.")

    def listener_callback(self, image_msg):
        gray_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="mono8")
        detected_tags = self.apriltagdetector.detect(gray_image)

        detection_array = AprilTagDetectionArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera"

        for tag in detected_tags:
            det = AprilTagDetection()

            det.family = self.family
            det.id = int(tag["id"])
            det.hamming = int(tag["hamming"])
            det.decision_margin = float(tag["margin"])
            det.goodness = float(
                tag["margin"]
            )  # how clearly the tag pattern stand out from the background

            det.centre = Point(x=float(tag["center"][0]), y=float(tag["center"][1]))

            corners = np.array(tag["lb-rb-rt-lt"]).reshape(4, 2)
            det.corners = [Point(x=float(x), y=float(y)) for x, y in corners]

            det.homography = [0.0] * 9  # left empty for now can be added if needed

            detection_array.detections.append(det)

        self.get_logger().info(
            f"Detected {len(detection_array.detections)} apriltags. Publishing to `/apriltag_detections`"
        )
        self.detections_publisher.publish(detection_array)


def main(args=None):
    rclpy.init(args=args)
    node = ApriltagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
