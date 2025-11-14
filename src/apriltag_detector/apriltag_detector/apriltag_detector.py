import cv2
import numpy as np
import rclpy
from apriltag import apriltag
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray, Point
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from apriltag_detector.utils import sharpen_img, upscale_img


class ApriltagDetector(Node):
    def __init__(self):
        super().__init__("apriltag_detector")
        self.apriltag_family = "tagStandard41h12"
        self.image_topic = "/image_raw"
        self.apriltag_topic = "/apriltag/detections"
        self.scaling_factor = 5

        self.image_subscriber = self.create_subscription(
            Image, self.image_topic, self.listener_callback, 10
        )

        self.detections_publisher = self.create_publisher(
            AprilTagDetectionArray, self.apriltag_topic, 10
        )

        self.bridge = CvBridge()

        self.apriltagdetector = apriltag(self.apriltag_family)

        self.total_num_tags = 0

        self.get_logger().info("Apriltag Detector Initialized.")

    def listener_callback(self, img_msg):
        gray_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")

        enhanced_img = sharpen_img(gray_img, 31, 0.8, 0.2)
        enhanced_img = upscale_img(enhanced_img, self.scaling_factor)
        detected_tags = self.apriltagdetector.detect(enhanced_img)

        self.publish_apriltags(detected_tags)
        self.get_logger().info(f"Detected {len(detected_tags)} tags.")
        self.get_logger().info(f"Detected {self.total_num_tags} apriltags in total.")

    def publish_apriltags(self, detected_tags):
        detection_array = AprilTagDetectionArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera"

        for tag in detected_tags:
            self.total_num_tags = self.total_num_tags + 1
            det = AprilTagDetection()
            det.family = self.apriltag_family
            det.id = int(tag["id"])
            det.hamming = int(tag["hamming"])
            det.decision_margin = float(tag["margin"])
            det.goodness = float(tag["margin"])  # pattern clarity

            det.centre = Point(
                x=float(tag["center"][0] / self.scaling_factor),
                y=float(tag["center"][1] / self.scaling_factor),
            )

            corners = np.array(tag["lb-rb-rt-lt"]).reshape(4, 2)
            det.corners = [
                Point(
                    x=float(x / self.scaling_factor), y=float(y / self.scaling_factor)
                )
                for x, y in corners
            ]

            det.homography = [
                0.0
            ] * 9  # meaningless, has to be there (can also be comuted if needed)
            detection_array.detections.append(det)

        self.detections_publisher.publish(detection_array)


def main(args=None):
    rclpy.init(args=args)
    node = ApriltagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
