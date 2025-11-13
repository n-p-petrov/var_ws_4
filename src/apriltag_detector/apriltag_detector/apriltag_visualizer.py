import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ApriltagVisualizer(Node):
    def __init__(self):
        super().__init__("apriltag_visualizer")

        self.image_subscriber = self.create_subscription(
            CompressedImage, "/image_rect/compressed", self.image_callback, 10
        )

        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag_detections", self.tag_callback, 10
        )

        self.latest_image = None
        self.latest_tags = []

    def image_callback(self, image_buffer):
        np_arr = np.frombuffer(image_buffer.data, np.uint8)
        self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if self.latest_tags:
            self.draw_tags(self.latest_image, self.latest_tags)

        cv2.imshow("AprilTag Visualizer", self.latest_image)
        cv2.waitKey(1)

    def tag_callback(self, msg):
        self.latest_tags = msg.detections

    def draw_tags(self, image, tags):
        for detection in tags:
            corners = detection.corners
            corners = [(int(pt.x), int(pt.y)) for pt in corners]

            for i in range(4):
                cv2.line(image, corners[i], corners[(i + 1) % 4], (0, 255, 0), 2)

            center_x = int(detection.centre.x)
            center_y = int(detection.centre.y)
            cv2.putText(
                image,
                str(detection.id),
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )


def main(args=None):
    rclpy.init(args=args)
    node = ApriltagVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
