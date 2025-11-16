import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ApriltagVisualizer(Node):
    def __init__(self):
        super().__init__("apriltag_visualizer")

        self.image_subscriber = self.create_subscription(
            Image, "/image_rect/compressed", self.image_callback, 10
        )

        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag/detections", self.tag_callback, 10
        )

        self.latest_image = None
        self.latest_tags = []

        self.bridge = CvBridge()

        self.line_color = (50, 255, 0)  # neon green bgr
        self.text_color = (50, 255, 0)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.latest_tags:
            self.draw_tags(self.latest_image, self.latest_tags)
        cv2.imshow("AprilTag Visualizer", self.latest_image)
        cv2.waitKey(1)

    def tag_callback(self, msg):
        self.latest_tags = msg.detections

    def draw_tags(self, image, tags):
        for detection in tags:
            corners = [(int(pt.x), int(pt.y)) for pt in detection.corners]

            for i in range(4):
                cv2.line(image, corners[i], corners[(i + 1) % 4], self.line_color, 2)

            center_x = int(detection.centre.x)
            center_y = int(detection.centre.y)
            cv2.putText(
                image,
                str(detection.id),
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.text_color,
                2,
            )


def visualize_from_stream(args=None):
    rclpy.init(args=args)
    node = ApriltagVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
