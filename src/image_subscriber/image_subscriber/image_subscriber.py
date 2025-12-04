import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageSubscriber(Node):
    def __init__(self):
        """
        Args:
            callback: A function that takes in a RGB frame and performs an action.
        """
        super().__init__("image_subscriber")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/oak/rgb/image_raw", self.listener_callback, 10
        )
        self.bridge = CvBridge()
        self.counter = 0

    def listener_callback(self, msg):
        bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if bgr_image is not None:
            cv2.imwrite(f"./mapping/frame_{self.counter}.jpg", bgr_image)
            self.get_logger().info(f"Frame {self.counter} saved.")
            self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
