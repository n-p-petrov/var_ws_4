# Image subscriber directly from /image_raw/compressed topic

from time import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ImageSubscriber(Node):
    def __init__(self, callback):
        """
        Args:
            callback: A function that takes in a RGB frame and performs an action.
        """
        super().__init__("image_subscriber")
        self.bridge = CvBridge()
        self.callback = callback
        self.subscription = self.create_subscription(
            CompressedImage, "/image_rect/compressed", self.listener_callback, 1
        )

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is not None:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.callback(rgb_image)
            self.get_logger().info("Pan tilt camera feed callback successfully executed for a frame.")
        else:
            self.get_logger().info("Pan tilt camera feed failed due to unsuccessful cv2.imdecode.")

