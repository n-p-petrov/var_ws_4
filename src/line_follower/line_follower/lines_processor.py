import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from .hough_tools import image_center, get_edges, polar_lines, draw_lines


class LinesProcessor(Node):
    def __init__(self):
        super().__init__("lines_processor")

        self.bridge = CvBridge()
        self.img_nmr = 0

        # Which line-detection method to use: "canny" or "skeleton"
        self.declare_parameter("line_method", "canny")

        # Subscriber: compressed RGB images from camera / rosbag
        self.subscription = self.create_subscription(
            CompressedImage,
            "/image_rect/compressed",
            self.listener_callback,
            1,
        )

        # Decide where to save debug images
        # Use /output if running in Docker, otherwise default course path.
        if os.path.isdir("/output"):
            self.save_dir = "/output"
        else:
            self.save_dir = "/home/ws/var_ws_4/imgs"

        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"Saving debug images to {self.save_dir}")

# ROS callback
    def listener_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if bgr_image is None:
            self.get_logger().error("Failed to decode image.")
            return

        # Convert to RGB for consistency with our drawing code
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        line_method = (
            self.get_parameter("line_method")
            .get_parameter_value()
            .string_value
        )

        processed_rgb = create_line_image(rgb_image, method=line_method)

        # Save processed image for later inspection
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        fname = os.path.join(self.save_dir, f"{self.img_nmr:07}.jpeg")
        self.img_nmr += 1
        cv2.imwrite(fname, processed_bgr)
        self.get_logger().info(f"saved to {fname}")


def create_line_image(rgb_image, method: str = "canny"):
    """
    Create a debug image with detected lines drawn.
    method: "canny" (baseline) or "skeleton" (new algorithm).
    """
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    origin = image_center(gray_image)

    # pick preprocessing algorithm
    edges = get_edges(gray_image, method=method)

    # run Hough
    r_theta = polar_lines(edges, origin)
    r_theta_fc = polar_lines(edges, origin=origin, full_circle=True)

    # draw on a copy so we don't modify input
    drawn = draw_lines(r_theta, r_theta_fc, rgb_image.copy(), origin=origin)
    return drawn


def main(args=None):
    rclpy.init(args=args)
    node = LinesProcessor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
