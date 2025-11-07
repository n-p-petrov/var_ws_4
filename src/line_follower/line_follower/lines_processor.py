from .hough_tools import *
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image


class LinesProcessor(Node):
    def __init__(self):
        super().__init__("lines_processor")

        self.bridge = CvBridge()

        self.img_nmr  = 0

        self.subscription = self.create_subscription(
            CompressedImage, "/image_rect/compressed",
            self.listener_callback, 1,
        )

        # self.publisher_ = self.create_publisher(
        #     Image, "/lines_image", 1,
        # )

        self.declare_parameter("use_sobel", False) # it allows to choose sobel without changing code paths anywhere else

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR_RGB)
        gray_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if gray_image is None:
            self.get_logger().error("Failed to decode image.")
            return

        use_sobel = self.get_parameter("use_sobel").get_parameter_value().bool_value # if we want to use sobel (use_sobel = True)
        processed_rgb = create_line_image(rgb_image, use_sobel=use_sobel)
        processed_rgb = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        
        fname = f"/home/ws/var_ws_4/imgs/{self.img_nmr:07}.jpeg"
        self.img_nmr += 1
        cv2.imwrite(fname, processed_rgb)
        print("saved to " + fname)

        # _, compressed_img = cv2.imencode(".jpg", processed_rgb)

        # out_msg = self.bridge.cv2_to_compressed_imgmsg(processed_rgb, dst_format="jpeg")
        # out_msg = self.bridge.cv2_to_imgmsg(processed_rgb, encoding="8UC3", header=msg.header)

        # out_msg = CompressedImage()
        # out_msg.header = msg.header
        # out_msg.format = "jpg"
        # out_msg.data = np.array(compressed_img).tobytes()

        # self.publisher_.publish(out_msg)
        # self.get_logger().info("Processed image published.")

        # image_msg = self.bridge.cv2_to_imgmsg(processed_rgb, encoding="rgb8")
        # image_msg.header = msg.header  # copy timestamp and frame_id
        # self.publisher_.publish(image_msg)
        # self.get_logger().info("Processed image published.")


def create_line_image(rgb_image, use_sobel=False):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    origin = image_center(gray_image)

    # IF SOBEL IS ON; default keeps our current Canny path
    if use_sobel:
        edges = image_preprocess_sobel(gray_image)
    else:
        edges = image_preprocess(gray_image)

    r_theta = polar_lines(edges, origin)
    r_theta_fc = polar_lines(edges, origin=origin, full_circle=True)
    drawn = draw_lines(r_theta, r_theta_fc, rgb_image, origin=origin)
    return drawn


def main(args=None):
    rclpy.init(args=args)
    node = LinesProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()
