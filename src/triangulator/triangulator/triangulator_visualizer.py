import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.node import Node

pkg_share = get_package_share_directory("triangulator")
field_path = os.path.join(pkg_share, "imgs", "field.png")


class TriangulatorVisualizer(Node):
    def __init__(self):
        super().__init__("triangulator_visualizer")

        self.pos_subsciber = self.create_subscription(
            Point, "/triangulated_pos", self.pos_callback, 10
        )

        self.field_image = cv2.imread(field_path, cv2.IMREAD_COLOR)
        self.pad = 120
        self.field_image = cv2.copyMakeBorder(
            self.field_image,
            self.pad,  # top
            self.pad,  # bottom
            self.pad,  # left
            self.pad,  # right
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        self.field_max_x = 7510
        self.field_max_y = 10520

        self.field_width_px = self.field_image.shape[1] - 2 * self.pad
        self.field_height_px = self.field_image.shape[0] - 2 * self.pad

        self.pos_color = (0, 0, 255)  # BGR

    def pos_callback(self, point_in_field):
        self.get_logger().info(f"pos: {point_in_field.x}, {point_in_field.y}")
        point_in_image = (
            int(
                self.pad

                + min(
                    max(
                        point_in_field.x / self.field_max_x * self.field_width_px,
                        0,
                    ),
                    self.field_width_px - 1,
                )
            ),
            int(
                self.pad
                + min(
                    max(
                        point_in_field.y / self.field_max_y * self.field_height_px,
                        0,
                    ),
                    self.field_height_px - 1,
                )
            ),
        )
        self.get_logger().info(f"pos in image: {point_in_image}")
        field_image_copy = self.field_image.copy()
        cv2.circle(
            field_image_copy, point_in_image, 5, self.pos_color, thickness=cv2.FILLED
        )

        cv2.putText(
            field_image_copy,
            f"({int(point_in_field.x)}, {int(point_in_field.y)})",
            (point_in_image[0] - 10, point_in_image[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.pos_color,
            2,
        )

        cv2.imshow("Triangulation Visualizer", field_image_copy)
        cv2.waitKey(5)


def visualize_from_stream(args=None):
    rclpy.init(args=args)
    node = TriangulatorVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
