import math
import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import Point
from rclpy.node import Node

pkg_share = get_package_share_directory("triangulator")
field_path = os.path.join(pkg_share, "imgs", "field.png")


class TriangulatorVisualizer(Node):
    def __init__(self):
        super().__init__("triangulator_visualizer")

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

        self.init_apriltag_fields()
        self.init_pos_fields()

        self.field_to_display = self.field_image.copy()
        self.display_timer = self.create_timer(1 / 20, self.display_callback)

    def display_callback(self):
        """
        Displays the field with whatever was drawn on it so far.
        Cleans it up for future drawings.
        """
        cv2.imshow("Triangulation Visualizer", self.field_to_display)
        cv2.waitKey(5)
        self.field_to_display = self.field_image.copy()

    # --- Apriltag visualization functions ---

    def init_apriltag_fields(self):
        self.apriltag_detections_subscriber = self.create_subscription(
            AprilTagDetectionArray,
            "/apriltag/detections",
            self.draw_apriltags_on_field,
            10,
        )

        self.apriltag_coordinates = (
            {  # 00 at arnouds desk, 7510 x 10520  at window at computers
                1: (3755, 9680),
                2: (760, 8080),
                3: (6750, 8080),
                4: (760, 5250),
                5: (6750, 4500),
                6: (620, 3380),
                7: (7510 - 110, 3380),
                8: (890, 760),
                9: (3755, 760),
                10: (7510 - 880, 760),
            }
        )
        self.apriltag_coordinates = {
            k: (
                x / self.field_max_x * self.field_width_px,
                y / self.field_max_y * self.field_height_px,
            )
            for k, (x, y) in self.apriltag_coordinates.items()
        }

        self.detected_apriltag_color = (0, 255, 0)  # green
        self.undetected_apriltag_color = (0, 0, 255)  # red
        self.apriltag_border_color = (0, 0, 0)  # black
        self.apriltag_square_size = 20

        self.apriltag_detections = None

    def draw_apriltags_on_field(self, msg: AprilTagDetectionArray):
        """
        Draws solid squares on the field image at each AprilTag's coordinates.
        """

        for tag_id, (px, py) in self.apriltag_coordinates.items():
            px = int(px)
            py = int(py)

            if self.apriltag_detections and any(
                d.id[0] == tag_id for d in msg.detections
            ):
                color = self.detected_apriltag_color
            else:
                color = self.undetected_apriltag_color

            x1 = px - self.apriltag_square_size
            y1 = py - self.apriltag_square_size
            x2 = px + self.apriltag_square_size
            y2 = py + self.apriltag_square_size

            cv2.rectangle(
                self.field_to_display, (x1, y1), (x2, y2), color, thickness=-1
            )
            cv2.rectangle(
                self.field_to_display,
                (x1, y1),
                (x2, y2),
                self.apriltag_border_color,
                thickness=2,
            )

    # --- Position visualization functions ---

    def init_pos_fields(self):
        self.pos_subsciber = self.create_subscription(
            Point, "/triangulated_pos", self.pos_callback, 10
        )

        self.filtered_pos_subsciber = self.create_subscription(
            Point, "/filtered_pose", self.pos_callback_filtered, 10
        )

        self.filtered_point_in_image = None
        self.filtered_point_in_field = None
        self.filtered_theta = None

        self.pos_color = (0, 0, 255)  # BGR
        self.filtered_pos_color = (255, 0, 0)  # BGR

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
        cv2.circle(
            self.field_to_display,
            point_in_image,
            5,
            self.pos_color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            self.field_to_display,
            f"({int(point_in_field.x)}, {int(point_in_field.y)})",
            (point_in_image[0], point_in_image[1] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.pos_color,
            2,
        )

        if self.filtered_point_in_image and self.filtered_point_in_field:
            cv2.circle(
                self.field_to_display,
                self.filtered_point_in_image,
                5,
                self.filtered_pos_color,
                thickness=cv2.FILLED,
            )

            cv2.putText(
                self.field_to_display,
                f"({int(self.filtered_point_in_field.x)}, {int(self.filtered_point_in_field.y)})",
                (self.filtered_point_in_image[0], self.filtered_point_in_image[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.filtered_pos_color,
                2,
            )
            if self.filtered_theta is not None:
                self._draw_orientation_arrow(
                    self.field_to_display,
                    self.filtered_point_in_image,
                    self.filtered_theta,
                    self.filtered_pos_color,
                )

    def _draw_orientation_arrow(self, img, origin, theta, color):
        """
        Draw an arrow showing orientation.

        theta: in radians, 0 = pointing right, positive CCW in field frame.
        Image coordinates: x right, y down.
        So y must be inverted relative to math convention.
        """
        arrow_length = 80  # pixels; tweak to taste

        x0, y0 = origin

        # Convert from field frame (standard math: up is +y) to image (down is +y)
        x1 = int(x0 + arrow_length * math.cos(theta))
        y1 = int(y0 + arrow_length * math.sin(theta))
        self.get_logger().info(
            f"Drawing arrow from ({x0}, {y0}) to ({x1}, {y1}) with theta={theta} rad"
        )

        cv2.arrowedLine(
            img,
            (x0, y0),
            (x1, y1),
            color,
            thickness=2,
            tipLength=0.3,
        )

    def pos_callback_filtered(self, point_in_field):
        self.filtered_point_in_field = point_in_field
        self.filtered_theta = point_in_field.z  # NEW

        self.get_logger().info(f"filtered_pos: {point_in_field.x}, {point_in_field.y}")
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
        self.get_logger().info(f"filtered pos in image: {point_in_image}")

        self.filtered_point_in_image = point_in_image


def visualize_from_stream(args=None):
    rclpy.init(args=args)
    node = TriangulatorVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
