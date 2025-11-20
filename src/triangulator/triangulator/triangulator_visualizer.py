import cv2
import rclpy
from rclpy.node import Node


class TriangulationVisualizer(Node):
    def __init__(self):
        super().__init__("triangulation_visualizer")

        self.pos_subsciber = self.create_subscription(
            tuple, "/triangulated_pos", self.pos_callback, 10
        )  # (x, y)

        self.field_image = cv2.imread("./imgs/field.png", cv2.IMREAD_COLOR)
        cv2.imshow(self.field_image)
        self.field_max_x = 7510
        self.field_max_y = 10520

        self.pos_color = (0, 0, 255)  # BGR

    def pos_callback(self, pos_in_field):
        pos_in_image = (
            pos_in_field[0] / self.field_max_x * self.field_image.shape[0],
            pos_in_field[1] / self.field_max_y * self.field_image.shape[1],
        )
        field_image_copy = self.field_image.copy()
        cv2.circle(
            field_image_copy, pos_in_image, 5, self.pos_color, thickness=cv2.FILLED
        )

        cv2.putText(
            field_image_copy,
            f"pos: {pos_in_field}",
            (pos_in_image[0] - 10, pos_in_image[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.pos_color,
            2,
        )

        cv2.imshow("Triangulation Visualizer", field_image_copy)
        cv2.waitKey(5)


def visualize_from_stream(args=None):
    rclpy.init(args=args)
    node = TriangulationVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
