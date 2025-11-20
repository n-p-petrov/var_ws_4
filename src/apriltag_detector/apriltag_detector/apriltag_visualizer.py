#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node

from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge

from sensor_msgs.msg import Image


class ApriltagVisualizer(Node):
    def __init__(self):
        super().__init__("apriltag_visualizer")

        # subscribe to the raw camera image
        self.image_subscriber = self.create_subscription(
            CompressedImage, "/image_rect/compressed", self.image_callback, 10
        )

        # subscribe to AprilTag detections from ApriltagDetector
        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag/detections", self.tag_callback, 10
        )
        
        

        self.latest_image = None
        self.latest_tags = []

        self.bridge = CvBridge()

        # colors for drawing (BGR)
        self.line_color = (50, 255, 0)   # neon green for box
        self.text_color = (50, 255, 0)   # neon green for ID
        self.dist_color = (0, 200, 255)  # yellow-ish for distance


        self.get_logger().info("Apriltag Visualizer Initialized.")

    # Image callback: draw tags (if any) and show window
    def image_callback(self, msg: Image):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.latest_tags:
            self.draw_tags(self.latest_image, self.latest_tags)

        cv2.imshow("AprilTag Visualizer", self.latest_image)
        # 1 ms wait so OpenCV can process window events
        cv2.waitKey(1)

    # Detections callback: just store latest detections
    def tag_callback(self, msg: AprilTagDetectionArray):
        self.latest_tags = msg.detections

    # Helper: draw bounding box, ID and distance on the image
    def draw_tags(self, image, tags):
        for detection in tags:
            # corners as integer pixel coordinates
            corners = [(int(pt.x), int(pt.y)) for pt in detection.corners]

            # draw the 4 edges of the tag
            for i in range(4):
                cv2.line(
                    image,
                    corners[i],
                    corners[(i + 1) % 4],
                    self.line_color,
                    2,
                )

            # center point
            cx = int(detection.centre.x)
            cy = int(detection.centre.y)

            # draw tag ID
            cv2.putText(
                image,
                f"ID {detection.id}",
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.text_color,
                2,
            )

            # distance was packed into 'goodness' by the detector
            distance = detection.goodness
            if distance >= 0.0:
                cv2.putText(
                    image,
                    f"{distance:.2f} m",
                    (cx, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.dist_color,
                    2,
                )


def visualize_from_stream(args=None):
    rclpy.init(args=args)
    node = ApriltagVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    visualize_from_stream()
