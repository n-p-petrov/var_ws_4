#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ApriltagVisualizer(Node):
    def __init__(self):
        super().__init__("apriltag_visualizer")

        # subscribe to the raw camera image
        self.image_subscriber = self.create_subscription(
            CompressedImage, "/image_raw/compressed", self.image_callback, 10
        )

        # subscribe to AprilTag detections from ApriltagDetector
        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag/detections", self.tag_callback, 10
        )

        self.latest_image = None
        self.latest_tags = []

        self.bridge = CvBridge()

        # colors for drawing (BGR)
        self.line_color = (50, 255, 0)  # neon green for box
        self.text_color = (50, 255, 0)  # neon green for ID
        self.dist_color = (0, 200, 255)  # yellow-ish for distance

<<<<<<< HEAD
<<<<<<< HEAD
=======
        # intrinsics
        fx = 298.904369
        fy = 300.029312
        cx = 333.732172
        cy = 257.804732
>>>>>>> main

        self.camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

<<<<<<< HEAD
    # Image callback: draw tags (if any) and show window
    def image_callback(self, msg: Image):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
=======
        # intrinsics
        fx = 298.904369
        fy = 300.029312
        cx = 333.732172
        cy = 257.804732

        self.camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

=======
>>>>>>> main
        # undistorted images,
        self.dist_coeffs = np.array(
            [-0.230681, 0.034978, -0.001247, 0.001166, 0.000000]
        ).reshape(-1, 1)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.latest_image = cv2.undistort(
            self.latest_image, self.camera_matrix, self.dist_coeffs
        )
<<<<<<< HEAD
>>>>>>> d64bcd6 (fix triangulator visualizer, add calibration to image raw in apriltag visualizer)
=======
>>>>>>> main

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
