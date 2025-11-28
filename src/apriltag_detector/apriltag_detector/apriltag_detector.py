#!/usr/bin/env python3
import math

import cv2
import numpy as np
import rclpy
from apriltag import apriltag
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray, Point
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from apriltag_detector.utils import sharpen_img, upscale_img


class ApriltagDetector(Node):
    def __init__(self):
        super().__init__("apriltag_detector")

        self.apriltag_family = "tagStandard41h12"

        # corrected OAK topic
        self.image_topic = "/image_raw"            # pan-tilt camera
        self.oak_image_topic = "/oak/rgb/image_raw"  # OAK corrected

        self.apriltag_topic = "/apriltag/detections"

        self.scaling_factor = 5
        self.tag_size_m = 0.160

        self.merge_publish_interval = 0.1

        self.init_pan_tilt_camera()
        self.init_oak_camera()

        self.image_subscriber = self.create_subscription(
            Image, self.image_topic, self.listener_callback, 10
        )
        self.oak_image_subscriber = self.create_subscription(
            Image, self.oak_image_topic, self.oak_listener_callback, 10
        )

        self.detections_publisher = self.create_publisher(
            AprilTagDetectionArray, self.apriltag_topic, 10
        )
        self.robot_orientation_publisher = self.create_publisher(Float32, "/orientation", 10)

        self.camera_pan_subscriber = self.create_subscription(
            Float32, "/camera_pan", self.camera_pan_callback, 10
        )
        self.camera_pan_angle = None

        self.bridge = CvBridge()
        self.apriltagdetector = apriltag(self.apriltag_family)

        self.tag_orientation = {
            1: np.pi / 2,
            2: np.pi,
            4: np.pi,
            6: np.pi,
            3: 0.0,
            5: 0.0,
            7: 0.0,
            8: -np.pi / 2,
            9: -np.pi / 2,
            10: -np.pi / 2,
        }

        self.all_tags_buffer = {}
        self.last_orientation = None

        self.merge_publish_timer = self.create_timer(
            self.merge_publish_interval, self.publish_merged_detections
        )

    def init_pan_tilt_camera(self):
        fx = 298.904369
        fy = 300.029312
        cx = 333.732172
        cy = 257.804732

        self.camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self.dist_coeffs = np.array([-0.230681, 0.034978, -0.001247, 0.001166, 0.0]).reshape(-1, 1)

        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (640, 480), 0
        )

    def init_oak_camera(self):
        fx = 1011.2320556640625
        fy = 1011.1708374023438
        cx = 643.5490112304688
        cy = 373.5168151855469

        self.camera_matrix_oak = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        self.dist_coeffs_oak = np.array(
            [
                -3.010956048965454,
                4.706149101257324,
                0.0003369069017935544,
                -0.00034450265229679679644,
                3.803163766860962,
                -3.1048195362091064,
                5.033864974975586,
                3.432560920715332,
            ]
        ).reshape(-1, 1)

        self.new_camera_matrix_oak, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix_oak, self.dist_coeffs_oak, (1280, 720), 0
        )

    def camera_pan_callback(self, msg):
        self.camera_pan_angle = float(msg.data)

    # pan-tilt
    def listener_callback(self, img_msg: Image):
        try:
            gray_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        except Exception:
            return

        gray_img = cv2.undistort(gray_img, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        enhanced_img = sharpen_img(gray_img, 31, 0.8, 0.2)
        enhanced_img = upscale_img(enhanced_img, self.scaling_factor)

        detected_tags = self.apriltagdetector.detect(enhanced_img)
        self.handle_camera_detections(detected_tags, camera="pan_tilt")

    # OAK corrected
    def oak_listener_callback(self, img_msg: Image):
        try:
            gray_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        except Exception:
            return

        gray_img = cv2.undistort(
            gray_img,
            self.camera_matrix_oak,
            self.dist_coeffs_oak,
            None,
            self.new_camera_matrix_oak,
        )
        enhanced_img = sharpen_img(gray_img, 31, 0.8, 0.2)
        enhanced_img = upscale_img(enhanced_img, self.scaling_factor)

        detected_tags = self.apriltagdetector.detect(enhanced_img)
        self.handle_camera_detections(detected_tags, camera="oak")

    # unified handler
    def handle_camera_detections(self, detected_tags, camera: str):
        for tag in detected_tags:
            tag_id = int(tag["id"])
            hamming = int(tag["hamming"])
            margin = float(tag["margin"])

            cx, cy = tag["center"]
            cx /= self.scaling_factor
            cy /= self.scaling_factor

            corners = np.array(tag["lb-rb-rt-lt"]).reshape(4, 2) / self.scaling_factor

            distance, rvec = self.calculate_distance_for_camera(corners, (cx, cy), tag_id, camera)
            if distance is None or distance < 0:
                continue

            if tag_id not in self.all_tags_buffer:
                self.all_tags_buffer[tag_id] = {"oak": None, "pan_tilt": None}

            entry = self.all_tags_buffer[tag_id]

            orientation_value = None
            if rvec is not None:
                orientation_value = self.calculate_orientation(rvec, (cx, cy), tag_id)

            slot = {
                "center": (cx, cy),
                "corners": corners.copy(),
                "distance": distance,
                "hamming": hamming,
                "margin": margin,
                "orientation": orientation_value,
            }

            if camera == "oak":
                entry["oak"] = slot
            else:
                entry["pan_tilt"] = slot

    def calculate_distance_for_camera(self, corners, img_space_center, tag_id, camera):
        S = self.tag_size_m
        object_points = np.array(
            [
                [-S / 2, S / 2, 0],
                [S / 2, S / 2, 0],
                [S / 2, -S / 2, 0],
                [-S / 2, -S / 2, 0],
            ],
            dtype=np.float32,
        )

        image_points = corners.astype(np.float32)

        cam_matrix = (
            self.new_camera_matrix if camera == "pan_tilt" else self.new_camera_matrix_oak
        )

        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                cam_matrix,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return -1, None
            tvec = tvec.reshape(3)
            distance = float(np.sqrt(tvec[0] ** 2 + tvec[2] ** 2))
            return distance, rvec
        except Exception:
            return -1, None

    def calculate_orientation(self, rvec, img_space_center, tag_id):
        R, _ = cv2.Rodrigues(rvec)
        invR = np.linalg.inv(R)

        optic = invR @ np.array([0, 0, 1], dtype=float)
        optic /= np.linalg.norm(optic)
        zc = float(np.clip(optic[2], -1.0, 1.0))
        angle = np.arccos(zc)

        invK = np.linalg.inv(self.new_camera_matrix)
        ray = invR @ (invK @ np.array([img_space_center[0], img_space_center[1], 1.0]))

        a = ray.copy()
        a[1] = 0
        b = optic.copy()
        b[1] = 0
        if np.cross(a, b)[1] < 0:
            angle = -angle

        angle += self.tag_orientation.get(tag_id, 0.0)

        if self.camera_pan_angle is not None:
            angle -= self.camera_pan_angle

        angle = wrap_angle(angle)

        self.last_orientation = angle
        msg = Float32()
        msg.data = float(angle)
        self.robot_orientation_publisher.publish(msg)

        return angle

    # merged publisher (OAK wins)
    def publish_merged_detections(self):
        if not self.all_tags_buffer:
            return

        arr = AprilTagDetectionArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = "camera"

        for tag_id, slots in self.all_tags_buffer.items():
            chosen = slots["oak"] if slots["oak"] is not None else slots["pan_tilt"]
            if chosen is None:
                continue

            det = AprilTagDetection()
            det.family = self.apriltag_family
            det.id = tag_id
            det.hamming = chosen["hamming"]
            det.decision_margin = chosen["margin"]

            cx, cy = chosen["center"]
            det.centre = Point(x=float(cx), y=float(cy))

            for x, y in chosen["corners"]:
                det.corners.append(Point(x=float(x), y=float(y)))

            det.homography = [0.0] * 9
            det.goodness = chosen["distance"]

            arr.detections.append(det)

        self.detections_publisher.publish(arr)
        self.all_tags_buffer.clear()


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = ApriltagDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
