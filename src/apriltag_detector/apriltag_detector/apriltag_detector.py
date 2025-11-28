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

        # config
        self.apriltag_family = "tagStandard41h12"

        # topics
        self.image_topic = "/image_raw"  # pan-tilt camera (original)
        self.oak_image_topic = "/oak/image_raw"  # OAK camera
        self.apriltag_topic = "/apriltag/detections"  # merged output (no new topic)

        # image processing
        self.scaling_factor = 5  # upscaling factor used before detection

        # physical tag size (meters)
        self.tag_size_m = 0.160

        # merged publish interval (seconds)
        self.merge_publish_interval = 0.1

        # initialize cameras
        self.init_pan_tilt_camera()
        self.init_oak_camera()

        # subscribers
        self.image_subscriber = self.create_subscription(
            Image, self.image_topic, self.listener_callback, 10
        )
        self.oak_image_subscriber = self.create_subscription(
            Image, self.oak_image_topic, self.oak_listener_callback, 10
        )

        # publishers
        self.detections_publisher = self.create_publisher(
            AprilTagDetectionArray, self.apriltag_topic, 10
        )
        self.robot_orientation_publisher = self.create_publisher(Float32, "/orientation", 10)

        # optional pan angle input (affects orientation calculation)
        self.camera_pan_subscriber = self.create_subscription(
            Float32, "/camera_pan", self.camera_pan_callback, 10
        )
        self.camera_pan_angle = None

        # bridge, detector
        self.bridge = CvBridge()
        self.apriltagdetector = apriltag(self.apriltag_family)

        # bookkeeping
        self.total_num_tags = 0
        self.get_logger().info("Apriltag Detector (pan-tilt + OAK fusion) Initialized.")

        # tag orientation map (used by calculate_orientation for pan-tilt)
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

        # merged buffer: keyed by tag_id
        # each entry: {
        #   "source": "oak" or "pan_tilt",
        #   "oak": {center, corners, distance, hamming, margin}  # optional
        #   "pan_tilt": {center, corners, distance, hamming, margin, orientation_from_pan (optional)}
        # }
        self.all_tags_buffer = {}

        # last orientation computed from pan-tilt (for optional smoothing or reuse)
        self.last_orientation = None

        # timer to publish merged detections
        self.merge_publish_timer = self.create_timer(self.merge_publish_interval, self.publish_merged_detections)

    def init_pan_tilt_camera(self):
        fx = 298.904369
        fy = 300.029312
        cx = 333.732172
        cy = 257.804732

        self.camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.dist_coeffs = np.array([-0.230681, 0.034978, -0.001247, 0.001166, 0.000000]).reshape(-1, 1)

        w = 640
        h = 480
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0)

    def init_oak_camera(self):
        fx = 1011.2320556640625
        fy = 1011.1708374023438
        cx = 643.5490112304688
        cy = 373.5168151855469

        self.camera_matrix_oak = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # OAK distortion (given)
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

        # expected OAK image size (adjust if necessary)
        w_oak = 1280
        h_oak = 720
        self.new_camera_matrix_oak, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix_oak, self.dist_coeffs_oak, (w_oak, h_oak), 0)

    def camera_pan_callback(self, msg):
        self.camera_pan_angle = float(msg.data)

    # pan-tilt camera callback
    def listener_callback(self, img_msg: Image):
        try:
            gray_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().warn(f"CvBridge pan-tilt conversion failed: {e}")
            return

        gray_img = cv2.undistort(gray_img, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        enhanced_img = sharpen_img(gray_img, 31, 0.8, 0.2)
        enhanced_img = upscale_img(enhanced_img, self.scaling_factor)

        detected_tags = self.apriltagdetector.detect(enhanced_img)
        self.handle_camera_detections(detected_tags, camera="pan_tilt")

    # OAK camera callback
    def oak_listener_callback(self, img_msg: Image):
        try:
            gray_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().warn(f"CvBridge OAK conversion failed: {e}")
            return

        gray_img = cv2.undistort(gray_img, self.camera_matrix_oak, self.dist_coeffs_oak, None, self.new_camera_matrix_oak)
        enhanced_img = sharpen_img(gray_img, 31, 0.8, 0.2)
        enhanced_img = upscale_img(enhanced_img, self.scaling_factor)

        detected_tags = self.apriltagdetector.detect(enhanced_img)
        self.handle_camera_detections(detected_tags, camera="oak")

    # unified handler for both cameras
    def handle_camera_detections(self, detected_tags, camera: str):
        for tag in detected_tags:
            tag_id = int(tag["id"])
            hamming = int(tag["hamming"])
            margin = float(tag["margin"])

            cx_scaled = float(tag["center"][0] / self.scaling_factor)
            cy_scaled = float(tag["center"][1] / self.scaling_factor)

            corners_arr = np.array(tag["lb-rb-rt-lt"]).reshape(4, 2)
            corners_scaled = corners_arr / float(self.scaling_factor)

            distance, rvec = self.calculate_distance_for_camera(corners_scaled, (cx_scaled, cy_scaled), tag_id, camera=camera)

            # ignore detections where distance couldn't be computed
            if distance is None or distance < 0:
                continue

            # ensure buffer entry exists
            entry = self.all_tags_buffer.get(tag_id)
            if entry is None:
                entry = {"oak": None, "pan_tilt": None}
                self.all_tags_buffer[tag_id] = entry

            if camera == "oak":
                # OAK always wins for final detection: overwrite oak slot
                entry["oak"] = {
                    "center": (cx_scaled, cy_scaled),
                    "corners": corners_scaled.copy(),
                    "distance": distance,
                    "hamming": hamming,
                    "margin": margin,
                }
                # keep any existing orientation_from_pan in pan_tilt slot untouched (so orientation can still be published)
            else:
                # pan-tilt detection: store pan_tilt slot and compute orientation (if rvec available)
                orientation_from_pan = None
                if rvec is not None:
                    orientation_from_pan = self.calculate_orientation(rvec, (cx_scaled, cy_scaled), tag_id)
                    # calculate_orientation publishes orientation immediately; we still store it
                entry["pan_tilt"] = {
                    "center": (cx_scaled, cy_scaled),
                    "corners": corners_scaled.copy(),
                    "distance": distance,
                    "hamming": hamming,
                    "margin": margin,
                    "orientation_from_pan": orientation_from_pan,
                }

                # If OAK already has an entry for this tag, we DO NOT overwrite it for final detection,
                # but we still store the pan_tilt info (especially orientation_from_pan)
                # (no further action needed here)

    # central solvePnP distance resolver; returns (distance, rvec) where rvec may be None on failure
    def calculate_distance_for_camera(self, corners, img_space_center, tag_id, camera="pan_tilt"):
        S = self.tag_size_m
        object_points = np.array(
            [
                [-S / 2.0, S / 2.0, 0.0],  # lb
                [S / 2.0, S / 2.0, 0.0],  # rb
                [S / 2.0, -S / 2.0, 0.0],  # rt
                [-S / 2.0, -S / 2.0, 0.0],  # lt
            ],
            dtype=np.float32,
        )

        image_points = corners.astype(np.float32)

        if camera == "pan_tilt":
            cam_matrix = self.new_camera_matrix
        else:
            cam_matrix = self.new_camera_matrix_oak

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                cam_matrix,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                tvec = tvec.reshape(3)
                distance = float(np.sqrt(tvec[0] ** 2 + tvec[2] ** 2))
                return distance, rvec
            else:
                self.get_logger().warn(f"PnP failed for tag {tag_id} on camera {camera}")
                return -1.0, None
        except cv2.error as e:
            self.get_logger().warn(f"solvePnP error for tag {tag_id} on camera {camera}: {e}")
            return -1.0, None

    # orientation computation derived from pan-tilt rvec (publishes orientation)
    def calculate_orientation(self, rvec, img_space_center, tag_id):
        obj_space_R, _ = cv2.Rodrigues(rvec)
        inv_obj_space_R = np.linalg.inv(obj_space_R)

        obj_space_optic_axis = inv_obj_space_R @ np.array([0, 0, 1]).T
        obj_space_optic_axis = obj_space_optic_axis / np.linalg.norm(obj_space_optic_axis)
        # clamp numeric rounding for arccos
        zc = float(np.clip(obj_space_optic_axis[2], -1.0, 1.0))
        angle_to_optic_axis = np.arccos(zc)

        inv_intrinsic_matrix = np.linalg.inv(self.new_camera_matrix)
        ray = inv_obj_space_R @ inv_intrinsic_matrix @ np.array([img_space_center[0], img_space_center[1], 1.0]).T

        a = ray.copy()
        a[1] = 0.0
        b = obj_space_optic_axis.copy()
        b[1] = 0.0
        cross = np.cross(a, b)
        if cross[1] < 0:
            angle_to_optic_axis = -angle_to_optic_axis

        robot_orientation = angle_to_optic_axis + self.tag_orientation.get(tag_id, 0.0)

        if self.camera_pan_angle is not None:
            robot_orientation = robot_orientation - self.camera_pan_angle

        robot_orientation = wrap_angle(robot_orientation)

        # store/publish orientation
        self.last_orientation = robot_orientation
        robot_orientation_msg = Float32()
        robot_orientation_msg.data = float(robot_orientation)
        self.robot_orientation_publisher.publish(robot_orientation_msg)

        return robot_orientation

    # publish merged detections, preferring OAK entries when present
    def publish_merged_detections(self):
        if not self.all_tags_buffer:
            return

        detection_array = AprilTagDetectionArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera"  # single merged frame

        for tag_id, slots in list(self.all_tags_buffer.items()):
            # select source: oak if present, else pan_tilt
            source = "pan_tilt"
            chosen = None
            if slots.get("oak") is not None:
                source = "oak"
                chosen = slots["oak"]
            elif slots.get("pan_tilt") is not None:
                source = "pan_tilt"
                chosen = slots["pan_tilt"]
            else:
                continue  # nothing valid

            det = AprilTagDetection()
            det.family = self.apriltag_family
            det.id = int(tag_id)
            det.hamming = int(chosen.get("hamming", 0))
            det.decision_margin = float(chosen.get("margin", 0.0))

            cx, cy = chosen["center"]
            det.centre = Point(x=float(cx), y=float(cy))

            corners = chosen["corners"]
            det.corners = [Point(x=float(x), y=float(y)) for x, y in corners]

            det.homography = [0.0] * 9
            det.goodness = float(chosen.get("distance", -1.0))

            detection_array.detections.append(det)

            # publish orientation if pan-tilt saw it (orientation published when computed in calculate_orientation)
            # orientation is published during pan-tilt handling already, so do not re-publish here.

            self.total_num_tags += 1

        # publish merged array on the single topic
        self.detections_publisher.publish(detection_array)

        # clear buffer for next window
        self.all_tags_buffer.clear()

    def destroy_node(self):
        # stop timers/subscriptions if needed (used on shutdown)
        try:
            super().destroy_node()
        except Exception:
            pass


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


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
