#!/usr/bin/env python3
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
        self.image_topic = "/image_raw"
        self.apriltag_topic = "/apriltag/detections"
        self.scaling_factor = 5  # for upscaling the image
        # physical size of your printed tag (meters): 28.8 cm -> 0.288 m
        self.tag_size_m = 0.160

        # image subscriber
        self.image_subscriber = self.create_subscription(
            Image, self.image_topic, self.listener_callback, 10
        )

        # detections publisher
        self.detections_publisher = self.create_publisher(
            AprilTagDetectionArray, self.apriltag_topic, 10
        )

        self.bridge = CvBridge()
        self.apriltagdetector = apriltag(self.apriltag_family)

        # HARDCODED CAMERA INTRINSICS (from calibration of image_raw)
        # width = 640, height = 480
        fx = 298.904369
        fy = 300.029312
        cx = 333.732172
        cy = 257.804732

        self.camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        # undistorted images,
        self.dist_coeffs = np.array(
            [-0.230681, 0.034978, -0.001247, 0.001166, 0.000000]
        ).reshape(-1, 1)

        self.total_num_tags = 0

        self.get_logger().info("Apriltag Detector with PnP Initialized.")
        self.get_logger().info(
            f"Using hardcoded intrinsics for /image_raw: "
            f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
        )

        # image size
        w = 640
        h = 480

        # compute the new camera matrix for the undistorted image
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            0,  # alpha=0 keeps a clean free-of-black-borders image
        )

        self.tag_orientation = {
            # RADIANS
            # angle between vector perpendicular to apriltag facing apriltag and x axis of the field

            # goal closer to where we sit
            1: np.pi / 2,

            # door side 
            2: np.pi,
            4: np.pi,
            6: np.pi, 

            # window side
            3: 0.0,
            5: 0.0,
            7: 0.0,

            # arnold desk goal
            8: -np.pi / 2,
            9: -np.pi / 2,
            10: -np.pi / 2,
        }

        # TODO fill this in (rads)
        self.camera_pan_subscriber = self.create_subscription(Float32, "/camera_pan", self.camera_pan_callback, 10)
        self.camera_pan_angle = None

    def camera_pan_callback(self, msg):
        self.camera_pan_angle = msg.data

    # Image callback: detect tags and publish detections
    def listener_callback(self, img_msg: Image):
        # convert ROS image -> OpenCV gray
        gray_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        gray_img = cv2.undistort(  # undistort via K and dist, make sure newK is self.new_camera_matrix for the output image
            gray_img, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix
        )
        # enhance for robustness (same as your friend's version)
        enhanced_img = sharpen_img(gray_img, 31, 0.8, 0.2)
        enhanced_img = upscale_img(enhanced_img, self.scaling_factor)

        # detect tags on enhanced, upscaled image
        detected_tags = self.apriltagdetector.detect(enhanced_img)

        # publish ROS detections + compute PnP pose
        self.publish_apriltags(detected_tags)

        self.get_logger().info(f"Detected {len(detected_tags)} tags in this frame.")
        self.get_logger().info(f"Detected {self.total_num_tags} apriltags in total.")

    # Build AprilTagDetectionArray and run solvePnP per tag
    def publish_apriltags(self, detected_tags):
        detection_array = AprilTagDetectionArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera"  # camera optical frame

        for tag in detected_tags:
            self.total_num_tags += 1

            det = AprilTagDetection()
            det.family = self.apriltag_family
            det.id = int(tag["id"])
            det.hamming = int(tag["hamming"])
            det.decision_margin = float(tag["margin"])  # keep margin here

            # centre & corners: scale back from upscaled coords
            cx_scaled = float(tag["center"][0] / self.scaling_factor)
            cy_scaled = float(tag["center"][1] / self.scaling_factor)
            self.get_logger().info(
                f"Center of tag {det.id}: ({cx_scaled}, {cy_scaled})"
            )
            det.centre = Point(x=cx_scaled, y=cy_scaled)

            corners_arr = np.array(tag["lb-rb-rt-lt"]).reshape(4, 2)
            corners_scaled = corners_arr / float(self.scaling_factor)

            det.corners = [Point(x=float(x), y=float(y)) for x, y in corners_scaled]

            # homography not used here; leave as zeros
            det.homography = [0.0] * 9

            # Store distance in 'goodness' so the visualizer can read it.
            # (decision_margin still holds the AprilTag margin.)
            det.goodness = float(
                self.calculate_distance(corners_scaled, (cx_scaled, cy_scaled), det.id)
            )

            detection_array.detections.append(det)

        self.detections_publisher.publish(detection_array)

    def calculate_distance(self, corners, center, id):
        # PnP pose estimation
        distance = -1.0  # default if we can't compute it
        # 3D model of tag corners in tag frame (lb, rb, rt, lt)
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

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.new_camera_matrix,  # intrinsics of the calibrated image
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                # flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if success:
                # orientation calculation
                print("RVEC", rvec)
                obj_space_R, _ = cv2.Rodrigues(rvec)
                print("ONJ_SPACE_R", obj_space_R)
                inv_obj_space_R = np.linalg.inv(obj_space_R)
                print("INV_ONJ_SPACE_R", inv_obj_space_R)
                obj_space_optic_axis = inv_obj_space_R @ np.array([0, 0, 1]).T
                print("OBJ_SPACE_OPTIC_AXIS", obj_space_optic_axis)
                obj_space_optic_axis = obj_space_optic_axis / np.linalg.norm(
                    obj_space_optic_axis
                )
                print("OBJ_SPACE_OPTIC_AXIS NORMALIZED", obj_space_optic_axis)
                angle_to_optic_axis = np.arccos(obj_space_optic_axis[-1])
                print("ANGLE TO OPTIC AXIS", angle_to_optic_axis)
                robot_orientation_wrt_apriltag = (
                    angle_to_optic_axis + self.camera_pan_angle
                )
                print("ROBOT ORIENTATION WRT APRILTAG", robot_orientation_wrt_apriltag)

                inv_intrinsic_matrix = np.linalg.inv(self.new_camera_matrix)
                print("inv intrinsic matr", inv_intrinsic_matrix)
                ray_camera_apriltag_center = inv_obj_space_R @ inv_intrinsic_matrix @ np.array([center[0], center[1], 1]).T
                print("ray from camera to apriltag center", ray_camera_apriltag_center)
                a = ray_camera_apriltag_center.copy()
                a[2] = 0.0
                print("A", a)
                b = obj_space_optic_axis
                b[2] = 0.0
                print("B", b)
                cross = np.cross(a, b)
                print("cross", cross)

                if cross[2] < 0:
                    angle_to_optic_axis = - angle_to_optic_axis
                # when the angle is positive the viewing axis is to the right of the apriltag
                # when the angle is negative the viewing axis is to the left of the apriltag
                print("final angle", angle_to_optic_axis)

                robot_orientation = angle_to_optic_axis + self.tag_orientation[id]
                print("robot orientation", robot_orientation)

                if self.camera_pan_angle:
                    print("camera pan", self.camera_pan_angle)
                    robot_orientation = robot_orientation + self.camera_pan_angle
                    print("corrected robot orientation", robot_orientation)
                else:
                    self.get_logger().warn("Camera pan angle is not available. Assuming it is 0.0 rads...")

                # distance calculation
                tvec = tvec.reshape(3)
                distance = float(np.sqrt(tvec[0] ** 2 + tvec[2] ** 2))
                self.get_logger().info(
                    f"Tag {id}: "
                    f"t = ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}) m, "
                    f"distance ≈ {distance:.3f} m"
                )
            else:
                self.get_logger().warn(f"PnP pose estimation failed for tag {id}")
        except cv2.error as e:
            self.get_logger().warn(f"solvePnP failed for tag {id}: {e}")

        return distance


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
