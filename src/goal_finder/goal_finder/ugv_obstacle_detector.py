#!/usr/bin/env python3
"""
Detect other UGV rovers using OAK-D RGB + depth.

Method: color (dark blobs) + depth (distance threshold)
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

#for publishing obstacle info
from geometry_msgs.msg import PointStamped
from rclpy.qos import qos_profile_sensor_data

class UGVObstacleDetector(Node):
    def __init__(self):
        super().__init__("ugv_obstacle_detector")

        # Parameters
        # self.rgb_topic = "/oak/rgb/image_raw"
        self.rgb_topic = "/color/image"  # it might be lower resolution 
        self.depth_topic = "/stereo/depth/compressedDepth"
        self.max_obstacle_distance_m = 1.5  # threshold for obstacle distance (idk, I put 1.5 m, we can change later)
        self.min_blob_area_px = 800        # ignore tiny noise blobs < 800 px
        self.show_debug_window = False      # set False if no GUI available

        self.bridge = CvBridge()

        # Store the latest depth frame; we process when a new RGB image arrives
        self.latest_depth = None

        # Subscribers
        self.rgb_subscriber = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, 10, qos_profile_sensor_data,
        )
        # depth is usually published as CompressedImage on *compressedDepth topics
        self.depth_subscriber = self.create_subscription(
            CompressedImage, self.depth_topic, self.depth_callback, 10, qos_profile_sensor_data,
        )
        
        # Publisher: centroid + distance
        #   point.x = centroid x (pixels)
        #   point.y = centroid y (pixels)
        #   point.z = distance (meters)
        #   header.frame_id = "camera"

        self.obstacle_publisher = self.create_publisher(PointStamped, "/obstacle_detected", 10)

        self.get_logger().info(
            f"UGVObstacleDetector started. RGB: {self.rgb_topic}, "
            f"depth: {self.depth_topic}"
        )        

    # Callbacks

    def depth_callback(self, msg: CompressedImage):
        """
        Store the latest depth image.

        We don't process it here; we just keep it so that when a new RGB
        image arrives, we have a reasonably recent depth frame to pair with it
        """
        # depth encoding is "passthrough": typically 16UC1, depth in millimetres
        depth_img = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        ) # it's in mm
        self.latest_depth = depth_img

    def rgb_callback(self, msg: Image):
        """
        Process a new RGB frame.

        Steps:
        1. Convert to OpenCV BGR image.
        2. Check if we have a depth image; if not, just return.
        3. Run color+depth detection.
        """
        if self.latest_depth is None:
            # No depth yet – cannot compute distances
            return

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        depth = self.latest_depth

        # Ensure depth and RGB have the same size; if not, resize depth --> [BASICALLY CHECK THE RESOLUTION]
        if depth.shape[:2] != bgr.shape[:2]:
            depth = cv2.resize(
                depth,
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) #With INTER_NEAREST we don’t interpolate between values; we just copy the nearest pixel

        obstacle_info, debug_img = self.detect_obstacles(bgr, depth)

        msg_out = PointStamped()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = "camera"

        if obstacle_info is not None:
            cx, cy, dist_m = obstacle_info

            # You can still log detection
            self.get_logger().info(
                f"Detected rover at distance {dist_m:.2f} m, "
                f"image position x={cx}, y={cy}"
            )

            msg_out.point.x = float(cx)
            msg_out.point.y = float(cy)
            msg_out.point.z = float(dist_m)

        else:
            # Publish -1 if there's no obstacle
            msg_out.point.x = 0.0
            msg_out.point.y = 0.0
            msg_out.point.z = -1.0

            # self.get_logger().debug("No rover detected.")

        self.obstacle_publisher.publish(msg_out)

        # Optional debug visualization
        if self.show_debug_window and debug_img is not None:
            cv2.imshow("UGV obstacle detection (RGB)", debug_img)
            cv2.waitKey(1)


    # Core detection logic

    def detect_obstacles(self, bgr_img: np.ndarray, depth_img: np.ndarray):
        """
        Detect dark rovers using color + depth.

        Returns:
            bounding_box_obstacle: (cx, cy, distance_m) or None
            debug_img: image with drawings (for visualization)
        """
        h, w, _ = bgr_img.shape #height h, width w, and 3 color channels
        debug_img = bgr_img.copy()

        # Work only on the lower half of the image (field region becuase rovers don't fly ;))
        roi_y_start = h // 2    #midpoint of the image height 
        region_of_interest = bgr_img[roi_y_start:h, :]  # y means from halfway down to the bottom and : is all x values (full width)

        # Convert ROI to HSV for robust brightness-based thresholding.
        hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)

        # Threshold for "dark" pixels
        #
        #  H range: 0..180 (don't care -> full range)
        #  S range: 0..255 (allow all saturation)
        #  V range: 0..60  (low brightness -> dark)
        
        #    This should catch the black rover body (darkest object)
        lower_dark = (0, 0, 0)
        upper_dark = (180, 255, 60)
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        # Morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Find contours (connected dark regions)
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bounding_box_obstacle = None
        closest_robot = float("inf")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_blob_area_px:
                # Too small to be a rover; ignore camera noise / shadows
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)

            # Bounding box coordinates are in ROI coords;
            # shift them back to full-image coords by adding roi_y_start.
            x0 = x
            y0 = y + roi_y_start
            x1 = x0 + w_box
            y1 = y0 + h_box

            # Draw the candidate region on debug image.
            cv2.rectangle(
                debug_img, (x0, y0), (x1, y1), (0, 255, 255), 2
            )

            # Compute center of the bounding box (in full image coords).
            cx = x0 + w_box // 2
            cy = y0 + h_box // 2
            cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)

            # Estimate distance using depth image.
            #
            #    Depth image is assumed to be 16-bit, depth in millimetres.
            #    We'll take a small central patch to avoid noisy edges.
            patch_size = 10
            px0 = max(cx - patch_size, 0)
            px1 = min(cx + patch_size, depth_img.shape[1])
            py0 = max(cy - patch_size, 0)
            py1 = min(cy + patch_size, depth_img.shape[0])

            depth_patch = depth_img[py0:py1, px0:px1].astype(np.float32)

            # Remove invalid depths: 0 or very large
            valid = depth_patch[(depth_patch > 0) & (depth_patch < 10000)]
            if valid.size == 0:
                continue

            # Median depth in mm -> convert to meters
            dist_m = float(np.median(valid) / 1000.0)

            # Label distance on debug image
            cv2.putText(
                debug_img,
                f"{dist_m:.2f} m",
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Check if this rover is close enough to matter
            if dist_m < self.max_obstacle_distance_m and dist_m < closest_robot:
                closest_robot = dist_m
                bounding_box_obstacle = (cx, cy, dist_m)

        # Draw a vertical line showing the "front corridor"
        img_h, img_w, _ = debug_img.shape
        center_x = img_w // 2
        corridor_half_width = img_w // 6  # central third of the image
        cv2.line(
            debug_img,
            (center_x - corridor_half_width, roi_y_start),
            (center_x - corridor_half_width, img_h),
            (255, 0, 0),
            1,
        )
        cv2.line(
            debug_img,
            (center_x + corridor_half_width, roi_y_start),
            (center_x + corridor_half_width, img_h),
            (255, 0, 0),
            1,
        )

        return bounding_box_obstacle, debug_img


def main(args=None):
    rclpy.init(args=args)
    node = UGVObstacleDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if node.show_debug_window:
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()