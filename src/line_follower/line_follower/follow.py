import math
import time
from functools import partial

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState

from driver.driver_publisher import DrivePublisher
from image_subscriber.image_subscriber import ImageSubscriber
from .hough_tools import image_center, get_edges, polar_lines, filter_lines

LINEAR_VELOCITY = 0.2
DURATION_LINEAR_MOVE = 0.5  # seconds

ANGULAR_VELOCITY = math.pi / 4  # radians per second

# Choose "canny" (baseline) or "skeleton" (new algorithm) once here:
LINE_METHOD = "canny"   # change to "skeleton" for your new detector


#line computation

def compute_lines(rgb_image, method: str = "canny"):
    """
    Compute Hough lines (rho, theta) from an RGB/BGR image using the
    chosen preprocessing method.
    """
    # ImageSubscriber usually gives RGB, but we treat it as BGR here
    # because original code used COLOR_BGR2GRAY.
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    origin = image_center(gray)
    edges = get_edges(gray, method=method)

    r_theta = polar_lines(edges, origin)
    filt_r_theta = filter_lines(r_theta)

    # convert to simple python list of tuples
    return [(float(r), float(t)) for r, t in filt_r_theta]


def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotation_parallel_from_hough(theta):
    """
    Convert Hough line normal angle theta into the robot rotation
    required to align parallel with the line.
    """
    r = wrap_to_pi(theta)
    r_opposite = wrap_to_pi(r + math.pi)
    return r if abs(r) <= abs(r_opposite) else r_opposite


def follow_line(drive_publisher: DrivePublisher, rgb_image: np.ndarray):
    """
    Called for each image captured from the camera.

    We:
    1. Detect ceiling lines.
    2. Pick the closest one (smallest rho).
    3. Rotate to be parallel with it.
    4. Move forward a bit.
    """
    hough_lines = compute_lines(rgb_image, method=LINE_METHOD)

    if not hough_lines:
        # No lines detected -> be conservative: do nothing this frame
        print("No lines detected, skipping motion update.")
        return

    closest_line = min(hough_lines, key=lambda x: x[0])
    rho, theta = closest_line

    # face direction of line
    turn_angle = rotation_parallel_from_hough(theta)
    drive_publisher.turn(turn_angle, ANGULAR_VELOCITY)

    # move along line
    drive_publisher.move_forward(DURATION_LINEAR_MOVE, LINEAR_VELOCITY)



def set_camera_joint_once():
    """
    Publish a one-time joint command to rotate camera 90 degrees up.
    """
    node = rclpy.create_node("joint_publisher_once")
    publisher = node.create_publisher(JointState, "/ugv/joint_states", 10)

    joint_msg = JointState()
    joint_msg.header.stamp = node.get_clock().now().to_msg()

    joint_msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
    joint_msg.position = [0.0, math.pi / 2]

    publisher.publish(joint_msg)
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    print("Setting camera orientation...")
    set_camera_joint_once()

    try:
        print("Initializing the image subscriber...")
        image_subscriber = ImageSubscriber(partial(follow_line, drive_publisher))
        rclpy.spin(image_subscriber)

    except KeyboardInterrupt:
        print("Shutting down...")

        if rclpy.ok():
            stop_msg = Twist()
            for _ in range(5):
                drive_publisher.publisher.publish(stop_msg)
            time.sleep(0.3)

    finally:
        try:
            drive_publisher.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
