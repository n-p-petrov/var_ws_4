import math
import time
from functools import partial

import cv2
import numpy as np
import rclpy
from driver.driver_publisher import DrivePublisher
from geometry_msgs.msg import Twist
from image_subscriber.image_subscriber import ImageSubscriber

from .hough_tools import *

LINEAR_VELOCITY = 0.2
DURATION_LINEAR_MOVE = 10  # seconds

ANGULAR_VELOCITY = math.pi / 4  # radians per second


def compute_lines(rgb_image):
    image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    origin = image_center(image)
    edges = image_preprocess(image)
    r_theta = polar_lines(edges, origin)
    filt_r_theta = filter_lines(r_theta)
    return [(float(r), float(t)) for r, t in filt_r_theta]


def follow_line(drive_publisher, rgb_image):
    """
    This function is executed for each image captured from the camera.

    The idea is to perform a small corrections to the direction of the robot
    on every frame of the ceiling such that it doesn't deviate too much from
    the line it is following.

    After each correction we move the robot forward for a bit and then repeat
    with a new image.
    """
    hough_lines = compute_lines(rgb_image)
    closest_line = min(hough_lines, key=lambda x: x[0])

    # face direction of line
    rho, theta = closest_line[0], closest_line[1]
    drive_publisher.turn(math.pi / 2 - theta, ANGULAR_VELOCITY)

    # move along line
    drive_publisher.move_forward(DURATION_LINEAR_MOVE, LINEAR_VELOCITY)


def main(args=None):
    rclpy.init(args=args)
    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    try:
        print("Initializing the image subscriber...")
        image_subscriber = ImageSubscriber(partial(follow_line, drive_publisher))
        rclpy.spin(image_subscriber)

    except KeyboardInterrupt:
        print("Shutting down...")

        if rclpy.ok():
            stop_msg = Twist()
            for i in range(5):
                drive_publisher.publisher.publish(stop_msg)

            time.sleep(0.3)

    finally:
        try:
            drive_publisher.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()
