import math
from functools import partial

import rclpy
from camera_subscriber.camera_subscriber import CameraSubscriber

from driver.driver import DrivePublisher


def compute_turn_angle(theta):
    return math.pi / 2 - theta

def compute_lines(rgb_image):
    # TODO: add the actual implementation here lol
    return [(1, math.pi / 3), (2, 0.0)]

def follow_line(drive_publisher, rgb_image):
    hough_lines = hough_lines(rgb_image)
    closest_line = min(hough_lines, key=0)

    drive_publisher.turn(compute_turn_angle(closest_line.1), 0.25)
    drive_publisher.move_forward(2, math.pi)

def main(args=None):
    rclpy.init(args=args)

    drive_publisher = DrivePublisher()
    camera_subscriber = CamaraSubscriber(partial(follow_line, drive_publisher))

    rclpy.shutdown()
