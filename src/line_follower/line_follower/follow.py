import math
from functools import partial

import rclpy
from image_subscriber.image_subscriber import ImageSubscriber

from driver.driver_publisher import DrivePublisher


def compute_turn_angle(theta):
    return math.pi / 2 - theta

def compute_lines(rgb_image):
    # TODO: add the actual implementation here lol
    return [(1, math.pi / 3), (2, 0.0)]

def follow_line(drive_publisher, rgb_image):
    hough_lines = compute_lines(rgb_image)
    closest_line = min(hough_lines, key=lambda x: x[0])

    drive_publisher.turn(compute_turn_angle(closest_line[1]), 0.25)
    drive_publisher.move_forward(2, math.pi)

def main(args=None):
    rclpy.init(args=args)

    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    print("Initializing the image subscriber...") 
    camera_subscriber = ImageSubscriber(partial(follow_line, drive_publisher))
    rclpy.spin(camera_subscriber)

    print("Line following finished")

    rclpy.shutdown()
