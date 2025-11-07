import math
from functools import partial

import rclpy
from driver.driver_publisher import DrivePublisher
from image_subscriber.image_subscriber import ImageSubscriber

LINEAR_VELOCITY = 2 * math.pi  # radians per second (2pi = 1 wheel rotation per second)
DURATION_LINEAR_MOVE = 3  # seconds

ANGULAR_VELOCITY = math.pi / 4  # radians per second


def compute_lines(rgb_image):
    # TODO: add the actual implementation here lol
    return [(1, math.pi / 3), (2, 0.0)]


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
    drive_publisher.turn(theta, ANGULAR_VELOCITY)

    # move along line
    drive_publisher.move_forward(DURATION_LINEAR_MOVE, LINEAR_VELOCITY)


def main(args=None):
    rclpy.init(args=args)

    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    print("Initializing the image subscriber...")
    image_subscriber = ImageSubscriber(partial(follow_line, drive_publisher))
    rclpy.spin(camera_image)

    print("Line following finished")

    rclpy.shutdown()
