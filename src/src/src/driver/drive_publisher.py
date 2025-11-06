import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class DrivePublisher(Node):
    def __init__(self):
        super().__init__("drive_publisher")
        # publisher of twist (movement) messages
        # if more than 10 msgs are not consumed it replaces the oldest one
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def turn(self, radians, angular_speed):
        """
        Tells the robot to turn by some angle.

        Args:
            angle (float): Angle to turn by in radians.
                If angle < 0, turns left.
                If angle > 0, turns right.
            angular_speed (float): radians per second
        """
        msg = Twist()
        duration = abs(radians) / angular_speed
        msg.angular.z = angular_speed if radians > 0 else -angular_speed

        self.get_logger().info(f"Turning {radians} radians for {duration:.2f}s")

        start = time.time()
        while time.time() - start < duration:
            self.publisher.publish(msg)
            self.publisher.sleep(
                0.05
            )  # wait 5 ms before asking the robot to turn again

        self.publisher.publish(Twist())
        self.get_logger().info("Turn completed.")


def main(args=None):
    rclpy.init(args=args)
    drive_publisher = DrivePublisher()
    turn(math.pi, 1)
    rclpy.shutdown()
