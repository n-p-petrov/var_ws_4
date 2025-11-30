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
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 1)

    def turn(self, radians, angular_speed):
        """
        Tells the robot to turn by some angle.

        positive angle -> rotate right
        negative angle -> rotate left

        Args:
            angle (float): Angle to turn by in radians.
            angular_speed (float): radians per second
        """
        msg = Twist()
        duration = abs(radians) / angular_speed
        msg.angular.z = -angular_speed if radians > 0 else angular_speed

        for _ in range(5):
            self.publisher.publish(msg)
        time.sleep(duration)

        for _ in range(5):
            self.publisher.publish(Twist())

    def move_forward(self, duration, speed):
        """
        Tells the robot to move forward for some duration and some speed.

        Args:
            duration (float): for how much time to move forward in seconds
            speed (float): speed in radians per sec (1 rotation of the wheels/sec = 2pi radians/sec)
        """
        msg = Twist()
        msg.linear.x = speed

        for _ in range(5):
            self.publisher.publish(msg)
        time.sleep(duration)

        for _ in range(5):
            self.publisher.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    drive_publisher = DrivePublisher()
    drive_publisher.turn(math.pi, 1.0)
    drive_publisher.move_forward(2, 2 * math.pi)
    rclpy.shutdown()
