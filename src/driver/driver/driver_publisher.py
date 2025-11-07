import math
import time
from functools import partial

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class DrivePublisher(Node):
    def __init__(self):
        super().__init__("drive_publisher")
        # publisher of twist (movement) messages
        # if more than 10 msgs are not consumed it replaces the oldest one
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def move(self, lin_vel, ang_vel, duration):
        """
        Tells the robot to move forward or/and by some angle for some amount of time.

        Args:
            lin_vel (float): linear velocity; if negative robot moves backward, otherwise moves forward;
            ang_vel (float): angular velocity; the sign determines the direction of rotation;
            duration (float): for how long to perform the movement in seconds;
        """
        start_time = time.time()
        self.timer = self.create_timer(
            0.05, partial(self._publish_twist, lin_vel, ang_vel, duration, start_time)
        )
        self.get_logger().info(
            f"Started moving for {duration:.2f}s with linear velocity of {lin_vel:.2f} and angular velocity of {ang_vel:.2f}."
        )

    def _publish_twist(self, lin_vel, ang_vel, duration, start_time):
        elapsed = time.time() - start_time

#        if elapsed < duration:
        msg = Twist()
        msg.linear.x = lin_vel
        msg.angular.z = ang_vel
        self.publisher.publish(msg)
#        else:
#            # stop the robot
#            self.publisher.publish(Twist())
#            self.timer.cancel()
#            self.get_logger().info("Movement completed.")
