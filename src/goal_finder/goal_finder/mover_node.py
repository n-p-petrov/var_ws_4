import time
from math import pi

import numpy as np
from driver.driver_publisher import DrivePublisher
from geometry_msgs.msg import Pose2D
from rclpy.node import Node


class MoverNode(Node):
    def __init__(self, drive_publisher: DrivePublisher):
        super().__init__("mover_node")
        self.LINEAR_VELOCITY = 0.2  # something per second
        self.DURATION_LINEAR_MOVE = 1.0  # seconds
        self.ANGULAR_VELOCITY = pi / 4  # radians per second
        self.STOPPING_MAGNITUDE = 500
        self.busy = False

        self.drive_publisher = drive_publisher

        self.gradient_subscriber = self.create_subscription(
            Pose2D,
            "/grad/gradient",
            self.gradient_callback,
            10,
        )
        self.latest_gradient_pose = None

    def gradient_callback(self, msg: Pose2D):
        self.latest_gradient_pose = msg
        self.move_along_gradient()

    def _wrap_to_pi(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def move_along_gradient(self):
        if self.busy:
            return

        if not self.latest_gradient_pose:
            return

        self.busy = True
        try:
            grad_magnitude = np.linalg.norm(
                [self.latest_gradient_pose.x, self.latest_gradient_pose.y]
            )
            if grad_magnitude > self.STOPPING_MAGNITUDE:
                turn_angle = self.latest_gradient_pose.theta
                if abs(turn_angle) > 0.05:
                    self.get_logger().info(
                        f"Turning by {int(turn_angle / pi * 180)} degrees."
                    )
                    self.drive_publisher.turn(self._wrap_to_pi(turn_angle), self.ANGULAR_VELOCITY)

                self.drive_publisher.move_forward(
                    self.DURATION_LINEAR_MOVE, self.LINEAR_VELOCITY
                )
                time.sleep(2)  # attempt to detect apriltags while stationary
        finally:
            self.busy = False
