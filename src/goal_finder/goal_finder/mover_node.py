from math import pi

import numpy as np
from driver.driver_publisher import DrivePublisher
from geometry_msgs.msg import Pose2D
from rclpy.node import Node


class MoverNode(Node):
    def __init__(self, drive_publisher: DrivePublisher):
        super().__init__("mover_node")
        self.LINEAR_VELOCITY = 0.2  # something per second
        self.DURATION_LINEAR_MOVE = 0.5  # seconds
        self.ANGULAR_VELOCITY = pi / 4  # radians per second
        self.STOPPING_MAGNITUDE = 500

        self.drive_publisher = drive_publisher

        self.pose_subscriber = self.create_subscription(
            Pose2D, "/filtered_pose", self.pose_callback, 10
        )
        self.latest_robot_pose = None

        self.gradient_subscriber = self.create_subscription(
            Pose2D,
            "/grad/gradient",
            self.gradient_callback,
            10,
        )
        self.latest_gradient_pose = None

        self.drive_timer = self.create_timer(0.5, self.move_along_gradient)

    def pose_callback(self, msg: Pose2D):
        self.latest_robot_pose = msg

    def gradient_callback(self, msg: Pose2D):
        self.latest_gradient_pose = msg

    def wrap_angle(self, a: float) -> float:
        return (a + pi) % (2.0 * pi) - pi

    def move_along_gradient(self):
        if self.latest_gradient_pose and self.latest_robot_pose:
            grad_magnitude = np.linalg.norm(
                [self.latest_gradient_pose.x, self.latest_gradient_pose.y]
            )
            if grad_magnitude > self.STOPPING_MAGNITUDE:
                turn_angle = self.wrap_angle(
                    self.latest_gradient_pose.theta - self.latest_robot_pose.theta
                )
                if abs(turn_angle) > 0.05:
                    self.get_logger().info(
                        f"Turning by {int(turn_angle / pi * 180)} degrees."
                    )
                    self.drive_publisher.turn(turn_angle, self.ANGULAR_VELOCITY)

                self.drive_publisher.move_forward(
                    self.DURATION_LINEAR_MOVE, self.LINEAR_VELOCITY
                )
