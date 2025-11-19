from math import pi as PI

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.node import Node
from sensor_msgs.msg import JointState


def to_rads(degrees):
    return degrees / 180 * PI


class AprilTagSearchNode(Node):
    def __init__(self):
        super().__init__("apriltag_search_node")
        self.STEP_SIZE = to_rads(10)
        self.ANGLE_LIMIT = to_rads(70)
        self.TOLERANCE = 10

        self.direction = 1
        self.empty_count = 0
        self.current_angle = 0

        self.detections_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag/detections", self.detections_callback, 10
        )

        self.camera_tilt_publisher = self.create_publisher(
            JointState, "/ugv/joint_states", 10
        )

        self.reset_camera_tilt()

    def detections_callback(self, msg: AprilTagDetectionArray):
        if len(msg.detections) > 1:
            self.empty_count = 0
        else:
            self.empty_count += 1

        if self.empty_count >= self.TOLERANCE:
            self.camera_step()
            self.empty_count = 0

    def camera_step(self):
        if self.current_angle >= self.ANGLE_LIMIT:
            self.direction = self.direction * (-1)

        self.current_angle = self.current_angle + self.direction * self.STEP_SIZE

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()

        joint_msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        joint_msg.position = [0.0, self.current_angle]
        self.camera_tilt_publisher.publish(joint_msg)

    def reset_camera_tilt(self):
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()

        joint_msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        joint_msg.position = [0.0, 0.0]
        self.camera_tilt_publisher.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagSearchNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
