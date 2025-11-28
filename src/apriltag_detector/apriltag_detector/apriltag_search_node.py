from math import pi as PI

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


def to_rads(degrees):
    return degrees / 180 * PI


class AprilTagSearchNode(Node):
    def __init__(self):
        super().__init__("apriltag_search_node")
        self.STEP_SIZE = to_rads(10)
        self.ANGLE_LIMIT = to_rads(80)
        self.TOLERANCE = 7

        self.direction = 1
        self.empty_count = 0
        self.current_angle = 0.0

        self.detections_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag/detections", self.detections_callback, 10
        )

        self.joint_states_publisher = self.create_publisher(
            JointState, "/ugv/joint_states", 10
        )

        self.reset_camera_tilt()

        self.get_logger().info("AprilTag Search Node Initialized.")

        self.pan_publisher = self.create_publisher(Float32, "/camera_pan", 10)
        self.timer = self.create_timer(0.5, self.publish_pan)

    def publish_pan(self):
        msg = Float32()
        msg.data = self.current_angle
        self.pan_publisher.publish(msg)

    def detections_callback(self, msg: AprilTagDetectionArray):
        if len(msg.detections) > 1:
            self.empty_count = 0
        else:
            self.empty_count += 1

        if self.empty_count >= self.TOLERANCE:
            # self.get_logger().info(
            #     f"No apriltags detected in the past {self.TOLERANCE} messages. Tilting the camera..."
            # )
            self.camera_step()
            self.empty_count = 0

    def camera_step(self):
        if abs(self.current_angle) >= self.ANGLE_LIMIT:
            self.direction = self.direction * (-1)

        self.current_angle = self.current_angle + self.direction * self.STEP_SIZE
        # self.get_logger().info(
        #     f"Moving the camera to position: {int(self.current_angle / PI * 180)} degrees."
        # )

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()

        joint_msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        joint_msg.position = [self.current_angle, 0.0]
        self.joint_states_publisher.publish(joint_msg)

    def reset_camera_tilt(self):
        self.get_logger().info("Resetting camera position to [0, 0].")
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()

        joint_msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        joint_msg.position = [0.0, 0.0]
        self.joint_states_publisher.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagSearchNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
