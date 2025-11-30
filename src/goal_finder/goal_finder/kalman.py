import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose2D, Twist
from rclpy.node import Node
from std_msgs.msg import Float32


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class EkfNode(Node):
    """
    EKF with state:
        x = [x, y, theta_robot]^T

    - Prediction: uses /cmd_vel (v, omega) for robot motion
    - Update:
        * (x, y) from triangulator
        * theta_robot from /orientation
    """

    def __init__(self):
        super().__init__("ekf_node")

        # --- State and covariance ---
        self.x = np.zeros((3, 1))  # [x, y, theta]
        self.P = np.eye(3) * 1.0  # initial covariance

        # Process noise (tune these!)
        # [x, y, theta]
        self.Q = np.diag([0.01, 0.01, 0.05])

        # Measurement noise
        self.R_xy = np.diag([0.02, 0.02])  # for (x, y) from triangulator

        # Control inputs from cmd_vel
        self.v = 0.0  # linear velocity
        self.omega = 0.0  # angular velocity

        # For dt computation
        self.last_time = self.get_clock().now()

        # --- Subscribers ---
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )

        self.triangulated_pos_sub = self.create_subscription(
            Point, "/triangulated_pos", self.triangulated_callback, 10
        )

        # orientation of robot (already accounts for camera angle / phi)
        self.orientation_sub = self.create_subscription(
            Float32, "/orientation", self.orientation_callback, 10
        )

        # --- Publisher ---
        self.filtered_pose_pub = self.create_publisher(Pose2D, "/filtered_pose", 10)

        # Timer for continuous prediction
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz

        self.get_logger().info("EKF node (3D: x,y,theta) initialized")

    # ------------- Callbacks -------------

    def cmd_vel_callback(self, msg: Twist):
        # Make sure units match your triangulated positions (m vs mm!)
        self.v = msg.linear.x * 1000.0  # or remove *1000 if everything is in meters
        self.omega = msg.angular.z

    def triangulated_callback(self, msg: Point):
        """
        (x, y) measurement from triangulator.
        """
        # Triangulator publishes (-1, -1) when it has no valid fix; ignore those.
        if msg.x < 0.0 or msg.y < 0.0:
            return

        z = np.array([[msg.x], [msg.y]])

        # H maps state [x, y, theta] -> [x, y]
        H = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        def h(x_vec):
            return x_vec[0:2, :]  # [x, y]

        self.ekf_update(z, H, self.R_xy, h)

    def orientation_callback(self, msg: Float32):
        """
        Direct measurement of theta_robot (e.g., from IMU).
        """
        theta_meas = wrap_angle(msg.data)
        z = np.array([[theta_meas]])

        # H maps [x, y, theta] -> [theta]
        H = np.array([[0.0, 0.0, 1.0]])

        R_theta = np.array([[0.001]])  # measurement noise for theta (tune this!)

        def h(x_vec):
            return np.array([[x_vec[2, 0]]])  # theta

        self.ekf_update(z, H, R_theta, h)

    def timer_callback(self):
        """
        Continuous prediction step + publishing.
        """
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0.0:
            return

        # Clamp dt to avoid huge jumps if the timer hiccups
        if dt > 0.1:
            dt = 0.1

        self.ekf_predict(self.v, self.omega, dt)
        self.publish_state()

    # ------------- EKF Core -------------

    def ekf_predict(self, v: float, omega: float, dt: float):
        """
        State: [x, y, theta]^T

        x_{k+1}     = x_k + v dt cos(theta)
        y_{k+1}     = y_k + v dt sin(theta)
        theta_{k+1} = theta_k + omega dt
        """
        x_val, y_val, th = self.x.flatten()

        x_new = x_val + v * dt * math.cos(th)
        y_new = y_val + v * dt * math.sin(th)
        th_new = wrap_angle(th + omega * dt)

        self.x = np.array([[x_new], [y_new], [th_new]])

        # Jacobian F of f(x, u)
        F = np.array(
            [
                [1.0, 0.0, -v * dt * math.sin(th)],
                [0.0, 1.0, v * dt * math.cos(th)],
                [0.0, 0.0, 1.0],
            ]
        )

        self.P = F @ self.P @ F.T + self.Q

    def ekf_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, h_func):
        """
        Generic EKF update step.
        For 1D measurements (theta), we wrap the innovation as an angle.
        """
        z_hat = h_func(self.x)  # expected measurement
        y = z - z_hat  # innovation

        # If this is a 1x1 measurement (like theta), treat it as an angle and wrap
        if y.shape == (1, 1):
            y[0, 0] = wrap_angle(y[0, 0])

        S = H @ self.P @ H.T + R  # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y

        # Wrap theta in the state
        self.x[2, 0] = wrap_angle(self.x[2, 0])

        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def publish_state(self):
        """
        Publish [x, y, theta] as Pose2D.
        """
        pose_msg = Pose2D()
        pose_msg.x = float(self.x[0, 0])
        pose_msg.y = float(self.x[1, 0])
        pose_msg.theta = float(self.x[2, 0])

        # If your visualizer needs Pose2D with theta, this is what it should subscribe to
        self.filtered_pose_pub.publish(pose_msg)

        # self.get_logger().info(
        #     f"x={pose_msg.x:.2f}, y={pose_msg.y:.2f}, theta={pose_msg.theta:.2f}"
        # )


def main(args=None):
    rclpy.init(args=args)
    ekf_node = EkfNode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()
