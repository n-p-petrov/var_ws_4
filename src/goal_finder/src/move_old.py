import math
import time
from functools import partial

import cv2
import numpy as np
import rclpy
from driver.driver_publisher import DrivePublisher
from geometry_msgs.msg import Twist
from image_subscriber.image_subscriber import ImageSubscriber
from triangulator.triangulator import Triangulator
from apriltag_detector.apriltag_detector import ApriltagDetector
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .hough_tools import *

LINEAR_VELOCITY = 0.2
DURATION_LINEAR_MOVE = 0.5  # seconds

ANGULAR_VELOCITY = math.pi / 4  # radians per second



class RobotState:
    x: float = 0.0       # robot position x
    y: float = 0.0       # robot position y
    theta: float = 0.0   # robot heading (rad)
    phi_cam: float = 0.0 # camera pan angle (rad)

    def as_vector(self):
        """4x1 column vector for EKF math."""
        return np.array([[self.x],
                         [self.y],
                         [self.theta],
                         [self.phi_cam]], dtype=float)

    @staticmethod
    def from_vector(vec: np.ndarray) -> "RobotState":
        """Convert 4x1 vector back to nice fields."""
        return RobotState(
            x=float(vec[0, 0]),
            y=float(vec[1, 0]),
            theta=float(vec[2, 0]),
            phi_cam=float(vec[3, 0]),
        )

def wrap_angle(a: float) -> float:
    """Wrap any angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
        
class SimpleEKF:
    def __init__(self):
        # internal state vector x (4x1) and covariance P (4x4)
        self.state = RobotState()
        self.x = self.state.as_vector()
        self.P = np.eye(4) * 1.0  # fairly uncertain at start

        # process noise (Q) – tune these numbers
        self.Q = np.diag([0.01, 0.01, 0.001, 0.001])

    # ---------------- PREDICTION STEP ----------------
    def predict(self, v: float, omega: float, phi_dot: float, dt: float):
        """
        v      : forward velocity (m/s)
        omega  : robot yaw rate (rad/s)
        phi_dot: camera pan rate (rad/s)
        dt     : time step (s)
        """
        xR, yR, th, phi = self.x.flatten()

        # --- nonlinear motion model ---
        xR_new = xR + v * dt * math.cos(th)
        yR_new = yR + v * dt * math.sin(th)
        th_new = wrap_angle(th + omega * dt)
        phi_new = wrap_angle(phi + phi_dot * dt)

        self.x = np.array([[xR_new],
                           [yR_new],
                           [th_new],
                           [phi_new]])

        # --- Jacobian F (4x4) of f(x,u) ---
        F = np.array([
            [1.0, 0.0, -v * dt * math.sin(th), 0.0],
            [0.0, 1.0,  v * dt * math.cos(th), 0.0],
            [0.0, 0.0, 1.0,                   0.0],
            [0.0, 0.0, 0.0,                   1.0],
        ])

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    # ---------------- COMMON UPDATE CORE ----------------
    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, h_func):
        """
        Internal EKF update step.
        z: measurement vector (n x 1)
        H: measurement Jacobian (n x 4)
        R: measurement noise (n x n)
        h_func: function(x) -> expected measurement (n x 1)
        """
        z_hat = h_func(self.x)  # expected measurement
        y = z - z_hat           # innovation

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        # keep angles normalized
        self.x[2, 0] = wrap_angle(self.x[2, 0])  # theta
        self.x[3, 0] = wrap_angle(self.x[3, 0])  # phi_cam

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    # ---------------- UPDATE: APRILTAG POSE ----------------
    def update_pose_from_tags(self, x_meas: float, y_meas: float, theta_meas: float | None = None):
        """
        Update using AprilTag-based robot pose (world frame).
        If theta_meas is None, only x,y are used.
        """
        if theta_meas is None:
            # measurement: [x_R, y_R]
            z = np.array([[x_meas],
                          [y_meas]])

            H = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ])

            # measurement noise (tune these)
            R = np.diag([0.02, 0.02])

            def h(x_vec):
                # expected measurement from current state
                return x_vec[0:2, :]  # [x_R, y_R]

        else:
            # measurement: [x_R, y_R, theta_R]
            z = np.array([[x_meas],
                          [y_meas],
                          [wrap_angle(theta_meas)]])

            H = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ])

            R = np.diag([0.02, 0.02, 0.01])

            def h(x_vec):
                return np.array([
                    [x_vec[0, 0]],
                    [x_vec[1, 0]],
                    [x_vec[2, 0]],
                ])

        self._update(z, H, R, h)

    # ---------------- UPDATE: CAMERA SERVO ANGLE ----------------
    def update_camera_servo_angle(self, phi_meas: float):
        """
        Use a direct measurement of the camera servo angle (relative to robot).
        This only corrects phi_cam, not robot pose.
        """
        z = np.array([[wrap_angle(phi_meas)]])

        # H only observes phi_cam (4th state)
        H = np.array([[0.0, 0.0, 0.0, 1.0]])

        # angle measurement noise (tune)
        R = np.array([[0.005]])

        def h(x_vec):
            return np.array([[x_vec[3, 0]]])  # phi_cam

        self._update(z, H, R, h)

    # ---------------- GET NICE STATE BACK ----------------
    def get_state(self) -> RobotState:
        """Return RobotState object with nicely named fields."""
        return RobotState.from_vector(self.x)


def main(args=None):
    rclpy.init(args=args)
    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    # print("Setting camera orientation...")
    # set_camera_joint_once()
    
    try:
        print("Initializing the apriltag detector...")
        apriltag_detector = ApriltagDetector()
        rclpy.spin(apriltag_detector)
        
        print("Initializing the triangulator...")
        triangulator = Triangulator()
        rclpy.spin(triangulator)
        
        camera_sweeper = 

    except KeyboardInterrupt:
        print("Shutting down...")

        if rclpy.ok():
            stop_msg = Twist()
            for i in range(5):
                drive_publisher.publisher.publish(stop_msg)

            time.sleep(0.3)

    finally:
        try:
            drive_publisher.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()
