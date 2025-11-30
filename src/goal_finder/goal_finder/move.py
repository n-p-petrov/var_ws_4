import argparse
import time

import rclpy
from apriltag_detector.apriltag_detector import ApriltagDetector
from apriltag_detector.apriltag_search_node import AprilTagSearchNode
from driver.driver_publisher import DrivePublisher
from geometry_msgs.msg import Twist
from triangulator.triangulator import Triangulator

from goal_finder.grad_angle import GradientAngle
from goal_finder.kalman import EkfNode
from goal_finder.mover_node import MoverNode
from goal_finder.ugv_obstacle_detector import UGVObstacleDetector


def main(ros_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="C")
    args, _ = parser.parse_known_args()

    rclpy.init(args=ros_args)
    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    try:
        print("Initializing the apriltag detector...")
        apriltag_detector = ApriltagDetector()

        print("Initializing the apriltag search node...")
        apriltag_search_node = AprilTagSearchNode()

        print("Initializing the triangulator...")
        triangulator = Triangulator()

        print("Starting the kalman filter loop...")
        kalman = EkfNode()

        print("Starting the obstacle detection...")
        ugv_obstacle_detector = UGVObstacleDetector()

        print("Starting the gradient node...")
        gradient_node = GradientAngle(target=args.target)

        print("Starting the mover node...")
        mover_node = MoverNode(drive_publisher)
        # ---- run all nodes together ----
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=7)

        # executor.add_node(drive_publisher)
        executor.add_node(apriltag_detector)
        executor.add_node(apriltag_search_node)
        executor.add_node(triangulator)
        executor.add_node(kalman)
        executor.add_node(ugv_obstacle_detector)
        executor.add_node(gradient_node)
        executor.add_node(mover_node)

        print("Spinning all nodes...")
        executor.spin()

    except KeyboardInterrupt:
        print("Shutting down on Ctrl+C...")

        if rclpy.ok():
            stop_msg = Twist()
            for _ in range(5):
                drive_publisher.publisher.publish(stop_msg)
            time.sleep(0.3)

    finally:
        # clean up
        try:
            drive_publisher.destroy_node()
            apriltag_detector.destroy_node()
            triangulator.destroy_node()
            kalman.destroy_node()
            ugv_obstacle_detector.destroy_node()
            apriltag_search_node.destroy_node()
            gradient_node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()
