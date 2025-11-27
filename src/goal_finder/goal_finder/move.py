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
from goal_finder.kalman import EkfNode
from rclpy.node import Node
from sensor_msgs.msg import JointState
from goal_finder.ugv_obstacle_detector import UGVObstacleDetector

#to run together you’d typically use a MultiThreadedExecutor or similar in ROS2 CHECK LATER

LINEAR_VELOCITY = 0.2
DURATION_LINEAR_MOVE = 0.5  # seconds

ANGULAR_VELOCITY = math.pi / 4  # radians per second

# def main(args=None):
#     rclpy.init(args=args)
#     print("Initializing the drive publisher...")
#     drive_publisher = DrivePublisher()

#     # print("Setting camera orientation...")
#     # set_camera_joint_once()
    
#     try:
#         print("Initializing the apriltag detector...")
#         apriltag_detector = ApriltagDetector()
#         rclpy.spin(apriltag_detector)
        
#         print("Initializing the triangulator...")
#         triangulator = Triangulator()
#         rclpy.spin(triangulator)
        
#         print("Starting the kalman filter loop...")
#         kalman = EkfNode()
#         rclpy.spin(kalman)

#         # some moving here 

#         print("Starting the obstacle detection...")
#         ugv_obstacle_detector = UGVObstacleDetector()
#         rclpy.spin(ugv_obstacle_detector)

#     except KeyboardInterrupt:
#         print("Shutting down...")

#         if rclpy.ok():
#             stop_msg = Twist()
#             for i in range(5):
#                 drive_publisher.publisher.publish(stop_msg)

#             time.sleep(0.3)

#     finally:
#         try:
#             drive_publisher.destroy_node()
#         except Exception:
#             pass

#         if rclpy.ok():
#             rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    try:
        print("Initializing the apriltag detector...")
        apriltag_detector = ApriltagDetector()

        print("Initializing the triangulator...")
        triangulator = Triangulator()

        print("Starting the kalman filter loop...")
        kalman = EkfNode()

        print("Starting the obstacle detection...")
        ugv_obstacle_detector = UGVObstacleDetector()

        # ---- run all nodes together ----
        executor = rclpy.executors.MultiThreadedExecutor()

        executor.add_node(drive_publisher)
        executor.add_node(apriltag_detector)
        executor.add_node(triangulator)
        executor.add_node(kalman)
        executor.add_node(ugv_obstacle_detector)

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
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()
