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
from rclpy.executors import MultiThreadedExecutor


LINEAR_VELOCITY = 0.2
DURATION_LINEAR_MOVE = 0.5  # seconds

ANGULAR_VELOCITY = math.pi / 4  # radians per second



def main(args=None):
    rclpy.init(args=args)
    print("Initializing the drive publisher...")
    drive_publisher = DrivePublisher()

    # print("Setting camera orientation...")
    # set_camera_joint_once()
    
    executor = MultiThreadedExecutor()
    managed_nodes = []

    try:
        print("Initializing the apriltag detector...")
        apriltag_detector = ApriltagDetector()
        managed_nodes.append(apriltag_detector)

        print("Initializing the triangulator...")
        triangulator = Triangulator()
        managed_nodes.append(triangulator)

        print("Starting the kalman filter loop...")
        kalman = EkfNode()
        managed_nodes.append(kalman)

        for node in managed_nodes:
            executor.add_node(node)

        print("Spinning executor with detector, triangulator, and EKF nodes...")
        executor.spin()

        # some moving here 
        
    except KeyboardInterrupt:
        print("Shutting down...")

        if rclpy.ok():
            stop_msg = Twist()
            for i in range(5):
                drive_publisher.publisher.publish(stop_msg)

            time.sleep(0.3)

    finally:
        for node in managed_nodes:
            try:
                executor.remove_node(node)
            except Exception:
                pass

            try:
                node.destroy_node()
            except Exception:
                pass

        executor.shutdown()

        try:
            drive_publisher.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()
