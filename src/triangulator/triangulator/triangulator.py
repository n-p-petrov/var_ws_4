import argparse

import cv2  # pip install opencv-python
import numpy as np
import rclpy
from apriltag import apriltag  # pip install apriltag
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray, Point
from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
from sympy import geometry


class Triangulator(Node):
    # so far assuming that distances are perpendicular projections on ground
    # TODO account for different heights of qr codes
    def __init__(self):
        super().__init__("triangulator")

        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag_detections", self.listener_callback, 10
        )
        self.coordinates_publisher = self.create_publisher(
            tuple , "/triangulated_pos", self.publisher_coords, 10              #TODO define coords
        )

        self.qr_coords = { # 00 at arnouds desk, 7510 x 10520  at window at computers
                        #   assuming objects are in the middle of the lines
            1: (3755, 9680),
            2: (760, 8080),
            3: (6750, 8080),
            4: (760, 5250),
            5: (6750, 4500),
            6: (620, 3380), #solid
            7: (7510-110, 3380), # solid
            8: (890, 760),
            9: (3755, 760),
            10: (7510 - 880, 760),
            }
        self.width = 7510  # mm 760 + 6000 + 750
        self.height = 10520  # mm  760 + 9000 + 760
        # 
        
        self.pos = None
        
    def two_point_triangl(self, qrids, distances) -> tuple:
        coords1 = self.qr_coords.get(qrids[0])
        coords2 = self.qr_coords.get(qrids[1])
        
        qr1 = geometry.Point2D(coords1[0], coords1[1])
        qr2 = geometry.Point2D(coords2[0], coords2[1])
        
        intersection = geometry.Circle(qr1, distances[0]).intersection(geometry.Circle(qr2, distance[1]))
        print('found intersections:', intersection)
        for i in intersection:
            if 0 <= i.x.evalf() <= self.width and 0 <= i.y.evalf() <= self.height:
                return (i.x.evalf(), i.y.evalf())
            
    
    def add_position(self, position):
        self.last10_positions.append(position)
        if len(self.last10_positions) > 10:
            self.last10_positions.pop(0)
        
    def multipoint_triangl(self, qrids, distances):
        # try using as a reference the most midle qr TODO
        # https://www.mi.fu-berlin.de/inf/groups/ag-tech/projects/ls2/ipin.pdf
        # solve by lstsq by susbstracting the circles
        qrids = np.array(qrids, float)
        distances  = np.array(distances,  float)

        x1, y1 = qrids[0]
        r1 = distances[0]
        A = []
        b = []

        for qrid, ri in zip(qrids[1:], distances[1:]):
            xi, yi = self.qr_coords.get(qrid)
            A.append([2*(xi - x1), 2*(yi - y1)])
            b.append(r1**2 - ri**2 + xi**2 - x1**2 + yi**2 - y1**2)

        A = np.array(A)
        b = np.array(b)

        pos, *_ = np.linalg.lstsq(A, b, rcond=None)  # least-squares solution
        return pos  # (x, y)
    
    
    
    def listener_callback(self, qrid_distance):
        n_tags = len(qrid_distance.detections)
        tag_ids = [det.id for det in qrid_distance.detections]
        tag_distances = [det.goodness for det in qrid_distance.detections]  # placeholder for actual distance
        if n_tags < 2:
            self.get_logger().info("Not enough QR codes detected for triangulation.")
            self.pos = None
        if n_tags == 2:
            position = self.two_point_triangl(tag_ids, tag_distances)
            if position:
                self.pos = position
                self.get_logger().info(f"Triangulated Position: x={position[0]}, y={position[1]}")
            else:
                self.get_logger().info("No valid intersection point found within bounds.")
                self.pos = None
        else:
            position = self.multipoint_triangl(tag_ids, tag_distances)
            if position:
                self.get_logger().info(f"Triangulated Position: x={position[0]}, y={position[1]}")
                self.pos = position
            else:
                self.get_logger().info("No valid intersection point found within bounds.")
                self.pos = None
                
    def publisher_coords(self):
        if self.pos:
            msg = self.pos
            self.coordinates_publisher.publish(msg)
            self.get_logger().info(f"Published triangulated position: {self.pos}.")
                
                
def main(args=None):
    rclpy.init(args=args)
    node = Triangulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
