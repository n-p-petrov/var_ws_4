import argparse
import math
from itertools import combinations

import cv2  # pip install opencv-python
import numpy as np
import rclpy
from apriltag import apriltag  # pip install apriltag
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from geometry_msgs.msg import Point
from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
from sympy import geometry


class Triangulator(Node):
    # so far assuming that distances are perpendicular projections on ground
    # TODO account for different heights of qr codes
    def __init__(self):
        super().__init__("triangulator")

        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray, "/apriltag/detections", self.listener_callback, 10
        )
        self.coordinates_publisher = self.create_publisher(
            Point, "/triangulated_pos", 10
        )
        self.last_coordinates_publisher = self.create_publisher(
            Point, "/last_valid_triangulated_pos", 10
        )

        self.qr_coords = {  # 00 at arnouds desk, 7510 x 10520  at window at computers
            #   assuming objects are in the middle of the lines
            1: (3755, 9680),
            2: (760, 8080),
            3: (6750, 8080),
            4: (760, 5250),
            5: (6750, 4500),
            6: (620, 3380),  # solid
            7: (7510 - 110, 3380),  # solid
            8: (890, 760),
            9: (3755, 760),
            10: (7510 - 880, 760),
        }
        self.width = 7510  # mm 760 + 6000 + 750
        self.height = 10520  # mm  760 + 9000 + 760
        #

        self.pos = None
        self.last_valid_pos = (-1, -1)

        self.image_width = 640  # placeholder, set according to actual image size
        self.image_height = 480  # placeholder, set according to actual image size

    # def two_point_triangl(self, qrids, distances) -> tuple:
    #     coords1 = self.qr_coords.get(qrids[0])
    #     coords2 = self.qr_coords.get(qrids[1])

    #     qr1 = geometry.Point2D(coords1[0], coords1[1])
    #     qr2 = geometry.Point2D(coords2[0], coords2[1])
    #     # self.get_logger().info(f'triangulating between QR {qrids[0]} at {coords1} and QR {qrids[1]} at {coords2} with distances {distances[0]}, {distances[1]}')
    #     # self.get_logger().info(f'Circles: Center1 {qr1}, Radius1 {distances[0]}; Center2 {qr2}, Radius2 {distances[1]}')

    #     intersection = geometry.Circle(qr1, distances[0]).intersection(geometry.Circle(qr2, distances[1]))
    #     # self.get_logger().info(f'found intersections: {intersection}')
    #     for i in intersection:
    #         self.get_logger().info(f'checking intersection: {(i.x.evalf(), i.y.evalf())}')
    #         # print('checking intersection:', i)
    #         if 0 <= i.x.evalf() <= self.width and 0 <= i.y.evalf() <= self.height:
    #             # print('valid intersection:', i)
    #             return (i.x.evalf(), i.y.evalf())

    def two_point_triangl(self, qrids, distances, qrmidpoints):
        id1, id2 = qrids

        # World coords of the tags
        t1 = self.qr_coords[id1]
        t2 = self.qr_coords[id2]

        C1 = geometry.Circle(geometry.Point2D(*t1), distances[0])
        C2 = geometry.Circle(geometry.Point2D(*t2), distances[1])

        intersections = C1.intersection(C2)

        # Keep only intersections inside the field
        candidates = []
        for inter in intersections:
            x = float(inter.x)
            y = float(inter.y)
            if 0 <= x <= self.width and 0 <= y <= self.height:
                candidates.append((x, y))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # ---- 1. Compute the image-space signed angle ----
        # Convert pixel midpoints into camera-centered coordinates
        u1, v1 = qrmidpoints[0]
        u2, v2 = qrmidpoints[1]

        cx = self.image_width / 2
        cy = self.image_height / 2

        img_ang = signed_angle_deg((cx, cy), (u1, v1), (u2, v2))
        img_sign = 1 if img_ang > 0 else -1 if img_ang < 0 else 0
        img_abs = abs(img_ang)

        # ---- 2. Evaluate each candidate by sign + magnitude ----
        best = None
        best_err = None

        for P in candidates:
            world_ang = signed_angle_deg(P, t1, t2)
            world_sign = 1 if world_ang > 0 else -1 if world_ang < 0 else 0
            world_abs = abs(world_ang)

            # Must have same left/right ordering
            if world_sign != img_sign:
                continue

            # Magnitude consistency error
            err = abs(world_abs - img_abs)

            if best is None or err < best_err:
                best = P
                best_err = err

        # If no candidate fits the observation → impossible geometry
        if best is None:
            return None

        return best

    def add_position(self, position):
        self.last10_positions.append(position)
        if len(self.last10_positions) > 10:
            self.last10_positions.pop(0)

    def extract_most_reliable(self, qr_ids, qr_distances, qr_mids):
        # choose the qr code closest to the center of the image
        center_x = self.image_width / 2
        center_y = self.image_height / 2
        print(
            np.array(center_x - qr_mids[0]) ** 2 + np.array(center_y - qr_mids[1]) ** 2
        )
        min_dist_qr = None
        most_middle_idx = -1
        for idx, (x, y) in enumerate(zip(qr_mids[0], qr_mids[1])):
            dist = (center_x - x) ** 2 + (center_y - y) ** 2
            if min_dist_qr is None or dist < min_dist_qr:
                min_dist_qr = dist
                most_middle_idx = idx

        # self.get_logger().info(f'Selected anchor QR code ID {qr_ids[most_middle_idx]} at index {most_middle_idx} as most reliable.')
        anchor_qr_id = qr_ids[most_middle_idx]
        anchor_distance = qr_distances[most_middle_idx]

        list_of_others = qr_ids[:most_middle_idx] + qr_ids[most_middle_idx + 1 :]
        list_of_distances = (
            qr_distances[:most_middle_idx] + qr_distances[most_middle_idx + 1 :]
        )
        # self.get_logger().info(f'other QR codes: {list_of_others} with distances {list_of_distances}.')

        return anchor_qr_id, anchor_distance, list_of_others, list_of_distances

    # def multipoint_triangl(self, qrids, distances, midpoints):
    #     # try using as a reference the most midle qr TODO
    #     # https://www.mi.fu-berlin.de/inf/groups/ag-tech/projects/ls2/ipin.pdf
    #     # solve by lstsq by susbstracting the circles
    #     qrids = np.array(qrids, float)
    #     distances  = np.array(distances,  float)
    #     qr_mids = np.array(midpoints, float)

    #     anchor_id, anchor_dist, other_ids, other_dists = self.extract_most_reliable(qrids.tolist(), distances.tolist(), qr_mids)
    #     x1, y1 = self.qr_coords.get(anchor_id)
    #     A = []
    #     b = []

    #     for qrid, ri in zip(other_ids, other_dists):
    #         xi, yi = self.qr_coords.get(qrid)
    #         A.append([2*(xi - x1), 2*(yi - y1)])
    #         b.append(anchor_dist**2 - ri**2 + xi**2 - x1**2 + yi**2 - y1**2 )

    #     A = np.array(A)
    #     b = np.array(b)
    #     pos, *_ = np.linalg.lstsq(A, b, rcond=None)  # least-squares solution
    #     # x = np.clip(pos[0], 0, self.width)
    #     # y = np.clip(pos[1], 0, self.height)
    #     return (x,y)  # (x, y)

    def multipoint_triangl(self, qrids, distances, midpoints):
        # all combinations of midpoiints
        qrids = np.array(qrids, float)
        distances = np.array(distances, float)
        qr_mids = np.array(midpoints, float)

        possible_positions = []
        for (i1, id1), (i2, id2) in combinations(enumerate(qrids), 2):
            pos = self.two_point_triangl(
                [id1, id2],
                [distances[i1], distances[i2]],
                ([qr_mids[0][i1], qr_mids[1][i1]], [qr_mids[0][i2], qr_mids[1][i2]]),
            )
            possible_positions.append(pos)

        if not possible_positions:
            # self.get_logger().info("No possible positions found from combinations.")
            return None

        # possible_pos = np.array([possible_positions], float)
        avg_x = np.mean([p[0] for p in possible_positions if p is not None])
        avg_y = np.mean([p[1] for p in possible_positions if p is not None])
        return (avg_x, avg_y)

    def listener_callback(self, qrid_distance):
        n_tags = len(qrid_distance.detections)
        tag_ids = [det.id for det in qrid_distance.detections]
        tag_distances = [
            det.goodness * 1000 for det in qrid_distance.detections
        ]  # from m to mm
        tag_midpoints = (
            [det.centre.x for det in qrid_distance.detections],
            [det.centre.y for det in qrid_distance.detections],
        )
        if n_tags < 2:
            # self.get_logger().info("Not enough QR codes detected for triangulation.")
            self.pos = (-1.0, -1.0)
        elif n_tags == 2:
            position = self.two_point_triangl(tag_ids, tag_distances, tag_midpoints)
            if position is not None:
                self.last_valid_pos = position
                self.pos = position
                # self.get_logger().info(f"Triangulated Position: x={position[0]}, y={position[1]} from 2")
            else:
                # self.get_logger().info("No valid intersection point found within bounds.")
                self.pos = (-1.0, -1.0)
        else:
            position = self.multipoint_triangl(tag_ids, tag_distances, tag_midpoints)
            if position is not None:
                # self.get_logger().info(f"Triangulated Position: x={position[0]}, y={position[1]}")
                self.last_valid_pos = position
                self.pos = position
            else:
                # self.get_logger().info("No valid intersection point found within bounds.")
                self.pos = (-1.0, -1.0)
        point_msg = Point(x=float(self.pos[0]), y=float(self.pos[1]), z=0.0)
        self.coordinates_publisher.publish(point_msg)
        msg2 = Point(
            x=float(self.last_valid_pos[0]), y=float(self.last_valid_pos[1]), z=0.0
        )
        self.last_coordinates_publisher.publish(msg2)
        # self.get_logger().info(f"Published triangulated position: {self.pos}.")


def signed_angle_deg(center, p1, p2):
    cx, cy = center
    x1, y1 = p1
    x2, y2 = p2

    v1 = (x1 - cx, y1 - cy)
    v2 = (x2 - cx, y2 - cy)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]  # cross product z

    ang = math.degrees(math.atan2(det, dot))
    return ang  # signed (-180, 180]


def main(args=None):
    rclpy.init(args=args)
    node = Triangulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
