import argparse

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Pose2D
from rclpy.node import Node
from std_msgs.msg import Float32


class GradientAngle(Node):
    def __init__(self, target):
        super().__init__("grad_angle")

    # ----- constants -----

        # self.position_topic = "/triangulated_pos"
        # self.orient_topic = "/orientation"
        self.pose_topic = "/filtered_pose"
        self.obstacle_topic = "/obstacle_detected"

        self.r_pos = None
        self.r_angle = None
        self.obs_pos = None
        self.grad_angle = None
        self.gradient = None

        '''
        (0,0)
            +--+-+-------+-+--+
            |  | D-------B |  |
            |  |     C     |  |
            |  E-----------A  |
            |                 |
            |                 |
            +-----------------+
            |                 |
            |                 |
            |  J-----------F  |
            |  |     H     |  |
            |  | I-------G |  |
            +--+-+-------+-+--+
        '''
   
        self.targets = {
            "A":    np.array([5700.0,   2350.0]),
            "B":    np.array([4800.0,   1300.0]),
            "C":    np.array([3700.0,   2000.0]),
            "D":    np.array([2600.0,   1300.0]),
            "E":    np.array([1700.0,   2350.0]),
            "F":    np.array([5700.0,   8050.0]),
            "G":    np.array([4800.0,   9100.0]),
            "H":    np.array([3700.0,   8400.0]),
            "I":    np.array([2600.0,   9100.0]),
            "J":    np.array([1700.0,   8050.0]),
        }

        self.target = self.targets[target]

        self.K = np.array([
            [1011.2320556640625,    0.0,                    643.5490112304688],
            [0.0,                   1011.1708374023438,     373.5168151855469],
            [0.0,                   0.0,                    1.0              ]
        ])

    # ----- subscribers/publishers
        self.pose_sub = self.create_subscription(
            Pose2D, self.pose_topic, self.pose_callback, 10
        )
        self.obstacle_sub = self.create_subscription(
            PointStamped, self.obstacle_topic, self.obstacle_callback, 10
        )
        self.timer = self.create_timer(0.1, self.timer_callback)

        # publish
        # gradient
        # obstacle location

        self.gradient_pub = self.create_publisher(
            Pose2D, "/grad/gradient", 10
        )
        self.obstacle_pub = self.create_publisher(
            Point, "grad/obstacle", 10
        )


    # ----- callbacks -----

    def obstacle_callback(self, msg:PointStamped):
        #print("obstacle_callback", msg)
        if self.r_angle and self.r_pos is not None:
            u = msg.point.x
            v = msg.point.y
            z = msg.point.z
            if z > 0:
                self.obs_pos = self.obstacle_world_coords(u,v,z)

    def pose_callback(self, msg:Pose2D):
        #print("pose_callback", msg)
        self.r_pos = np.array([msg.x, msg.y])
        self.r_angle = msg.theta

        c = np.cos(self.r_angle)
        s = np.sin(self.r_angle)
        R = np.array([[c,s],[-s,c]])
        self.v_heading = R @ np.array([1.0, 0.0]) # heading direction as vector
        self.v_perp = R @ np.array([0.0, 1.0]) #

    def publish_state(self):
        if self.gradient is not None:
            grad_msg = Pose2D()
            grad_msg.x = self.gradient[0]
            grad_msg.y = self.gradient[1]
            grad_msg.theta = self.grad_angle
            self.gradient_pub.publish(grad_msg)

        if self.obs_pos is not None:
            obs_msg = Point()
            obs_msg.x = self.obs_pos[0]
            obs_msg.y = self.obs_pos[1]
            self.obstacle_pub.publish(obs_msg)

    def timer_callback(self):
        if self.r_angle and self.r_pos is not None:
            self.get_logger().info(f"Grad delta: {self.grad_angle}")
            self.grad_angle = self.calc_grad_angle()
        self.publish_state()

    # ----- coordinate transformations -----

    def obstacle_world_coords(self, u,v,z):
        if self.r_angle and self.r_pos is not None:
            z *= 1000 # to mm
            obs_cam = z * (np.linalg.inv(self.K) @ np.array([u,v,1]).T)
            fwd_offset = obs_cam[2] * self.v_heading
            side_offset = obs_cam[0] * self.v_perp
            obs_relto_robot = fwd_offset + side_offset
            obs_world = self.r_pos - obs_relto_robot
            return obs_world
    
    # ----- gradients -----

    def U_att_grad(self, alpha=1.0):
        if self.r_angle and self.r_pos is not None:
            grad = alpha * (self.r_pos - self.target)
            return grad
    
    def U_rep_grad(self, radius=250, beta=1.0):
        if self.r_angle and self.r_pos is not None:
            d = np.linalg.norm(self.r_pos - self.obs_pos)
            if d <= radius:
                grad = -beta * ((1 / d) - (1 / radius)) * ((self.r_pos - self.obs_pos)/d**3)
            else:
                grad = np.array([0.0, 0.0])
            return grad
    
    # note: this is not the gradient angle, but rather the angle between
    #       the gradient and the heading direction.
    def calc_grad_angle(self):
        unit = lambda u: u/np.linalg.norm(u)

        if self.r_angle and self.r_pos is not None:
            gradient = -self.U_att_grad()
            if self.obs_pos is not None:
                gradient -= self.U_rep_grad()
            self.gradient = gradient

            cos_a = unit(self.v_heading).T @ unit(self.gradient)
            angle = np.arccos(cos_a)

            h3 = np.array([self.v_heading[0], self.v_heading[1], 0.0])
            g3 = np.array([self.gradient[0], self.gradient[1], 0.0])
            cross = np.cross(h3, g3)
            angle = -angle if cross[-1]<=0.0 else angle

            return angle


def main(ros_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="C")
    args = parser.parse_args()
    rclpy.init(args=ros_args)
    grad_node = GradientAngle(target=args.target)
    rclpy.spin(grad_node)
    grad_node.destroy_node()
    rclpy.shutdown()
