import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Pose2D
from std_msgs.msg import Float32

class GradientAngle(Node):
    def __init__(self):
        super().__init__("grad_angle")

    # ----- constants -----

        # self.position_topic = "/triangulated_pos"
        # self.orient_topic = "/orientation"
        self.pose_topic = "/filtered_pose"
        self.obstacle_topic = "/obstacle_detected"

        self.r_pos = None
        self.r_angle = None
        self.obs_pos = None


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


        # self.position_sub = self.create_subscription(
        #     Pose2D, self.position_topic, self.position_callback, 10
        # )
        # self.orient_sub = self.create_subscription(
        #     Float32, self.orient_topic, self.orient_callback, 10
        # )
        # angle publisher

    # ----- callbacks -----

    def obstacle_callback(self, msg:PointStamped):
        print("obstacle_callback", msg)
        if self.r_angle and self.r_pos is not None:
            u = msg.point.x
            v = msg.point.y
            z = msg.point.z
            if z > 0:
                self.obs_pos = self.obstacle_world_coords(u,v,z)

    def pose_callback(self, msg:Pose2D):
        print("pose_callback", msg)
        self.r_pos = np.array([msg.x, msg.y])
        self.r_angle = msg.theta

    # def position_callback(self, msg:Pose2D):
    #     self.r_pos = np.array([msg.x, msg.y])

    # def orient_callback(self, msg:Float32):
    #     self.r_angle = msg.position[0]

    def timer_callback(self):
        if self.r_angle and self.r_pos is not None:
            grad_angle = self.grad_angle()
    
    # ----- coordinate transformations -----

    def obstacle_world_coords(self, u,v,z):
        if self.r_angle and self.r_pos is not None:
            z *= 1000 # to mm
            ray_cam_obst = np.linalg.inv(self.K) @ np.array([u,v,1]).T
            unit_ray = ray_cam_obst / np.linalg.norm(ray_cam_obst)
            theta = np.arccos(unit_ray[-1]) # angle optical and ray
            ray_y0 = ray_cam_obst.copy()
            ray_y0[1] = 0.0
            cross = np.cross(np.array([0,0,1]), ray_y0)
            theta = -theta if cross[1] < 0 else theta  # obstacle left or right of optical
            rho = np.linalg.norm(z * ray_y0)

            angle = theta + self.r_angle
            ca = np.cos(angle)
            sa = np.sin(angle)

            v = np.array([1,0])
            R = np.array([[ca, sa],[-sa, ca]]) # clockwise
            obs_relto_robot = rho * (R@v)
            obs_world = np.array(self.r_pos) - obs_relto_robot

            print("[OBSTACLES]")
            print(f"obstacle relto robot:   {obs_relto_robot}")
            print(f"obs in world coord  :   {obs_world}")

            return obs_world
    
    # ----- gradients -----

    def U_att_grad(self, t="A", alpha=1.0):
        if self.r_angle and self.r_pos is not None:
            grad = alpha * (self.r_pos - self.targets[t])
            return grad
    
    def U_rep_grad(self, radius=250, beta=1.0):
        if self.r_angle and self.r_pos is not None:
            d = np.linalg.norm(self.r_pos - self.obs_pos)
            if d <= self.rho:
                grad = -beta * ((1 / d) - (1 / radius)) * ((self.r_pos - self.obs_pos)/d**3)
            else:
                grad = np.array([0.0, 0.0])
            return grad
    
    def grad_angle(self):
        if self.r_angle and self.r_pos is not None:
            gradient = -1 * (self.U_att_grad() + self.U_rep_grad()) # descent
            grad_angle = -np.atan2(gradient[1], gradient[0])

            print("[GRADIENTS]")
            print(f"gradient :   {gradient}")
            print(f"angle    :   {grad_angle:.3f}")
            print(f"magnitude:   {np.linalg.norm(gradient)}")

            return grad_angle


def main(args=None):
    rclpy.init(args=args)
    ekf_node = GradientAngle()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()