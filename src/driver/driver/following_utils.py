from geometry_msgs.msg import Twist
import math

def line2twist(line_angle, allow_reverse=True):
    """
    Simple controller for line following.
    line_angle: radians, relative to robot forward (0 = straight ahead,
                 positive = line rotated to the right).
    Returns: geometry_msgs.msg.Twist
    """
    # assume im on the line
    msg = Twist()
    k_ang = 1.5   # proportional gain for steering
    k_lin = 0.2   # forward speed when roughly straight
    max_ang = 1.0 # rad/s clamp

    # normalize angle to [-pi, pi]
    ang = math.atan2(math.sin(line_angle), math.cos(line_angle))

    forward_sign = 1.0
    if allow_reverse and abs(ang) > math.pi/2:
        # flip direction by pi so we rotate less; drive backward instead
        ang = math.atan2(math.sin(ang - math.copysign(math.pi, ang)),
                         math.cos(ang - math.copysign(math.pi, ang)))
        forward_sign = -1.0
        
    # angular velocity proportional to angle error
    msg.angular.z = max(-max_ang, min(max_ang, -k_ang * ang))

    # slow down when the error is large
    msg.linear.x = forward_sign * k_lin * max(0.0, 1.0 - abs(ang) / (math.pi / 4))

    return msg

def get_angle_from_line(list_of_line_params):
    # assume pairs of (rho, theta)
    eps = 1e-2
    closest = list_of_line_params.sorted(key=lambda x: abs(x[0]))
    if closest[0][0] < eps:
        return closest[0][1]
    return None

    
def image2twist(image):
    # ######### work in progress ##############3
    
    #assume line speicifc color
    line_color = [255, 0, 0]  # red line ig
    # assume im in the middle and on the line
    midpoint = (image.width // 2, image.height // 2)
    
    
    
    