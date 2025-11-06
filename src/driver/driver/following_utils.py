from geometry_msgs.msg import Twist
import math
import numpy as np
import cv2

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

    

def img2twist(img_bgr, yaw_offset=-math.pi/2):
    """
    Input:  BGR image (np.ndarray) with a pure red line (255,0,0) in BGR.
    Output: geometry_msgs.msg.Twist for following the longer continuation
            of that line from the image center.
    """
    # --- make simple binary mask for pure red pixels ---
    mask = np.all(img_bgr == (255, 0, 0), axis=-1).astype(np.uint8) * 255

    H, W = mask.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    # --- tiny helper to find longest ray of red from center ---
    def longest_red_ray(mask, cx, cy, step=1, corridor=3, gap_tol=2, n0=64):
        def inb(x, y): return 0 <= x < W and 0 <= y < H
        def raylen(theta):
            vx, vy = math.cos(theta), math.sin(theta)
            nx, ny = -vy, vx
            x, y, gaps, L = cx, cy, 0, 0.0
            while True:
                x += step * vx; y += step * vy
                ix, iy = int(round(x)), int(round(y))
                if not inb(ix, iy): break
                # check corridor
                hit = 0
                for t in range(-corridor, corridor + 1):
                    px, py = int(round(ix + t * nx)), int(round(iy + t * ny))
                    if inb(px, py) and mask[int(py), int(px)]:
                        hit += 1
                if hit > 0:
                    gaps = 0
                else:
                    gaps += 1
                    if gaps > gap_tol: break
                L += step
            return L

        thetas = np.linspace(-math.pi, math.pi, n0, endpoint=False)
        lengths = [raylen(t) for t in thetas]
        best_i = int(np.argmax(lengths))
        best_th, best_L = thetas[best_i], lengths[best_i]
        opp_L = raylen(best_th + math.pi)
        if opp_L > best_L:
            best_th += math.pi; best_L = opp_L
        return math.atan2(math.sin(best_th), math.cos(best_th)), best_L

    theta_img, length_px = longest_red_ray(mask, cx, cy)
    if length_px <= 0:
        return Twist()  # no line found → stop

    # --- map image angle to robot yaw error ---
    err = math.atan2(math.sin(theta_img + yaw_offset),
                     math.cos(theta_img + yaw_offset))

    # --- proportional steering & slowdown ---
    k_ang, k_lin, max_ang = 1.5, 0.22, 1.0
    cutoff = math.pi / 4
    ang = max(-max_ang, min(max_ang, -k_ang * err))
    lin = k_lin * max(0.0, 1.0 - min(abs(err)/cutoff, 1.0))

    cmd = Twist()
    cmd.linear.x = lin
    cmd.angular.z = ang
    return cmd
