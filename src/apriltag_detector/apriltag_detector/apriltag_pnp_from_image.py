#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
from apriltag import apriltag
# from pupil_apriltags import Detector

from apriltag_detector.utils import sharpen_img, upscale_img


# camera + tag configuration
# from /camera_info (raw camera)
FX = 289.11451
FY = 289.75319
CX = 347.23664
CY = 235.67429
DIST_COEFFS = np.array(
    [-0.208848, 0.028006, -0.000705, -0.00082, 0.0], dtype=np.float64
).reshape(-1, 1)

# physical size of your printed tag (meters)
TAG_SIZE_M = 0.288  # 28.8 cm


def build_camera_matrix(width, height):
    """
    Build a camera matrix adapted to the current image size.

    Original calibration was done at 640x480. If the screenshot has
    a different size, we scale fx, fy, cx, cy accordingly.
    """
    orig_w, orig_h = 640.0, 480.0
    scale_x = width / orig_w
    scale_y = height / orig_h

    fx = FX * scale_x
    fy = FY * scale_y
    cx = CX * scale_x
    cy = CY * scale_y

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--image", required=True, help="path to input image containing AprilTag"
    )
    args = ap.parse_args()

    # load and preprocess
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # optional contrast improvement
    gray = cv2.equalizeHist(gray)

    enhanced_img = sharpen_img(gray, 31, 0.8, 0.2)
    enhanced_img = upscale_img(enhanced_img, 5)
    scaling_factor = 5.0

    h, w = gray.shape
    camera_matrix = build_camera_matrix(w, h)

    detector = apriltag("tagStandard41h12")
    detections = detector.detect(enhanced_img)

    print(f"Detected {len(detections)} apriltags.")

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for det in detections:
        # corners come from upscaled image -> scale them back
        corners_up = np.array(det["lb-rb-rt-lt"]).reshape(4, 2)
        corners = corners_up / scaling_factor

        # 2D image points
        image_points = corners.astype(np.float32)

        # 3D model of tag corners in tag reference frame (lb, rb, rt, lt)
        S = TAG_SIZE_M
        object_points = np.array(
            [
                [-S / 2.0,  S / 2.0, 0.0],  # lb
                [ S / 2.0,  S / 2.0, 0.0],  # rb
                [ S / 2.0, -S / 2.0, 0.0],  # rt
                [-S / 2.0, -S / 2.0, 0.0],  # lt
            ],
            dtype=np.float32,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            print(f"PnP failed for tag {det['id']}")
            continue

        tvec = tvec.reshape(3)
        distance = float(np.linalg.norm(tvec))

        print(
            f"Tag {det['id']}: t = ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}) m, "
            f"distance ≈ {distance:.3f} m"
        )

        # draw on visualization image
        # draw box
        corners_int = [(int(x), int(y)) for x, y in corners]
        for i in range(4):
            cv2.line(
                vis,
                corners_int[i],
                corners_int[(i + 1) % 4],
                (50, 255, 0),
                2,
            )

        # center + text
        cx = int(det["center"][0] / scaling_factor)
        cy = int(det["center"][1] / scaling_factor)

        cv2.putText(
            vis,
            f"ID {det['id']}",
            (cx, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"{distance:.2f} m",
            (cx, cy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

    cv2.imshow("AprilTag PnP Visualization", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
