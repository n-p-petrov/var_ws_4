import argparse
import os

import cv2
from apriltag import apriltag

from apriltag_detector.utils import sharpen_img, upscale_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--image", required=True, help="path to input image containing AprilTag"
    )
    args = ap.parse_args()

    image = cv2.imread(args.image)
    enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    enhanced_img = sharpen_img(enhanced_img, 15, 0.8, 0.2)
    enhanced_img = upscale_img(enhanced_img, 5)

    detector = apriltag("tagStandard41h12")
    detections = detector.detect(enhanced_img)

    print(f"Detected {len(detections)} apriltags.")
    print(detections)
    print()

    line_color = (50, 255, 0)
    text_color = (50, 255, 0)

    height, width = image.shape[:2]
    enhanced_img = cv2.resize(
        enhanced_img,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    vis = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    for det in detections:
        corners = [(int(pt[0] / 5), int(pt[1] / 5)) for pt in det["lb-rb-rt-lt"]]

        for i in range(4):
            cv2.line(vis, corners[i], corners[(i + 1) % 4], line_color, 2)

        top_y = min(c[1] for c in corners)

        cv2.putText(
            vis,
            str(det["id"]),
            (corners[0][0], top_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )

    in_path = args.image
    in_dir = os.path.dirname(in_path)
    filename = os.path.basename(in_path)

    out_dir = os.path.join(in_dir, "enhanced_detections")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, filename)

    cv2.imwrite(out_path, vis)
    print(f"Saved output to: {out_path}")

    cv2.imshow("AprilTag Visualization", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
