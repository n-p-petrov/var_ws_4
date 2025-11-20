import argparse

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    enhanced_img = sharpen_img(gray, 31, 0.8, 0.2)
    enhanced_img = upscale_img(enhanced_img, 5)

    detector = apriltag("tagStandard41h12")
    detections = detector.detect(enhanced_img)

    print(f"Detected {len(detections)} apriltags.")
    print(detections)
    print()

    line_color = (50, 255, 255)
    text_color = (50, 255, 255)

    vis = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    for det in detections:
        corners = [(int(pt[0]), int(pt[1])) for pt in det["lb-rb-rt-lt"]]

        for i in range(4):
            cv2.line(vis, corners[i], corners[(i + 1) % 4], line_color, 2)

        cv2.putText(
            vis,
            str(det["id"]),
            (int(det["center"][0]), int(det["center"][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )

    cv2.imshow("AprilTag Visualization", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
