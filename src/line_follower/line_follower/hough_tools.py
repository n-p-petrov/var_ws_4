import cv2
import numpy as np

def load_image(image_file):
    return cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # or load from topic

def circmean(data, high=2 * np.pi, low=0):
    data = np.asarray(data)
    if data.size == 0:
        return 0.0
    
    # Normalize to [0, 2π)
    angles = (data - low) * 2 * np.pi / (high - low)
    # Compute mean angle
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    # Map mean angle back to original range
    mean = (mean_angle * (high - low) / (2 * np.pi)) + low
    # Ensure result is within [low, high)
    if mean < low:
        mean += (high - low)
    elif mean >= high:
        mean -= (high - low)
    return float(mean)

def image_center(image):
    h = image.shape[0]
    w = image.shape[1]
    return w//2, h//2   # (x,y)

# def image_preprocess(image):
#     image = cv2.GaussianBlur(image, (5,5), 0)
#     _, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
#     image = cv2.Canny(image, 0, 255)
#     return image


def image_preprocess(image):
    """
    Baseline preprocessing (Canny-based):
    1) Blur
    2) High threshold to emphasize very bright strips
    3) Canny edges
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 0, 255)
    return edges

# New line detection algorithm
def image_preprocess_skeleton(image):
    """
    Alternative preprocessing (skeleton-based):
    1) Blur
    2) Threshold bright strips
    3) Morphological closing to fill gaps
    4) Morphological skeletonization (thinning)
    Returns a binary skeleton image.
    """
    # 1) blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # 2) threshold bright ceiling lights
    _, binary = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)

    # 3) close small gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) morphological skeletonization
    skeleton = np.zeros_like(closed)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = closed.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def get_edges(image, method="canny"):
    """
    Factory that picks the preprocessing algorithm.
    method: "canny" (baseline) or "skeleton" (new algorithm).
    Falls back to "canny" if unknown.
    """
    method = (method or "canny").lower()
    if method == "skeleton":
        return image_preprocess_skeleton(image)
    # default
    return image_preprocess(image)


# Hough transform functions

def polar_lines(edges, origin=(0, 0), full_circle=False):
    """
    Run standard Hough transform and express lines in polar form
    relative to a chosen origin.

    Returns:
        If full_circle == False:
            r_theta with shape (N, 2) containing (rho>=0, theta in [0, π))
        If full_circle == True:
            r_theta_fc with shape (N, 2) containing raw (rho, theta)
            where rho can be positive or negative and theta in [0, 2π)
    """
    if edges is None or edges.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)
    if lines is None:
        return np.empty((0, 2), dtype=np.float32)

    ox, oy = origin

    rhos = lines[:, 0, 0]
    thetas = lines[:, 0, 1]

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # shift rho so that it is measured from given origin
    rhos_shift = rhos - (ox * cos_t + oy * sin_t)

    # normalized representation (rho >= 0, theta in [0, π))
    rhos_norm = np.abs(rhos_shift.copy())
    thetas_norm = thetas.copy()
    r_theta = np.column_stack((rhos_norm, thetas_norm))

    if full_circle:
        # keep signed rho and original theta (for visualization)
        rhos_fc = rhos_shift.copy()
        thetas_fc = thetas.copy()

        # adjust negative rhos so lines are continuous over [0, 2π)
        neg_mask = rhos_fc < 0
        if np.any(neg_mask):
            rhos_fc[neg_mask] *= -1.0
            thetas_fc[neg_mask] += np.pi

        r_theta_fc = np.column_stack((rhos_fc, thetas_fc))
        return r_theta_fc

    return r_theta


def filter_lines(r_theta, tolerance=np.pi / 6):
    """
    Keep only lines whose angle is within `tolerance` of the circular mean.
    This filters out spurious orientations.
    """
    if r_theta is None or r_theta.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    cmean = circmean(r_theta[:, 1], high=np.pi)
    delta = np.abs(r_theta[:, 1] - cmean)
    mask = delta < tolerance
    return r_theta[mask]


def line_endpoints(x, y, a, b):
    """
    Compute long segment endpoints for a line defined by point (x, y)
    and unit direction (a, b).
    """
    x1 = int(round(x + 1000 * (-b)))
    y1 = int(round(y + 1000 * (a)))
    x2 = int(round(x - 1000 * (-b)))
    y2 = int(round(y - 1000 * (a)))
    return x1, x2, y1, y2


def draw_lines(r_theta, r_theta_fc, image, origin=(0, 0)):
    """
    Draw all detected Hough lines + the dominant average line and image center.
    """
    ox, oy = origin

    # draw all raw lines (for debug)
    if r_theta_fc is not None and r_theta_fc.size > 0:
        for r, theta in r_theta_fc:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = ox + r * a
            y0 = oy + r * b
            x1, x2, y1, y2 = line_endpoints(x0, y0, a, b)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

    # draw the dominant line (average theta)
    if r_theta is not None and r_theta.size > 0:
        avg_theta = circmean(r_theta[:, 1], high=np.pi)
        a = np.cos(avg_theta)
        b = np.sin(avg_theta)
        x1, x2, y1, y2 = line_endpoints(ox, oy, a, b)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 4)  # yellow

    # draw image center
    cv2.drawMarker(
        image,
        origin,
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=4,
    )

    return image