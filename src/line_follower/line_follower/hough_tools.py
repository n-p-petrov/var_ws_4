import cv2
import numpy as np

def load_image(image_file):
    return cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # or load from topic

def circmean(data, high=2*np.pi, low=0):
    data = np.asarray(data)
    # Normalize data to [0, 2π)
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
    return mean

def image_center(image):
    h = image.shape[0]
    w = image.shape[1]
    return w//2, h//2   # (x,y)

def image_preprocess(image):
    image = cv2.GaussianBlur(image, (5,5), 0)
    _, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    image = cv2.Canny(image, 0, 255)
    return image

def polar_lines(edges, origin=(0,0), full_circle=False):
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
    ox, oy = origin

    rhos = lines[:,0,0]
    thetas = lines[:,0,1]

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    rhos_shift = rhos - (ox * cos_t + oy * sin_t) # ?

    rhos_norm = np.abs(rhos_shift.copy()) # only postive rho
    thetas_norm = thetas.copy() # theta in range [0, 1pi)
    r_theta = np.column_stack((rhos_norm, thetas_norm))

    # for visualizaiton
    if full_circle:
        thetas_fc = thetas.copy()
        rhos_fc = rhos_shift.copy()
        neg_mask = rhos_fc < 0
        if np.any(neg_mask):
            rhos_shift[neg_mask] *= -1.0 # only postive rho
            thetas[neg_mask] += np.pi # if rho was negtaive, shift angle 2 quadr
        r_theta_fc = np.column_stack((rhos_fc, thetas_fc))
        return r_theta_fc
    
    return r_theta

def filter_lines(r_theta, tolerance=np.pi/4):
    cmean = circmean(r_theta[:,1], high=np.pi)
    delta = r_theta[:,1] - cmean
    mask = delta < tolerance
    return r_theta[mask]

def line_endpoints(x,y,a,b):
    x1 = int(round(x + 1000 * (-b)))
    y1 = int(round(y + 1000 * (a)))
    x2 = int(round(x - 1000 * (-b)))
    y2 = int(round(y - 1000 * (a)))
    return x1,x2,y1,y2

def draw_lines(r_theta, r_theta_fc, imagefile, origin=(0,0)):
    image = cv2.imread(imagefile, cv2.IMREAD_COLOR_RGB)
    ox, oy = origin
    for r, theta in r_theta_fc:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = ox + r * a
        y0 = oy + r * b
        x1,x2,y1,y2 = line_endpoints(x0,y0,a,b)
        cv2.line(image, (x1, y1), (x2, y2), (255,0,0), 2)

    avg_theta = circmean(r_theta[:,1], high=np.pi)
    a = np.cos(avg_theta)
    b = np.sin(avg_theta)
    x1,x2,y1,y2 = line_endpoints(ox,oy,a,b)
    cv2.line(image, (x1, y1), (x2, y2), (255,2255,0), 4)

    cv2.drawMarker(image, origin, (0,255,0), markerType=cv2.MARKER_CROSS, 
                   markerSize=24, thickness=4)

    return image

