import cv2


def upscale_img(gs_img, scaling_factor):
    height, width = gs_img.shape[:2]
    upscaled_img = cv2.resize(
        gs_img,
        (width * scaling_factor, height * scaling_factor),
        interpolation=cv2.INTER_CUBIC,
    )

    return upscaled_img


def sharpen_img(gs_img, gaussian_kernel_size, orig_frac, high_pass_frac):
    blurred_img = cv2.GaussianBlur(gs_img, (gaussian_kernel_size, gaussian_kernel_size), 0)
    high_pass = cv2.subtract(gs_img, blurred_img)
    sharpened_img = cv2.addWeighted(gs_img, orig_frac, high_pass, high_pass_frac, 0)

    return sharpened_img
