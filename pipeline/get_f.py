import ImageFunctions as imagefunctions
import numpy as np
def get_features(img):
    return [
            imagefunctions.num_corners(img),
            imagefunctions.num_edges(img),
            imagefunctions.num_red_pixels(img),
            imagefunctions.num_white_pixels(img),
            imagefunctions.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(100, 200)),
            imagefunctions.mag_thresh(img, sobel_kernel=5, mag_thresh=(100, 180)),
            imagefunctions.dir_threshold(img, sobel_kernel=3, thresh=(np.pi/8, np.pi/4))]
    out = img[::2,::2,::2].ravel()
    return out
 

