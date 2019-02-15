import numpy as np
import cv2

########################################
# perspective transform function
def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[43,26],
         [38,58],
         [90,26],
         [96,58]]
    )
    dst = np.float32(
        [[31,23],
         [31,71],
         [95,23],
         [95,71]]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

#######################################
# pick points for interpolation
def pickpoints(img_bin):
    nz = np.nonzero(img_bin)
    pts_x = nz[1]
    pts_y = nz[0]
    return pts_x, pts_y


def pickpoints2(img_bin,minx=0,miny=0,maxx=999999,maxy=999999):
    nz = np.nonzero(img_bin)
    all_x = nz[1]
    all_y = nz[0]
    pts_x = []
    pts_y = []
    for x,y in zip(all_x,all_y):
        if (x>=minx and x<=maxx and y>=miny and y<=maxy):
            pts_x.append(x)
            pts_y.append(y)
    return pts_x, pts_y
#######################################
# feature extraction for SVM

# Pre-process
def preprocess_grayscale(X):

    ## Grayscale
    X_gray = np.dot(X[...][...,:3],[0.299,0.587,0.114])

    ## Histogram Equalization (Improve contrast)
    X_gray_eq = np.zeros(shape=X_gray.shape)
    for i in range(X_gray.shape[0]):
        img = cv2.equalizeHist(X_gray[i].squeeze().astype(np.uint8))
        X_gray_eq[i] = img

    ## scale to [0,1]
    X_gray_eq_scale = np.divide(X_gray_eq,255.0)

    ## expand to fit dimensions
    X_prep = np.expand_dims(X_gray_eq_scale, axis=3)

    return X_prep

def equalize_Y_channel(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

def preprocess_rgb(X):
    ## Histogram Equalization (Improve contrast)
    X_eq = np.zeros(shape=X.shape)
    for i in range(X.shape[0]):
        img = X[i].squeeze().astype(np.uint8)
        X_eq[i] = equalize_Y_channel(img)
    ## scale to [0,1]
    X_eq_scale = np.divide(X_eq,255.0)
    return X_eq_scale

def preprocess_one_rgb(img):
    img = np.array(img, dtype=np.uint8)
    img_eq = equalize_Y_channel(img)
    img_eq_scale = np.divide(img_eq,255.0)
    return img_eq_scale

# Functions for Features
def num_white_pixels(img):
    img = np.array(img, dtype=np.float32)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(gray, 0.8, 1., cv2.THRESH_BINARY)
    return np.sum(img_bin) #/(50*50)
def num_red_pixels(img):
    img = np.array(img, dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, img_bin = cv2.threshold(hsv[:, :, 1], .4, 1., cv2.THRESH_BINARY)
    #print 'red ' + str(np.sum(img_bin))
    return np.sum(img_bin) #/(50*50)
def num_corners(img):
    img = np.array(img, dtype=np.float32)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.1)
    return np.sum(dst>0.01*dst.max())
def num_edges(img):
    img = np.array(img, dtype=np.float32)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.uint8(255*gray)
    dst = cv2.Canny(gray,10,20)
    return np.sum(dst>0.01*dst.max())

# Sobel gradient in one direction and thresholding
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    img = 255*np.array(img, dtype=np.float32)
    thresh_min = thresh[0]
    thresh_max = thresh[1]

    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    res = np.sum(binary_output>0.01*binary_output.max())
    return res

# Magnitude of the gradient
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    img = 255*np.array(img, dtype=np.float32)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    res = np.sum(binary_output>0.01*binary_output.max())
    return res

# Direction of the Gradient
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    img = np.array(img, dtype=np.float32)
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    res = np.sum(binary_output>0.01*binary_output.max())
    return res

