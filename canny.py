import numpy as np
import cv2


def Gaussian_kernel(size,sigma = 1):
    # Noise Reduction Using Gaussian Kernel
    kernel_size = size   # 5*5 kernel, can be tuned
    kernel_size = int(kernel_size) // 2
    
    x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal    

    return kernel



def sobel_smooth(image):
    # Gradient Calculation Using Sobel
    dx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]], np.float32)
    dy = np.array([[1, 2, 1], 
                   [0, 0, 0], 
                   [-1, -2, -1]], np.float32)
        
    Ix = cv2.filter2D(image, -1, dx)
    Ix = cv2.convertScaleAbs(Ix)
    Iy = cv2.filter2D(image, -1, dy)
    Iy = cv2.convertScaleAbs(Iy)
        
    sobel_img = np.hypot(Ix, Iy).astype(np.uint8)
    sobel_img = cv2.convertScaleAbs(sobel_img)
    theta = np.arctan2(Iy, Ix)
    
    return sobel_img,theta


def noMaxSuppression(sobel_img,theta):
    # Non - Maximum Suppression
    M, N = sobel_img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
           
            q = 255
            r = 255

            #angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = sobel_img[i, j+1]
                r = sobel_img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = sobel_img[i+1, j-1]
                r = sobel_img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = sobel_img[i+1, j]
                r = sobel_img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = sobel_img[i-1, j-1]
                r = sobel_img[i+1, j+1]

            if (sobel_img[i, j] >= q) and (sobel_img[i, j] >= r):
                Z[i, j] = sobel_img[i, j]
            else:
                Z[i, j] = 0
                
    return Z


def Thresholding(image):
    # Double Thresholding
    highThresholdRatio = 0.15     # highThresholdRatio , lowThresholdRatio can be tuned
    lowThresholdRatio = 0.05
    highThreshold = image.max() * highThresholdRatio;
    lowThreshold = image.max()  * lowThresholdRatio;
        
    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)
        
    weak = np.int32(25)          # strong and weak intensity values can be tuned
    strong = np.int32(255)
        
    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)
        
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
        
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res,weak,strong


def hystoris(image,weak,strong):
    # Hysteresis
    M, N = image.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i,j] == weak):
                if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong) 
                or (image[i, j-1] == strong) or (image[i, j+1] == strong) or (image[i-1, j-1] == strong) 
                or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image



# Input Image
img = cv2.imread('tiger2.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Input Image', img)



kernel = Gaussian_kernel(5)

gaussian_smoothed_image = cv2.filter2D(img, -1, kernel)
gaussian_smoothed_image = cv2.convertScaleAbs(gaussian_smoothed_image)
cv2.imshow('Gaussian Smoothed Image',gaussian_smoothed_image)



sobel_img,theta = sobel_smooth(gaussian_smoothed_image)
cv2.imshow('Sobel Output',sobel_img)




noMaximg = noMaxSuppression(sobel_img, theta)
noMaximg = cv2.convertScaleAbs(noMaximg)
cv2.imshow('Non-Maximum Suppression',noMaximg)




thresholdimg,w,s = Thresholding(noMaximg)
thresholdimg = cv2.convertScaleAbs(thresholdimg)
cv2.imshow('After Double thresholding', thresholdimg)



output = hystoris(thresholdimg, w, s)

cv2.imshow('Final Output (After Hysteresis)',output)


cv2.waitKey(0)
cv2.destroyAllWindows()