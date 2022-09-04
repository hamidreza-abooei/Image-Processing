import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 

def n_range(img):
    '''input: 2D image, output: uint8 2D image in range [0, 255]'''
    mi, ma = img.min(), img.max()
    return np.round(255*(img.astype(float) - mi)/(ma-mi)).astype('uint8')
#read fingerprint image
fingerprint = cv.imread("fingerprint.png",0)
#define rectangle kernel 
kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
#apply opening to remove noises
fingerfilter = cv.morphologyEx(fingerprint,cv.MORPH_OPEN,kernel)
#apply closing for connecting the lines
fingerdilation = cv.morphologyEx(fingerfilter,cv.MORPH_CLOSE,kernel)
#show 
plt.imshow(fingerdilation,cmap="gray")
plt.title("opening and closing")
plt.axis(False)
plt.show()
#read headCT
head = cv.imread("headCT.png",0)
#define a 3x3 kernel 
kernel = np.ones((3,3),np.uint8)
#apply 3x3 gradient filter
headgradient3 = cv.morphologyEx( head , cv.MORPH_GRADIENT , kernel )
#define a 7x7 kernel
kernel = np.ones((7,7),np.uint8)
#apply 7x7 gradient filter 
headgradient7 = cv.morphologyEx( head , cv.MORPH_GRADIENT , kernel )
#show 3x3 gradient
plt.imshow(headgradient3,cmap='gray')
plt.title("kernel 3x3 gradient")
plt.axis(False)
plt.show()
#show 7x7 gradient
plt.imshow(headgradient7,cmap='gray')
plt.title("kernel 7x7 gradeint")
plt.axis(False)
plt.show()
#read rice
rice = cv.imread("rice.tif")
#defiene a 100x100 ellipse for detecting the backgraound colormap
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(100,100))
#erode image
bg = cv.erode(rice,kernel , iterations=1)
#remove the gradient of background
equalizedbg = rice -bg
#turn to 0-255
equalizedbg = n_range(equalizedbg)
#thereshholding
out1,thr = cv.threshold(equalizedbg,100,255,cv.THRESH_BINARY)
#show
plt.imshow(thr,cmap='gray')
plt.title("rices")
plt.axis(False)
plt.show()
