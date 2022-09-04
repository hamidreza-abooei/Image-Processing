import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

#read sonography image 
sonography=cv.imread("sonography.jpg")
#convert to gray scale
sonography = cv.cvtColor(sonography,cv.COLOR_BGR2GRAY)
#show original image:
plt.subplot(2,3,1)
plt.imshow (sonography,cmap="gray")
plt.title("Original")
plt.axis(False)
#get soble filter

sonography_sobel_x = np.abs( cv.Sobel(sonography,cv.CV_16U,1,0,ksize=5))
sonography_sobel_y = np.abs( cv.Sobel(sonography,cv.CV_16U,0,1,ksize=5))
sonography_sobel = cv.addWeighted(sonography_sobel_x, 1, sonography_sobel_y, 1, 0.0)
#show sobel filtered image 
plt.subplot(2,3,2)
plt.imshow (sonography_sobel,cmap="gray")
plt.title("Sobel Filter")
plt.axis(False)

#calculate prewitt filter
prewittx_k = np.array([[1,1,1],[0,0,0],[1,1,1]])
prewittx = cv.filter2D(sonography, cv.CV_16U, prewittx_k)
prewitty_k = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitty = cv.filter2D(sonography, cv.CV_16U, prewitty_k)
prewitt = cv.addWeighted(prewittx, 1, prewitty, 1, 0.0)

#show prewitt filtered image 
plt.subplot(2,3,3)
plt.imshow (prewittx,cmap="gray")
plt.title("prewitt Filter")
plt.axis(False)

#calculate LoG filter
kernel = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,1,2],[0,1,2,1,0],[0,0,1,0,0]])
LoG = cv.filter2D(sonography, cv.CV_16U, kernel)

#show LoG filtered image 
plt.subplot(2,3,4)
plt.imshow (LoG,cmap="gray")
plt.title("LoG Filter")
plt.axis(False)

#calculate canny edge detection
canny = cv.Canny(sonography,0,220)

#show canny edge detection 
plt.subplot(2,3,5)
plt.imshow (canny,cmap="gray")
plt.title("canny Filter")
plt.axis(False)

#calculate roberts filter
kernelx = np.array([[-1,0],[0,1]])
kernely = np.array([[0,-1],[1,0]])
robertsx = cv.filter2D(sonography, cv.CV_16U, kernelx)
robertsy = cv.filter2D(sonography, cv.CV_16U, kernely)
roberts = cv.addWeighted(robertsx, 1, robertsy, 1, 0.0)

#show roberts filtered image 
plt.subplot(2,3,6)
plt.imshow (roberts,cmap="gray")
plt.title("Roberts Filter")
plt.axis(False)

plt.show()