import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
#read image
rectangle = cv.imread("noisy_rectangle.png",0)
#use ELLIPSE kernel 15x15
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
#apply erosion
erosion= cv.erode(rectangle,kernel,iterations = 1)
#show irosion
plt.imshow(erosion , cmap = 'gray', vmin = 0 , vmax = 255)
#add title
plt.title("Erosion")
plt.axis(False)
plt.show()
# apply dilation 
dilation = cv.dilate(rectangle,kernel,iterations = 1)
plt.imshow(dilation,cmap = 'gray', vmin = 0 , vmax = 255)
plt.title("Dilation")
plt.axis(False)
plt.show()
#use a rect kernel with minimum size  
kernel = cv.getStructuringElement(cv.MORPH_RECT,(29,52))
#apply opening to remove outer noises
opening = cv.morphologyEx(rectangle,cv.MORPH_OPEN,kernel)
# show image
plt.imshow(opening,cmap = 'gray', vmin = 0 , vmax = 255)
plt.title("Opening")
plt.axis(False)
plt.show()

#use a rect kernel with minimum size
kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,40))
#apply closing to remove holes
closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel)
plt.imshow(closing,cmap = 'gray', vmin = 0 , vmax = 255)
plt.title("Closing of opening")
plt.axis(False)
plt.show()