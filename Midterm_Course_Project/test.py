import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

# img = cv.imread("CT_1.tif")
# cv.imshow("",img)
# cv.waitKey(0)


# cv.imwrite("CT_1.png",img)

# img = cv.imread("CT_2.tif")

# cv.imwrite("CT_2.png",img)


# a = [1,2,3,4,5,6,7,8,9,10,11,12]
# b = [8,7,6,5,4,3,2,1,1,1,1,1]

# print(np.subtract(a-b)) 
retina = cv.imread("retina.jpg",0)
retina_sub = cv.imread("retina_sub.jpg",0)

# plt.imshow(retina)
# plt.show()
hist , bins = np.histogram(retina[2080:565+2080,880:565+880].ravel(), bins=64)
hist_sub , bins = np.histogram(retina_sub.ravel(), bins=64)
hist2 , bins = np.histogram(retina[1850:565+1850,800:565+800].ravel(), bins=64)
# print(hist)
# print(hist_sub)
# print(hist-hist_sub)
plt.subplot(3,2,1)
plt.plot(hist)
plt.subplot(3,2,2)
plt.plot(hist_sub)
plt.subplot(3,2,3)
plt.hist(retina[2080:565+2080,880:565+880].ravel(),bins=64)
plt.subplot(3,2,4)
plt.hist(retina[1850:565+1850,800:565+800].ravel(),bins=64)
plt.subplot(3,2,5)
plt.plot(np.abs(hist_sub-hist) )
plt.subplot(3,2,6)
plt.plot( np.abs(hist_sub-hist2))
print(np.sum(np.abs(hist_sub-hist)))
print(np.sum(np.abs(hist_sub-hist2)))
plt.show()

