from tools import *
import cv2 as cv
import numpy as np
from skimage.restoration import wiener 
import matplotlib.pyplot as plt

def findsimilarity(img1,img2):
    return np.sum((np.subtract(img1,img2)**2))

#read retina mution blured
retina = cv.imread("retina_motionblurred.jpg",0)
#read original retina
oretina = cv.imread("Retina.jpg",0)
#define kernel
kernel = np.eye(13) 
kernel = kernel/np.sum(kernel)
#finding best balance number
bestbalance = 0
#set a big number
maxsimilarity = 1000000000
#start searching the best balance
for balance in np.linspace(0,0.5,10):
    #apply wiener filter 
    restored = wiener(normal(retina),kernel,balance)
    #turn to uint8
    restored = n_range(restored)
    #find similarity with original retina image 
    similarity=findsimilarity(oretina,restored)
    if similarity<maxsimilarity:
        maxsimilarity = similarity
        bestbalance=balance
# this is a retry for more accuracy and more speed
for balance in np.linspace(bestbalance-0.05,bestbalance+0.05,10):
    restored = wiener(normal(retina),kernel,balance)
    restored = n_range(restored)
    similarity=findsimilarity(oretina,restored)
    if similarity<maxsimilarity:
        maxsimilarity = similarity
        bestbalance=balance

#apply wiener filter with best balance
restored = wiener(normal(retina),kernel,bestbalance)
#turn back to uint8
restored = n_range(restored)

#add a figure
plt.figure()
#subplot 1 of 2x2
plt.subplot(2,2,1)
#show motionblured image
plt.imshow(retina,cmap='gray')
plt.axis(False)
plt.title("Retina")

#subplot 1 of 2x2
plt.subplot(2,2,2)
#show restored image 
plt.imshow(restored,cmap='gray')
plt.axis(False)
plt.title("Restored with " + str(bestbalance)+" balance")

#subplot 1 of 2x2
plt.subplot(2,2,3)
#show fourier transform of motionblured image
plt.imshow(logmagnitude(retina),cmap='gray')
plt.axis(False)
plt.title("Logaritm amp of retina")

#subplot 1 of 2x2
plt.subplot(2,2,4)
#show fourier transform of restored image
plt.imshow(logmagnitude(restored),cmap='gray')
plt.axis(False)
plt.title("Logaritm amp of restored")

#show
plt.show()