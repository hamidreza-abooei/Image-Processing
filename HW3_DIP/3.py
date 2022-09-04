import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
#import images
mandrill =cv.imread("images/mandrill.tif" , 0)
clown = cv.imread("images/clown.tif",0)

#get descrite fourier transform of mandrill
fm = cv.dft(np.float32(mandrill),flags = cv.DFT_COMPLEX_OUTPUT)
#get amp and phase of mandrill
ampm , phim = cv.cartToPolar(fm[:,:,0], fm[:,:,1])

#get descrite fourier transform of clown
fc = cv.dft(np.float32(clown),flags = cv.DFT_COMPLEX_OUTPUT)
#get amp and phase of clown
ampc , phic = cv.cartToPolar(fc[:,:,0], fc[:,:,1])

cart1=np.zeros(np.shape(fm))
#convert polar to cartesian coordiantion phase mandrill amp clown
cartrev1 = cv.polarToCart(ampc,phim)
cart1[:,:,0] = cartrev1[0][:][:]
cart1[:,:,1] = cartrev1[1][:][:]
#get inverse fourier  
ifourierrev1 = cv.idft(cart1)
#3d to 2d image
ifourierrev1 = cv.magnitude(ifourierrev1[:,:,0],ifourierrev1[:,:,1])

cart2=np.zeros(np.shape(fc))
#convert polar to cartesian coordiantion phase clown amp mandrill
cartrev2 = cv.polarToCart(ampm,phic)
cart2[:,:,0] = cartrev2[0][:][:]
cart2[:,:,1] = cartrev2[1][:][:]
#get inverse fourier 
ifourierrev2 = cv.idft(cart2)
#3d to 2d image
ifourierrev2 = cv.magnitude(ifourierrev2[:,:,0],ifourierrev2[:,:,1])

#show mandrill
plt.subplot(2,2,1)
plt.imshow(mandrill,cmap='gray')
plt.title("mandrill")
plt.axis(False)
#show clown
plt.subplot(2,2,2)
plt.imshow(clown,cmap='gray')
plt.title("clown")
plt.axis(False)
#show phase mandrill amp clown
plt.subplot(2,2,3)
plt.imshow(ifourierrev1,cmap='gray')
plt.title("phase mandrill amp clown")
plt.axis(False)
#show phase clown amp mandrill
plt.subplot(2,2,4)
plt.imshow(ifourierrev2,cmap='gray')
plt.title("phase clown amp mandrill")
plt.axis(False)
plt.show()

