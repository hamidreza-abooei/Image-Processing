import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#import image
image = cv.imread("images/chest.tif",0)
#get descrite fourier transform of image
f = cv.dft(np.float32(image),flags = cv.DFT_COMPLEX_OUTPUT)
#shift it to show better 
fshift = np.fft.fftshift(f)

# calculate amp and phase of figure by transfering the fourier from cartesian to polar
amp , phi = cv.cartToPolar(fshift[:,:,0], fshift[:,:,1])
#and this is log to show all range of it
magnitude_spectrum = 20*np.log(np.abs(amp))

#show amp
plt.figure()
plt.subplot(2,2,1)
plt.imshow(magnitude_spectrum,cmap = 'gray')
plt.title("Magnitude spectrum")
plt.axis(False)
#show phase
plt.subplot(2,2,2)
plt.imshow(phi,cmap = 'gray')
plt.title("Phase")
plt.axis(False)

#return the shift that we had made to see amp and phase in the center
ishift = np.fft.ifftshift(fshift)
#get inverse descrite fourier transform
ifourier = cv.idft(ishift)
#convert 3d to 2d image
ifourier = cv.magnitude(ifourier[:,:,0],ifourier[:,:,1])
#show recoustructed image
plt.subplot(2,2,3)
plt.imshow(ifourier,cmap='gray')
plt.title("reconstructed")
plt.axis(False)

#cart is an auxiliary variable to correct polarToCart output shape
cart=np.zeros(np.shape(fshift))
#turn image from polar to cartesian coordination with negative phase to rotate
cartrev = cv.polarToCart(amp,-phi)
cart[:,:,0] = cartrev[0][:][:]
cart[:,:,1] = cartrev[1][:][:]
#return the shift that we had made to see amp and phase in the center
ishiftrev = np.fft.ifftshift(cart)
#get inverse descrite fourier transform
ifourierrev = cv.idft(ishiftrev)
#convert 3d to 2d image
ifourierrev = cv.magnitude(ifourierrev[:,:,0],ifourierrev[:,:,1])
#show mirrored image ref: center (rotate 180deg)
plt.subplot(2,2,4)
plt.imshow(ifourierrev,cmap='gray')
plt.title("mirror")
plt.axis(False)
plt.show()

