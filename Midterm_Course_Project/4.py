import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 

#read image
image = cv.imread("xray_checkered.png",0)
#fourier transform 
f = cv.dft(np.float32(image),flags = cv.DFT_COMPLEX_OUTPUT)
#shift it to centering 
fshift = np.fft.fftshift(f)
# calculate amp and phase of figure by transfering the fourier from cartesian to polar
amp , phi = cv.cartToPolar(fshift[:,:,0], fshift[:,:,1])

#get the shape of amp
row , col = np.shape(amp)

#find four maximum point 
for number in range(4):
    #pre define maximum intensity in fourier amp
    maxi=0
    #loop for finding the maximum intensity except the center of the amp 
    for i in range (row):
        for j in range (col):
            #in this if we ignore center of amp for calculating maximum value
            if (((i-(row/2))**2 + (j-(col/2))**2)**0.5 > 20):
                #find maximum value
                if amp[i,j] > maxi:
                    maxi = amp[i,j]
                    maxx,maxy = i,j
    #substitute amount of maximum point with 4 neigherhood 
    amp[maxx,maxy] = (amp[maxx+1,maxy] + amp[maxx,maxy+1] + amp[maxx-1,maxy] + amp[maxx,maxy-1] )/4



#cart is an auxiliary variable to correct polarToCart output shape
cart=np.zeros(np.shape(fshift))
#turn image from polar to cartesian coordination 
cartrev = cv.polarToCart(amp,phi)
cart[:,:,0] = cartrev[0][:][:]
cart[:,:,1] = cartrev[1][:][:]
#return the shift that we had made to see amp and phase in the center
ishiftrev = np.fft.ifftshift(cart)
#get inverse descrite fourier transform
ifourierrev = cv.idft(ishiftrev)
#convert 3d to 2d image
ifourierrev = cv.magnitude(ifourierrev[:,:,0],ifourierrev[:,:,1])
#show noise reduced image
plt.figure()
plt.imshow(ifourierrev,cmap='gray')
plt.title("Frequency operation applied")
plt.axis(False)
plt.show()
