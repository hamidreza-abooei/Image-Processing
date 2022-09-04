import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
#def function consider that D0 is used in Butterworth filter and it is pre defined as 2 and we can get other values through input
def filterfunction( image , filtertype , radius , D0 = 2):
    #zero padding
    image = cv.copyMakeBorder(image ,0 , np.shape(image)[0] , 0 , np.shape(image)[0] , 0)
    #get shape of image
    shapex , shapey = np.shape(image)
    #get dft
    f = cv.dft(np.float32(image),flags = cv.DFT_COMPLEX_OUTPUT)
    #shift it to center
    fshift = np.fft.fftshift(f)
    #get amp and phase
    amp , phi = cv.cartToPolar(fshift[:,:,0], fshift[:,:,1])
    
    #define ideal Low Pass filter
    if (filtertype == "idealLP"):
        ffilter = np.zeros((shapex,shapey))
        for i in range(shapex):
            for j in range(shapey):
                if ((i-shapex/2)**2 + (j-shapey/2)**2)**0.5 < radius:
                    ffilter[i,j] = 1   
    
    #define ideal High Pass filter
    if (filtertype == "idealHP"):
        ffilter = np.ones((shapex,shapey))
        for i in range(shapex):
            for j in range(shapey):
                if ((i-shapex/2)**2 + (j-shapey/2)**2)**0.5 < radius:
                    ffilter[i,j] = 0
        
    #define Butterwirth Low Pass filter
    if (filtertype == "butterworthLP"):
        ffilter = np.zeros((shapex,shapey))
        for i in range(shapex):
            for j in range(shapey):
                ffilter[i,j] = 1/(1+((((i-shapex/2)**2 + (j-shapey/2)**2)**0.5) / radius ) ** (2*D0) )

    #define Butterworth High Pass filter = 1 - Low Pass
    if (filtertype == "butterworthHP"):
        ffilter = np.zeros((shapex,shapey))
        for i in range(shapex):
            for j in range(shapey):
                ffilter[i,j] = 1/(1+((((i-shapex/2)**2 + (j-shapey/2)**2)**0.5) / radius ) ** (2*D0) )
        ffilter = 1 - ffilter

    #define Gaussian Low Pass filter
    if (filtertype == "GaussianLP"):
        ffilter = np.zeros((shapex,shapey))
        for i in range(shapex):
            for j in range(shapey):
                ffilter[i,j] = np.exp(-((i-shapex/2)**2 + (j-shapey/2)**2)/(2*(radius**2)))

    #define Gaussian High Pass filter = 1 - Low Pass
    if (filtertype == "GaussianHP"):
        ffilter = np.zeros((shapex,shapey))
        for i in range(shapex):
            for j in range(shapey):
                ffilter[i,j] = np.exp(-((i-shapex/2)**2 + (j-shapey/2)**2)/(2*(radius**2)))
        ffilter = 1 - ffilter

    #filter image
    amp = amp * ffilter
    #correct type from foat64 to float32
    amp = amp.astype(np.float32)
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
    #crop image
    ifourierrev = ifourierrev[0:int(shapex/2),0:int(shapey/2)]
    #return filtered image
    return ifourierrev


img=cv.imread("images/a.tif",0)
#Apply and show Ideal Low pass filter with radius = 50,100,200 
plt.subplot(3,6,1)
filterimage = filterfunction(img,"idealLP",50)
plt.imshow(filterimage,cmap='gray')
plt.title("ideal 50R  LP")
plt.axis(False)
plt.subplot(3,6,2)
filterimage = filterfunction(img,"idealLP",100)
plt.imshow(filterimage,cmap='gray')
plt.title("ideal 100R LP")
plt.axis(False)
plt.subplot(3,6,3)
filterimage = filterfunction(img,"idealLP",200)
plt.imshow(filterimage,cmap='gray')
plt.title("ideal 200R LP")
plt.axis(False)

#Apply and show Ideal High pass filter with radius = 50,100,200 
plt.subplot(3,6,4)
filterimage = filterfunction(img,"idealHP",50)
plt.imshow(filterimage,cmap='gray')
plt.title("ideal 50R HP")
plt.axis(False)
plt.subplot(3,6,5)
filterimage = filterfunction(img,"idealHP",100)
plt.imshow(filterimage,cmap='gray')
plt.title("ideal 100R HP")
plt.axis(False)
plt.subplot(3,6,6)
filterimage = filterfunction(img,"idealHP",200)
plt.imshow(filterimage,cmap='gray')
plt.title("ideal 200R HP")
plt.axis(False)

#Apply and show Butterworth Low pass filter with radius = 50,100,200 D0=2
plt.subplot(3,6,7)
filterimage = filterfunction(img,"butterworthLP",50 , 2)
plt.imshow(filterimage,cmap='gray')
plt.title("Butterworth 50R LP")
plt.axis(False)
plt.subplot(3,6,8)
filterimage = filterfunction(img,"butterworthLP",100 ,2)
plt.imshow(filterimage,cmap='gray')
plt.title("Butterworth 100R LP")
plt.axis(False)
plt.subplot(3,6,9)
filterimage = filterfunction(img,"butterworthLP",200 , 2)
plt.imshow(filterimage,cmap='gray')
plt.title("Butterworth 200R LP")
plt.axis(False)

#Apply and show Butterworth High pass filter with radius = 50,100,200 D0 = 2 
plt.subplot(3,6,10)
filterimage = filterfunction(img,"butterworthHP",50 , 2)
plt.imshow(filterimage,cmap='gray')
plt.title("Butterworth 50R HP")
plt.axis(False)
plt.subplot(3,6,11)
filterimage = filterfunction(img,"butterworthHP",100 , 2)
plt.imshow(filterimage,cmap='gray')
plt.title("Butterworth 100R HP")
plt.axis(False)
plt.subplot(3,6,12)
filterimage = filterfunction(img,"butterworthHP",200 , 2)
plt.imshow(filterimage,cmap='gray')
plt.title("Butterworth 200R HP")
plt.axis(False)

#Apply and show Gaussian Low pass filter with radius = 50,100,200 
plt.subplot(3,6,13)
filterimage = filterfunction(img,"GaussianLP",50 )
plt.imshow(filterimage,cmap='gray')
plt.title("Gaussian 50R LP")
plt.axis(False)
plt.subplot(3,6,14)
filterimage = filterfunction(img,"GaussianLP",100 )
plt.imshow(filterimage,cmap='gray')
plt.title("Gaussian 100R LP")
plt.axis(False)
plt.subplot(3,6,15)
filterimage = filterfunction(img,"GaussianLP",200 )
plt.imshow(filterimage,cmap='gray')
plt.title("Gaussian 200R LP")
plt.axis(False)

#Apply and show Gaussian High pass filter with radius = 50,100,200 
plt.subplot(3,6,16)
filterimage = filterfunction(img,"GaussianHP",50 )
plt.imshow(filterimage,cmap='gray')
plt.title("Gaussian 50R HP")
plt.axis(False)
plt.subplot(3,6,17)
filterimage = filterfunction(img,"GaussianHP",100 )
plt.imshow(filterimage,cmap='gray')
plt.title("Gaussian 100R HP")
plt.axis(False)
plt.subplot(3,6,18)
filterimage = filterfunction(img,"GaussianHP",200 )
plt.imshow(filterimage,cmap='gray')
plt.title("Gaussian 200R HP")
plt.axis(False)

plt.show()