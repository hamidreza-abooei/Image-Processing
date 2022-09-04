import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#read images
ct1 = cv.imread("CT_1.tif" , 0 )
ct2 = cv.imread("CT_2.tif" , 0 )
#apply transform function and then round it and then convert to uint8
def transformation_function(img ):
    return (np.round((255 * np.sin(np.pi /2 / 255 * img)))).astype('uint8')

#apply TF to CT1 and CT2
ct1t = transformation_function(ct1)
ct2t = transformation_function(ct2)


#open a new figure to show
plt.figure()
#choose first subplot
plt.subplot(2,2,1)
#show original CT 1
plt.imshow(ct1,cmap = 'gray', vmin = 0 , vmax=255)
#add title
plt.title("Original CT1")
#turn axis off
plt.axis(False)

#chose second subplot to show
plt.subplot(2,2,3)
plt.imshow(ct1t,cmap = 'gray', vmin = 0 , vmax=255)
plt.title("Transformed CT1")
plt.axis(False)

#choose third subplot to show
plt.subplot(2,2,2)
plt.imshow(ct2,cmap = 'gray', vmin = 0 , vmax=255)
plt.title("Original CT2")
plt.axis(False)

#choose fourth subplot to show
plt.subplot(2,2,4)
plt.imshow(ct2t,cmap = 'gray', vmin = 0 , vmax=255)
plt.title("Transformed CT2")
plt.axis(False)

#show 
plt.show()

# define x axis (L, intensity levels)
x = np.arange(256)
#plot transformation function with red color
plt.plot(x,(255 * np.sin(np.pi /2 / 255 * x)),'r',label='s(r)')
#plot identity function with blue color
plt.plot(x,x,'b',label='Identity')
#show legend
plt.legend()
#show
plt.show()

