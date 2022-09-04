import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

brain = cv.imread("images/brains.png",0)

def powerlow (image , gamma):
    #power law 
    image =  image ** gamma 
    #This is better than calculating c 
    image = image * 255 / image.max()
    #change type from floot16 to uint8 
    image = image.astype('uint8')
    return image 
# define subplots 
fig , axs = plt.subplots(nrows = 2 , ncols = 3 )
# make title
fig.suptitle("intensity based opertaions")
# apply function 
power = powerlow(brain,0.5) 
#show image
axs[0 , 0].imshow(power, cmap = 'gray' , vmin = 0 , vmax = 255)
#make title 
axs[0 , 0].set_title("gamma=0.5")
#turn off axiss
axs[0 , 0].set_axis_off() 

#show histogram
axs[1 , 0].hist(power.ravel(),bins=range(256),log=True)
axs[1 , 0].set_title("histogram")
#turn off fram 
axs[1 , 0].set_frame_on(False)
#turn off y axis 
axs[1 , 0].axes.get_yaxis().set_visible(False)

#log transform function 
def logtransformation(img , k):
    #log transform
    img = np.log (img+1 ) /np.log(k)
    #scale to 0 to 255
    img = img * 255 / img.max()
    #change to uint8
    img = img.astype('uint8')
    return img

#apply function 
logimg = logtransformation(brain,1000)
#show 
axs[0 , 1].imshow(logimg, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 , 1].set_title("log")
axs[0 , 1].set_axis_off() 
#show histogram
axs[1 , 1].hist(logimg.ravel(),bins=range(256) , log=True)
axs[1 , 1].set_title("histogram")
axs[1 , 1].set_frame_on(False)
axs[1 , 1].axes.get_yaxis().set_visible(False)


#show original image
axs[0 , 2].imshow(brain, cmap = 'gray' )
axs[0 , 2].set_title("Original")
axs[0 , 2].set_axis_off() 
#show original image histogram
axs[1 , 2].hist(brain.ravel(),bins=range(256),log=True )
axs[1 , 2].set_title("histogram")
axs[1 , 2].set_frame_on(False)
axs[1 , 2].axes.get_yaxis().set_visible(False)

plt.show()