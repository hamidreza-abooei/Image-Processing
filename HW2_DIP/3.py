import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#function for getting normalized intensity 
def normalizedf(img):
    #define histogram empty
    out = np.zeros(256,dtype= "int")
    #gettng shape 
    shape = np.shape(img)
    #calculating histogram 
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[img[i,j]] += 1 
    #normalize histogram 
    out = out / (shape[0] * shape[1] )
    return out

def imagenormalized(img):
    #image shape
    shape = img.shape
    #getting normalized histogram from previous function
    p = normalizedf(img)
    #define out vector
    out = np.zeros(256)
    s=0
    #calculating sk from p
    for k in range(256):
        for j in range(k+1):
            s += p[j]
        out[k] = s * 256
        s=0
    #change out type to uint8
    out = out.astype('uint8')
    #apply results to image
    for i in range(shape[0]):
        for j in range (shape[1]):
            img[i][j] = out[img[i][j]]
    return img

#read dark image 
img=cv.imread('images/dark.tif' , 0)
#define subtitles
fig , axs = plt.subplots(nrows = 2 , ncols = 2 )
#make title 
fig.suptitle("Histogram equalization")
#show image
axs[0 , 0].imshow(img, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 , 0].set_title("Original Dark")
axs[0 , 0].set_axis_off() 
#show histogram
axs[1 , 0].hist(img.ravel(),bins=range(256))
axs[1 , 0].set_title("histogram")
axs[1 , 0].set_frame_on(False)
axs[1 , 0].axes.get_yaxis().set_visible(False)
#apply function 
normalized = imagenormalized(img)
axs[0 , 1].imshow(normalized, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 , 1].set_title("normalized")
axs[0 , 1].set_axis_off() 
#show histogram
axs[1 , 1].hist(normalized.ravel(),bins=range(256))
axs[1 , 1].set_title("histogram")
axs[1 , 1].set_frame_on(False)
axs[1 , 1].axes.get_yaxis().set_visible(False)

plt.show()

####

#read bright image
img = cv.imread("images/bright.tif")
fig2 , axs = plt.subplots(nrows = 2 , ncols = 2 )
fig2.suptitle("Histogram equalization")

axs[0 , 0].imshow(img, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 , 0].set_title("Original Bright")
axs[0 , 0].set_axis_off() 

axs[1 , 0].hist(img.ravel(),bins=range(256))
axs[1 , 0].set_title("histogram")
axs[1 , 0].set_frame_on(False)
axs[1 , 0].axes.get_yaxis().set_visible(False)

#apply function 
normalized = imagenormalized(img)
axs[0 , 1].imshow(normalized, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 , 1].set_title("normalized")
axs[0 , 1].set_axis_off() 

axs[1 , 1].hist(normalized.ravel(),bins=range(256))
axs[1 , 1].set_title("histogram")
axs[1 , 1].set_frame_on(False)
axs[1 , 1].axes.get_yaxis().set_visible(False)

plt.show()

####
#read low contrast image
img = cv.imread("images/Lowcontrast.tif")
fig2 , axs = plt.subplots(nrows = 2 , ncols = 2 )
fig2.suptitle("Histogram equalization ")

axs[0 , 0].imshow(img, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 , 0].set_title("Original Low contrast")
axs[0 , 0].set_axis_off() 

axs[1 , 0].hist(img.ravel(),bins=range(256))
axs[1 , 0].set_title("histogram")
axs[1 , 0].set_frame_on(False)
axs[1 , 0].axes.get_yaxis().set_visible(False)

#apply function 
normalized = imagenormalized(img)
axs[0 , 1].imshow(normalized, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 , 1].set_title("normalized")
axs[0 , 1].set_axis_off() 

axs[1 , 1].hist(normalized.ravel(),bins=range(256))
axs[1 , 1].set_title("histogram")
axs[1 , 1].set_frame_on(False)
axs[1 , 1].axes.get_yaxis().set_visible(False)

plt.show()