import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

shade1 = cv.imread("images/shade1.tif",0)
shade2 = cv.imread("images/shade2.tif",0)

# cv.imshow("shade1",shade1)
# cv.imshow("shade2",shade2)
# print(shade1)
# print(shade2)
mean1=np.mean(shade1)
mean2=np.mean(shade2)
s1=0
s2=0
shape1 = np.shape(shade1)
for i in range(shape1[0]):
    for j in range(shape1[1]):
        s1 += abs(shade1[i][j]-mean1) 
        #we know that their shape is equal
        s2 += abs(shade2[i][j]-mean2)
print(s1/shape1[0]/shape1[1])
print(s2/shape1[0]/shape1[1])

plt.subplot(121)
plt.imshow(shade1,cmap='gray' ,vmin=0 , vmax=7)
plt.title("shade1")
plt.subplot(122)
plt.imshow(shade2,cmap='gray' ,vmin=0 , vmax=7)
plt.title("shade2")
plt.show()





#### Question 1
def absdev(img):
    mean=np.mean(img)
    shape = np.shape(img)
    s = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            s += abs(img[i][j]-mean) 
    s = s / (shape[0]/shape[1])
    return s


brain = cv.imread("images/brains.png",0)

gamma=0.1

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
contrast = absdev(brain) #gamma2 = 1
gammamax = 0

print(contrast)
for gamma in np.linspace(0,1,num = 11 ):
    power = powerlow(brain,gamma)
    cv.imshow(str(gamma),power)
    contrast1 = absdev(power)
    print(gamma , contrast1)
    if (contrast1 > contrast):
        print(contrast1)
        contrast = contrast1
        gammamax = gamma 
        print(gamma , contrast)

power = powerlow(brain , gammamax)
#show image
axs[0 , 0].imshow(power, cmap = 'gray' , vmin = 0 , vmax = 255)
#make title 
axs[0 , 0].set_title("gamma = " + str(gammamax))
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


cv.waitKey(0)
cv.destroyAllWindows()