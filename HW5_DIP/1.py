# import libs
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

# import images
mrif = cv.imread("MRIF.png",0)
mris = cv.imread("MRIS.png",0)
# define points that user enter :
points = []

# affine transform with entered points
def affinetransform(points):
    points = np.float32(points)
    # find transform matrix a11 -a23
    affine = cv.getAffineTransform(points[0:3],points[3:6])
    # get shape of mris
    shape=np.shape(mris)
    # apply transformed matrix to image 
    transformed_image = cv.warpAffine(mrif,affine,(shape[1],shape[0]))
    #show transformed image :
    ax.imshow(transformed_image,cmap='gray',vmax=255,vmin=0)
    ax.axis(False)
    # add title
    plt.title("Transformed image with entered points")

# function to change pic from mrif to mris
def changepic():
    ax.clear()
    ax.imshow(mris,cmap='gray',vmax=255,vmin=0)
    ax.axis(False)

# function to read points
def onclick(event):
    # read point
    x , y = (event.xdata) , (event.ydata)
    # add new point to points
    points.append([x,y])
    # show selected point
    ax.scatter(x,y,color='blue')
    # change pic if 3 points choosed
    if (len(points)==3):
        changepic()
    # after getting 3 more points go and find transform matrix
    if (len(points)==6):
        ax.clear()
        affinetransform(points)
    plt.show()

# show first image 
fig,ax = plt.subplots()
ax.imshow(mrif,cmap='gray',vmax=255,vmin=0)
ax.axis(False)

# get butten from clicking the mouse 
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
