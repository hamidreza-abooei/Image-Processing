import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv

#read image
img = cv.imread("reflections.jpg",0)
#convert to binary image 
_,img = cv.threshold(img,127,1,cv.THRESH_BINARY)

def hole_filling(x,y):
    #define img as global variable
    global img
    # if user clicks on dark point:
    if img[x,y] == 0:
        #find img compliment
        img_c = np.ones(np.shape(img)) - img
        #define a dark image inorder to 
        black = np.zeros(np.shape(img))
        #define clicked image white:
        black[x,y] = 1
        #this variable will difine whether we finished hole filling 
        end_finder = 0
        #start loop for define whole hole section
        while (np.sum(black) > end_finder):
            # count the number of points that are detected yet
            end_finder = np.sum(black)
            # define a cross for dilate kernel
            kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
            # dilate section with + kernel
            black = cv.dilate(black,kernel)
            # "and" operation between dilated section with inverted img
            black = cv.bitwise_and(black,img_c)
        # "or" operation between img and detected image 
        img = img + black
        # update image 
        plt.imshow(img,cmap='gray',vmax=1,vmin=0)
        plt.show()
    
# function to read points
def onclick(event):
    #read x and y of clicked points:
    x , y = (event.xdata) , (event.ydata)
    #call hole filling function 
    hole_filling(int(y),int(x))

#function to recognize pressing esc key
def esc(event):
    # esc recognition 
    if event.key == "escape":
        #close the figure 
        plt.close()

#create figure
fig,ax = plt.subplots()
#show image
ax.imshow(img,cmap='gray',vmax=1,vmin=0)
plt.axis(False)
#active sensitivity to determine which event occured 
fig.canvas.mpl_connect('key_press_event', esc)
#ready to get mouse click position 
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()