import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv

img = cv.imread("fMRI.jpg",0)
# determine the threshold as mentioned in question 
threshold = 11
#determine that we want to use 4 or 8 nighberhood
neighberhood_4 = True #or False for 8 neighberhood
#this flag is for just segment once
flag = True

#function for region growing
def region_growing(x,y):
    #set flag as global variable
    global flag
    #change flag to false to avoid repetition because of mouse more clicks
    flag = False
    #import img as global variable
    global img
    #predefine background
    black = np.zeros(np.shape(img), dtype = "uint8")
    #this variable is for finding end of segementing
    end_finder = 0
    #SEED
    black[x,y] = 1
    #define kernel as cross (4 neighber) or rect (8 neighber)
    if neighberhood_4:
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    #start repeating and segmenting
    while(np.sum(black) > end_finder):
        #count the number of segmented points
        end_finder = np.sum(black)
        #define a black image for new points that are in the range 
        next_black = np.zeros(np.shape(img),dtype='uint8')
        #dilate black segmentation to reach new neighberhood points
        new_section = cv.dilate(black,kernel)
        #this section is for geting the mean of segmented points in order to find indicator
        black_img = cv.bitwise_and(img,img , mask=black)
        #new points obtained by subtracting dilated section with section
        new_points  = new_section - black
        #find indicator 
        indicator = np.sum(black_img) / end_finder
        # find pixels which are new points
        new_points_img = cv.bitwise_and(img,img,mask=new_points)
        #I subtracted finded pixels with indicator and get absolout of it
        diff_seed = np.abs ( new_points_img - indicator )
        # thresholding from subtracted with threshold level. this return us the invert of what we want
        _,inv_thresholding = cv.threshold(diff_seed,threshold,1,cv.THRESH_BINARY )
        #convert to uint8
        inv_thresholding = inv_thresholding.astype("uint8")
        # invert the invert thresholded 
        thresholding = np.ones(np.shape(img),dtype = "uint8") - inv_thresholding
        # find new points that are in the segmentation
        next_black = cv.bitwise_and( new_points , new_points , mask=thresholding )
        # next step segmentation done 
        black = black + next_black

    #show segmented result
    ax.imshow(black,cmap = "gray", vmax=1,vmin = 0)
    #add title according to 4 or 8 neighberhood
    if neighberhood_4:
        plt.title("segmented by region based growing with cross 3x3 kernel")
        #save fig as out4.png
        plt.savefig("out4.png")
    else:
        plt.title("segmented by region based growing with rectangle 3x3 kernel")
        #save fig as out8.png
        plt.savefig("out8.png")
    
    plt.show()
    
def onclick(event):
    global flag
    #read x and y of clicked points:
    x , y = (event.xdata) , (event.ydata)
    #call region based growing if its the first time
    if flag :
        region_growing(int(y),int(x))

#create figure
fig,ax = plt.subplots()
#show image
ax.imshow(img,cmap='gray',vmax=255,vmin=0)
plt.axis(False)
#ready to get mouse position
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()