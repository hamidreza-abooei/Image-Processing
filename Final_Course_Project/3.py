import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv

# Create a VideoCapture object and read from input file
cap = cv.VideoCapture("Q_three.AVI")
#get details of video
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Check if file opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

#pre define background
background = np.zeros((height,width))
back = []
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #read until end
    if ret == True:
        #convert frames to grayscale
        frame1 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        #append frame to calculate background
        back.append(frame1)
        # Press Q on keyboard to  exit
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
    # Break the loop
    else: 
        break

#calculate the background by getting median of image through the axis of time
background = np.median(back,axis=0)
#save background
cv.imwrite("background.jpg",background)
#read image again
cap = cv.VideoCapture("Q_three.AVI")

# a variable to detect the difference between frame and background
diff = np.zeros((height,width),dtype="float32")
#a res background
red = np.zeros((height,width,3))
#turn mask to red
red[:,:,2] = 255
#change type to uint8
red = red.astype("uint8")

#read image once again to find what we wants 
counter = 0
while(cap.isOpened()):
    counter += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    #read until end
    if ret == True:
        #convert frames to grayscale
        frame1 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        #subtract each frame from background
        diff = np.abs( np.subtract(frame1.astype("float32") , background.astype("float32"))) 
        #find motion 
        ret,mask = cv.threshold(diff,30,255,cv.THRESH_BINARY)
        #change type to uint8
        mask = mask.astype("uint8")
        #this section is for change color of motioned sections 
        #mask frame image with mask (but we want invert of it)
        frame_BGfree = cv.bitwise_and(frame,frame,mask=mask)
        #mask red image with mask
        red_masked = cv.bitwise_and(red,red,mask=mask)
        #invert of that 
        frame = frame - frame_BGfree
        #add masked image and red masked
        frame = cv.add(frame,red_masked)
        # Display the resulting frame 
        cv.imshow('motion detected',frame)
        cv.imwrite("frame %d.png"%counter,frame)
        # Press Q on keyboard to  exit
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
    # Break the loop
    else: 
        break
    
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()