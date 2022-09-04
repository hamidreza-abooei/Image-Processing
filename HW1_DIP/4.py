#Exit by pressing Esc

import cv2 as cv
import numpy as np
#name show window
cv.namedWindow("preview")
#capture video from webcam
vc = cv.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    #grayscale
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #copy for comparison 
    frame1 = frame
    #get webcam resoulotion
    row , col = frame.shape
    #detecting the number of motions (it used to show new motion in the output easily)
    motion = 0
else:
    rval = False
    
while rval:
    #Show 
    cv.imshow("preview", frame)
    #copy for comparison
    frame1=frame
    #read from webcam
    rval, frame = vc.read()
    #grayscale
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #detect difference
    diff = cv.subtract(frame , frame1)
    #avgarage of difference
    mean = np.mean(diff)
    #motion detector
    if mean > 10 :
        motion = motion + 1
        print( motion ,"motion detected!" )
    
    key = cv.waitKey(1)
    if key == 27: # exit on ESC
        break
cv.destroyWindow("preview")