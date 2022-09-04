#import libraries
import cv2 as cv
import numpy as np 
#import image 
kidney = cv.imread("images/kidney.tif" ,0)
#defiene the first function 
def s1(r):
    if ( 160 < r < 240 ):
        return int(150)
    return int(0)
#define the second function 
def s2(r):
    if ( 100 < r < 165 ):
        return 200
    return r
#define a transformation matrix function 
transformation1 = np.vectorize(s1)
#apply first function to original image
transformed1 = transformation1(kidney)
#correct transformation matrix array type
transformed1 = transformed1.astype('uint8')

#define a transformation matrix function
transformation2 = np.vectorize(s2)
#apply second function to original image
transformed2 = transformation2(kidney)
#correct transformation matrix array type
transformed2 = transformed2.astype('uint8')

#concatinate 3 images to show them together
concated = np.concatenate((kidney , transformed1 , transformed2),1)
#show image
cv.imshow("transformed2",concated)
cv.waitKey(0)
