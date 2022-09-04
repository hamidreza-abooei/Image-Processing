import cv2 as cv 
import numpy as np
import math
#import
T = cv.imread("Images/T.jpg",0)
#scaling
cx = 1.5
cy = 1.5
#forming transform Matrix for scaling
TransformM = np.float32([[cx,0,0],[0,cy,0]])
row , col = T.shape
#Apply transformation 
scaled = cv.warpAffine(T,TransformM ,(int(col*cx) , int(row*cy)))
#show
cv.imshow("scaled",scaled)

#forming transform Matrix for Translation
TransformM = np.float32([[1,0,80],[0,1,-100]])
row , col = T.shape
#Apply transformation 
Translated = cv.warpAffine(T,TransformM ,(col , row))
#show
cv.imshow("Translation",Translated)

#horizontal sheer
sh=0.2
#forming transform Matrix for horizontal sheer
TransformM = np.float32([[1,0,0],[sh,1,0]])
row , col = T.shape
#apply transformation
sheerh = cv.warpAffine(T,TransformM ,(col , row))
#show
cv.imshow("Horizontal sheer",sheerh)

#rotaion forward
tetta = 0.5
row , col = T.shape
#empty image with original image size
rotation = np.zeros( (row , col) )
#forming transform Matrix for rotation
TransformM = np.float32([[math.cos(tetta),- math.sin(tetta)],[math.sin(tetta),math.cos(tetta)]])
#applying transformation
print(T[5][7])
for i in range(row):
    for j in range(col):
        outrow , outcol = TransformM @ ( i , j )
        outrow , outcol = int(outrow) , int(outcol)
        if outrow in range(row) and outcol in range(col):
            rotation[outrow][outcol] = T[i][j]
#show
cv.imshow("rotation forward",rotation)

#rotation reverse
tetta = 0.3
#empty image with original image size
rotationrev = np.zeros( (col , row) )
#forming transform Matrix for rotation
TransformM = np.float32([[math.cos(tetta),- math.sin(tetta)],[math.sin(tetta),math.cos(tetta)]])
#apply transformation reversed
for i in range(col):
    for j in range(row):
        inrow , incol   = TransformM @ (j , i)
        incol , inrow = int(incol), int(inrow)
        if inrow in range(row) and incol in range(col):
            rotationrev[i][j] = T[inrow][incol]
#show
cv.imshow("rotation reversed" , rotationrev)

#vertical sheer
sv=0.2
#forming transform Matrix for vertical sheer
TransformM = np.float32([[1,sv,0],[0,1,0]])
row , col = T.shape
#apply transformation
sheerv = cv.warpAffine(T,TransformM ,(col , row))
#show
cv.imshow("vertical sheer",sheerv)


cv.waitKey(0)
cv.destroyAllWindows()