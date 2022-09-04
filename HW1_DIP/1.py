import cv2 as cv
import numpy as np
# importing image
mandrill = cv.imread("Images\mandrill.jpg")
# Error if image doesn't exist
if mandrill is None:
    print('Could not open or find the image: ', "mandrill")
    exit(0)
# show shape and type
print( np.shape(mandrill) , type(mandrill[0][0][0]) )
#convert to gray scale
cvtmandrill = cv.cvtColor( mandrill , cv.COLOR_BGR2GRAY )
cv.imshow("convert to gray level",cvtmandrill)
# convert it to 64 levels
mandrill64 = (cvtmandrill // 4)
mandrill64 = mandrill64 * 4
#convert it to 16 levels
mandrill16 = (cvtmandrill // 16)
mandrill16 = mandrill16 * 16
# convert it to binary
_,mandrill2 = cv.threshold(cvtmandrill,128,255,cv.THRESH_BINARY)

#show
cv.imshow("64",mandrill64)
cv.imshow("16",mandrill16)
cv.imshow("2",mandrill2)
# cut image . R and L
mandrillL = mandrill[:,0:mandrill.shape[1]//2]
mandrillR = mandrill[:,mandrill.shape[1]//2:mandrill.shape[1]]
# concat achived images
concatinatedRL=np.hstack((mandrillR,mandrillL)) 

cv.imshow("concatinated R & L of original image ",concatinatedRL)

#inverse figure
mandrillhreversed = mandrill[:,::-1]
cv.imshow("mandrill horizontal reversed ", mandrillhreversed)

mandrillvreversed = mandrill[::-1]
cv.imshow("mandrill vertical reversed ", mandrillvreversed)

#save figure
#cv.imwrite("Images\mandrillvrev.png" , mandrillvreversed)

#scaling *3 with Linear interpolation
scale31 = cv.resize(cvtmandrill,(3*cvtmandrill.shape[0],3*cvtmandrill.shape[1]),interpolation=cv.INTER_LINEAR)
cv.imshow("scale 3 Linear interpolation",scale31)
#scaling *3 with Nearest interpolation
scale32 = cv.resize(cvtmandrill,(3*cvtmandrill.shape[0],3*cvtmandrill.shape[1]),interpolation=cv.INTER_NEAREST)
cv.imshow("scale 3 Nearest interpolation",scale32)
#scaling *3 with Area interpolation
scale33 = cv.resize(cvtmandrill,(3*cvtmandrill.shape[0],3*cvtmandrill.shape[1]),interpolation=cv.INTER_AREA)
cv.imshow("scale 3 Area interpolation",scale33)
#scaling /3 
dim =(cvtmandrill.shape[0]//3,cvtmandrill.shape[1]//3)
scale13 = cv.resize(cvtmandrill , dim )
cv.imshow("scale 1/3",scale13)


if cv.waitKey(0):
    cv.destroyAllWindows()