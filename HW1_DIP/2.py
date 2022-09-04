import cv2 as cv
import numpy as np
#read images 
dental_xray_mask = cv.imread("Images/dental_xray_mask.tif")
dental_xray = cv.imread("Images/dental_xray.tif")
# apply mask 
masked = cv.bitwise_and(dental_xray,dental_xray_mask)
#show
cv.imshow("Masked Image", masked)

partial_body_scan = cv.imread("Images/partial_body_scan.tif")
#convert to gray scale
partial_body_scan_gray = cv.cvtColor(partial_body_scan,cv.COLOR_BGR2GRAY)
partial_body_scan_gray_complement = 255 - partial_body_scan_gray  
#uinon
partial_body_scan_union = cv.bitwise_or(partial_body_scan_gray,partial_body_scan_gray_complement)
#concating 3 image to show
partial_body_scan_output = np.concatenate((partial_body_scan_gray , partial_body_scan_gray_complement , partial_body_scan_union) , axis = 1)
cv.imshow ("Partial Body scan", partial_body_scan_output)

#read image
angiography_live = cv.imread("Images/angiography_live.tif")
angiography_mask = cv.imread("Images/angiography_mask.tif")
#subtract images 
angiography_diff =cv.subtract(angiography_mask , angiography_live)
#show
cv.imshow("diff",angiography_diff)
#normalizing 
angiography_normalized = np.zeros(angiography_diff.shape)
angiography_normalized = cv.normalize(angiography_diff , angiography_normalized , 0 , 255 , cv.NORM_MINMAX)
cv.imshow("normalized Angiography",angiography_normalized)
print(np.amax(angiography_normalized))

cv.waitKey(0)
cv.destroyAllWindows