import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

#read images 
surgery = cv.imread("surgery.jpeg")
cells = cv.imread("redcell.jpeg")
#convert color to gray and remove noise(smothing)
cells_gray = cv.cvtColor(cells,cv.COLOR_BGR2GRAY)
cells_gray = cv.medianBlur(cells_gray,5)
surgery_gray = cv.cvtColor(surgery, cv.COLOR_BGR2GRAY)
surgery_gray = cv.GaussianBlur(surgery_gray,(7,7),2)
#using canny in order to edge detection
edges=cv.Canny(surgery_gray,50,120)
#using hough transform for line detection
lines = cv.HoughLinesP(edges, 1, np.pi/180, 85, minLineLength=50, maxLineGap=20)

#draw line on the image 
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(surgery, (x1, y1), (x2, y2), (50, 0, 0), 2)
# show image 
plt.figure()
surgery = cv.cvtColor(surgery, cv.COLOR_BGR2RGB)
plt.imshow(surgery)
plt.axis(False)
plt.title("Recognize medical instrunments")
plt.show()
# using hough transformation in order to identify cells 
circles = cv.HoughCircles(cells_gray, cv.HOUGH_GRADIENT, 1, cells.shape[0]/11, param1=250, param2=13, minRadius=0, maxRadius=45)

# Draw detected circles
if circles is not None:
    #radius should be integer
    circles = np.uint16(np.around(circles))
    #detect white or red cell
    for i in circles[0, :]:
        # Draw circle
        if i[2]>40:
            cv.circle(cells, (i[0], i[1]), i[2], (0, 0, 0), 2)
        else:
            cv.circle(cells, (i[0], i[1]), i[2], (200, 0, 255), 2)

#show image with detected cells
plt.figure()
cells = cv.cvtColor(cells, cv.COLOR_BGR2RGB)
plt.imshow(cells)
plt.axis(False)
plt.title("determine white cell with black color and red cell with purple")
plt.show()


