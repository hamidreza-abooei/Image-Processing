import cv2 as cv
import numpy as np

img = cv.imread("surgery.jpeg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.medianBlur(gray, 7)
# img_blur = gray
edges = cv.Canny(img_blur, 50, 120, apertureSize=3)
lines = cv.HoughLinesP(edges, 1, np.pi/180, 85, minLineLength=50, maxLineGap=20)

for line in lines:
    x0,y0,x1,y1 = line[0]
    cv.line(img, (x0, y0), (x1, y1), [0, 255, 255], 2)

cv.imshow("Houghlines", img)
cv.waitKey(0)