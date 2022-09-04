import cv2 as cv
import numpy as np

def get_points(img, points):
    cv.imshow("image", img)
    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            print(x, ' ', y)
            points.append((x, y))
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, str(x) + ',' + str(y), (x, y), font, 0.5, (0, 0, 255), 1)
            cv.imshow('image', img)

    while True:
        cv.setMouseCallback('image', click_event)
        k = cv.waitKey(0)
        if k == ord('n'):
            cv.destroyAllWindows()
            break



imgf = cv.imread("MRIF.png")
imgf_c = imgf.copy()
points_f = []
imgs = cv.imread("MRIS.png")
imgs_c = imgs.copy()
points_s = []

print("click 3 point then press n")
get_points(imgf_c, points_f)
points_f = np.float32(points_f)
print("click 3 point then press n")
get_points(imgs_c, points_s)
points_s = np.float32(points_s)

print(points_f)
print(points_s)
print(np.shape(points_s))
Affine = cv.getAffineTransform(points_f, points_s)
print(Affine)

img_transformed = cv.warpAffine(imgf, Affine, (imgs.shape[1], imgs.shape[0]))
cv.imshow('new image', img_transformed)
cv.waitKey(0)