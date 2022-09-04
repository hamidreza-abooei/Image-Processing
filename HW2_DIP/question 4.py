import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('bone-scan.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def median_mean(image, array):
    copy_image = image.copy()
    if len(array) == 3 and len(array[0]) == 3 and len(array[1]) == 3 and len(array[2]) == 3:
        b=0
        bordersize = 1
        border = cv.copyMakeBorder(copy_image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                   borderType=cv.BORDER_REFLECT101)
        kernel = np.zeros([3,3],dtype='float32')
        for i in range(3):
            for j in range(3):
                kernel[i,j] =array[i][j]
                b +=abs(kernel[i,j])
        for i in range(800):
            for j in range(500):
                copy_image[i, j] = (kernel[0, 0] * border[i, j ] + kernel[0, 1] * border[i , j+1] + kernel[
                    0, 2] * border[i , j + 2] + kernel[1, 0] * border[i+1, j] + kernel[1, 1] * border[i+1, j+1] +
                                    kernel[1, 2] * border[i+1, j + 2] + kernel[2, 0] * border[i +2, j ] + kernel[
                                        2, 1] * border[i + 2, j+1] + kernel[2, 2] * border[i + 2, j + 2]) / b

                #copy_image[i,j]=(kernel[0,0]*border[i-1,j-1]+kernel[0,1]*border[i-1,j]+kernel[0,2]*border[i-1,j+1]+kernel[1,0]*border[i,j-1]+kernel[1,1]*border[i,j]+kernel[1,2]*border[i,j+1]+kernel[2,0]*border[i+1,j-1]+kernel[2,1]*border[i+1,j]+kernel[2,2]*border[i+1,j+1])/b
                #if copy_image[i, j] > 255:
                    #copy_image[i, j] = 255
                #elif copy_image[i, j] < 0:
                    #copy_image[i, j] = 0

        copy_image = copy_image.astype('uint8')
        return copy_image
    elif 'median' in array:
        #median_image = cv.medianBlur(copy_image, 3)
        bordersize=1
        border = cv.copyMakeBorder(copy_image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                   borderType=cv.BORDER_REFLECT101)
        for i in range(800):
            for j in range(500):
                list=[border[i-1,j-1],border[i-1,j],border[i-1,j+1],border[i,j-1],border[i,j],border[i,j+1],border[i+1,j-1],border[i+1,j],border[i+1,j+1]]
                list.sort()
                copy_image[i,j]=list[4]


        return copy_image
    else:
        print('False')


cv.imshow('original', img)
img1 = median_mean(img, 'median')
img2 = median_mean(img, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
copy_image = img1.copy()
cv.imshow('median', img1)
cv.imshow('mean', img2)
#k = np.zeros([3, 3], dtype='float32')
#k[0, 1] = k[1, 0] = k[1, 2] = k[2, 1] = 1
#k[1, 1] = -4
#laplacian = cv.filter2D(copy_image, -1, k)
#laplacian1 = np.zeros([800, 500], dtype='float32')
#bordersize = 1
#border = cv.copyMakeBorder(copy_image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                           #borderType=cv.BORDER_CONSTANT, value=[0])
#for i in range(800):
    #for j in range(500):
        #border[i + 1, j + 1] = (-4) * border[i + 1, j + 1] + border[i, j + 1] + border[i + 1, j] + border[
            #i + 2, j + 1] + border[i + 1, j + 2]
#for i in range(800):
    #for j in range(500):
        #if border[i, j] > 255:
            #laplacian1[i, j] = 255
        #elif border[i, j] < 0:
            #laplacian1[i, j] = 0
        #else:
            #laplacian1[i, j] = laplacian[i, j]
#print(np.all(laplaciancv == laplacian))
#print(np.all(laplaciancv == laplacian1))
kernel=np.zeros([3,3],dtype='float32')
kernel[0,1]=kernel[1,0]=kernel[1,2]=kernel[2,1]=1
kernel[1,1]=-4

laplacian=median_mean(copy_image,kernel)
laplaciancv = cv.Laplacian(copy_image,cv.CV_8U)
print(type(laplacian[5][5]))




print(np.all(laplaciancv==laplacian))
cv.imshow('2',laplaciancv)
cv.imshow('1',laplacian)



cv.waitKey(0)
cv.destroyAllWindows()
