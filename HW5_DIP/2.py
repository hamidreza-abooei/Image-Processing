import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

def variance(a):
    # print(len(a))
    shape = np.shape(a)
    shape = shape[0]*shape[1]
        
    mean = np.mean(a)
    var = (np.sum(( a - mean )**2  ) / shape)**0.5
    return var


def split ( subimg , minshape , th):
    shape = np.shape(subimg)
    minimum = np.min(subimg)
    maximum = np.max(subimg)
    if shape[0] == minshape:
        return
    elif (maximum - minimum) > th :
        split(subimg[0:shape[0]/2,0:shape[1]/2] , minshape , th)
        split(subimg[0:shape[0],shape[1]/2:shape[1]], minshape , th)
        split(subimg[shape[0]/2:shape[0],0:shape[1]], minshape , th)
        split(subimg[shape[0]:shape[0],shape[1]/2:shape[1]], minshape , th)


# def splitmerge(size):
    
#     if 
#         return 


fmri = cv.imread("fMRI.jpg",0)
# print(np.shape(fmri))
shape=np.shape(fmri)
# fmri = np.pad(fmri,int((1024-shape[0])/2))
fmri = fmri[2:shape[0]-2,2:shape[1]-2]
shape=np.shape(fmri)
# print(np.shape(fmri))
# cv.imshow ("",fmri)
# cv.waitKey(0)
# cv.imshow("",fmri)
# cv.waitKey(0)
# splitmerge(2)
# splitmerge(4)
# splitmerge(16)


final = 16 
# final2 = 8
# final3 = 4
# final4 = 2

depth = int (shape[0] / final)

depth = int(np.log(depth)/np.log(2))

# print( depth)
x=1,2,3,2,1
# print(x)
# print(x[:])


def splitting(a):
    print(a)
    if len(a) == 1:
        return a
    else :
        
        return a[0] * splitting(a[1:])

x=(1,1,2,3,2)
print(splitting(x))

# def merging():
    
# x = [1,2,3,4,5,6,7]
# print(fmri)
# print(variance(fmri))
a = []
w = 256
x=np.zeros((w,w))
# print(x)
r = int(shape[0] / w)
for i in range(w):
    for j in range(w):
        x[i,j]=(variance(fmri[i*r:(i+1)*r,j*r:(j+1)*r]))
        a.append(x[i,j])
# print(a)    

# print(np.mean(a))
# print(x)
plt.imshow(x,cmap='gray')
plt.show()