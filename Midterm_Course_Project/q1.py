import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#show the image
retina = cv.imread('retina.jpg')
cv.imshow('retina',retina)
cv.waitKey(0)
#size of image=(width,height,number of channel)
rshape=np.shape(retina)
print('size:',rshape[0] ,'×', rshape[1])
#type of image
print('type:',retina.dtype)

# number  of channel of image=3(rgb)
print("channel : ", rshape[2])
#blue, green, red = cv.split(retina)

#show channels
#cv.imshow('red',red) # Display the red channel in the image
#cv.imshow('blue',blue) # Display the red channel in the image
#cv.imshow('green',green) # Display the red channel in the image
gray_retina = cv.imread('retina.jpg',0)


#a)memory=size of imge(grayscale) *(number of channel)
#channelnum=3
#size of imge(grayscale)
print(gray_retina.shape)
#memory storage
#memory=channelnum * gray_retina.shape
#print('memory:',memory)




#histogram tasvir tak kanale
#we plot a histogram, ravel() flatens our image array
plt.hist(gray_retina.ravel(), bins=256)
plt.title('histogram retina')
plt.show()

retina_sub = cv.imread('retina_sub.jpg',0)
#find the shape of image
shape_sub = np.shape(retina_sub)
print(shape_sub)
hists, bins = np.histogram(retina_sub.ravel(), bins=64)
#we plot a histogram, ravel() flatens our image array
#plt.hist(retina_sub.ravel(),256,[0,256])
#plt.title('histogram retina_sub')
#plt.show()
#c)
hist_sub , bins = np.histogram(retina_sub.ravel(), bins=64)
# this is a big number
maxdiff = 100000000
# step is 80
for i in np.arange(0,rshape[0]-shape_sub[0],80):
    for j in np.arange(0,rshape[1]-shape_sub[1],80):
        # getting histogram of a window of retina
        hist , bins=np.histogram(gray_retina[i:shape_sub[0]+i,j:shape_sub[1]+j].ravel(),bins=64)
        #calculate difference of window and retina_sub
        diff = np.sum(np.abs(hist_sub - hist))
        #if window was more similar , pick it up
        if diff < maxdiff:
            #pick x,y and diff amount of maximum similarity
            maxx , maxy , maxdiff = i , j , diff


# histogram of retina_sub and bins=256
hist_sub , bins = np.histogram(retina_sub.ravel(), bins=256)
# another big number
maxdiff = 1000000000000
for i in np.arange(maxx-80,maxx+80,10):
    for j in np.arange(maxy-80,maxy+80,10):
        # hist,binss = cv.calcHist(retina[i:shape_sub[0]-i,j:shape_sub[1]-j].ravel(), 64,[0,256])
        hist , bins=np.histogram(retina[i:shape_sub[0]+i,j:shape_sub[1]+j].ravel(),bins=256)
        # print((hist_sub - hist))
        diff = np.sum(np.abs(hist_sub - hist))
        #if window was more similar , pick it up
        if diff < maxdiff:
            #pick x,y and diff amount of maximum similarity
            maxx , maxy , maxdiff = i , j , diff


#show founded image
plt.imshow(retina[maxx:shape_sub[0]+maxx,maxy:shape_sub[1]+maxy] , cmap='gray' , vmin = 0 ,vmax= 256)
plt.title("similarity founded here")
plt.axis(False)
plt.show()


#d)
def bitplaneslicing(img):
    # Iterate over each pixel and change pixel value to binary using np.binary_repr() and store it in a list.
    lst = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # this have to repare

            lst.append(np.binary_repr(img[i][j], width=8))  # width = no. of bits

    # We have a list of strings where each string represents binary pixel value. To extract bit planes we need to iterate over the strings and store the characters corresponding to bit planes into lists.
    # Multiply with 2^(n-1) and reshape to reconstruct the bit image.
    eight_bit_img = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(img.shape[0], img.shape[1])
    seven_bit_img = (np.array([int(i[1]) for i in lst], dtype=np.uint8) * 64).reshape(img.shape[0], img.shape[1])
    six_bit_img = (np.array([int(i[2]) for i in lst], dtype=np.uint8) * 32).reshape(img.shape[0], img.shape[1])
    five_bit_img = (np.array([int(i[3]) for i in lst], dtype=np.uint8) * 16).reshape(img.shape[0], img.shape[1])
    four_bit_img = (np.array([int(i[4]) for i in lst], dtype=np.uint8) * 8).reshape(img.shape[0], img.shape[1])
    three_bit_img = (np.array([int(i[5]) for i in lst], dtype=np.uint8) * 4).reshape(img.shape[0], img.shape[1])
    two_bit_img = (np.array([int(i[6]) for i in lst], dtype=np.uint8) * 2).reshape(img.shape[0], img.shape[1])
    one_bit_img = (np.array([int(i[7]) for i in lst], dtype=np.uint8) * 1).reshape(img.shape[0], img.shape[1])

    return one_bit_img, two_bit_img, three_bit_img, four_bit_img, five_bit_img,six_bit_img , seven_bit_img , eight_bit_img


# Read the image in greyscale
image = cv.imread('retina.jpg', 0)
# apply function
plans = bitplaneslicing(image)
plt.suptitle("Bit plane slicing")

# show image
plt.subplot(2, 4, 1)
plt.imshow(plans[7], cmap='gray')
plt.title("8 bit")
plt.axis(False)

plt.subplot(2, 4, 2)
plt.imshow(plans[6], cmap='gray')
plt.title("7 bit")
plt.axis(False)

plt.subplot(2, 4, 3)
plt.imshow(plans[5], cmap='gray')
plt.title("6 bit")
plt.axis(False)

plt.subplot(2, 4, 4)
plt.imshow(plans[4], cmap='gray')
plt.title("5 bit")
plt.axis(False)

plt.subplot(2, 4, 5)
plt.imshow(plans[3], cmap='gray')
plt.title("4 bit")
plt.axis(False)

plt.subplot(2, 4, 6)
plt.imshow(plans[2], cmap='gray')
plt.title("3 bit")
plt.axis(False)

plt.subplot(2, 4, 7)
plt.imshow(plans[1], cmap='gray')
plt.title("2 bit")
plt.axis(False)

plt.subplot(2, 4, 8)
plt.imshow(plans[0], cmap='gray')
plt.title("1 bit")
plt.axis(False)


plt.show()

