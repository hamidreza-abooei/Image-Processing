import numpy as np
import matplotlib.pyplot as plt
import cv2

#read image
img = cv2.imread('lung.png',0)
#remove salt paper noise
img_2 = cv2.medianBlur(img, 3)
#gradient with sobel filter in y direction 
img_3 = cv2.Sobel(img_2, cv2.CV_64F, dx=1, dy=0)
#get absoulute from gradient for taking both upward and downward gradient
img_3 = np.abs(img_3)
#correct the period
img_3 = img_3 * 255 /np.max(img_3)
#correct the type of gradient
img_3 = img_3.astype('uint8')

#open figure to plot
plt.figure()
#write superior title
plt.suptitle('Problem 3 Figure')

#first part of subplot 
plt.subplot(1, 3, 1)
#title
plt.title('Original')
#show image
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
#hide axis
plt.axis(False)

#second image showing
plt.subplot(1, 3, 2)
plt.title('Denoised')
plt.imshow(img_2, cmap='gray',vmin=0,vmax=255)
plt.axis(False)

#third image showing 
plt.subplot(1, 3, 3)
plt.title('Gradient')
plt.imshow(img_3, cmap='gray',vmin=0,vmax=255)
plt.axis(False)

#show 
plt.show()

#this part is for getting gradient from Both sides
# img_4 = cv2.Sobel(img_2, cv2.CV_8U, dx=1, dy=0)
# img5 = (img_3 + img_4) / 2
# plt.title('both')
# plt.imshow(img5,cmap= 'gray', vmin = 0 , vmax= 255)
# plt.axis(False)
# plt.show()