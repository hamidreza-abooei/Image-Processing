import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv

#read image
noisy_skull = cv.imread("noisy_skull.png",0)
#determine shape
shape = np.shape(noisy_skull)
#get two parts which contain only noise 
black_niosy1 = noisy_skull[:,0:150] 
black_niosy2 = noisy_skull[:,(shape[1]-150):shape[1]]
#concatinate two parts to one part
black_niosy = np.concatenate((black_niosy1, black_niosy2),axis=1)
#plot normalized histogram
plt.figure()
plt.hist(black_niosy.ravel(),bins=255)
#add title
plt.title("Normalized Histogram of Noise")

#calculate the mean of noise 
mean_noisy = np.mean(black_niosy)
print("Mean of noise is:     ",mean_noisy)
#calculate the variance of noise 
variance_noisy = np.var(black_niosy)
print("Variance of noise is: ", variance_noisy)


##Addaptive Local noise reduction filter section 

#Padding image :
bordered_image = cv.copyMakeBorder(noisy_skull,4,4,4,4,cv.BORDER_REPLICATE)
# pre define output 
adaptive_res = np.zeros(shape)
#pre define mean of windows 
local_mean = np.zeros(shape)
#pre define variance of windows 
local_var =  np.zeros(shape)
#search through whole image for windows
for i in range(4,shape[0]+4):
    for j in range(4,shape[1]+4):
        #define window 
        local = bordered_image[i-4:i+4,j-4:j+4]
        #calculate mean and variance of local window
        local_mean[i-4,j-4] = np.mean(local)
        local_var[i-4,j-4]  = np.var(local)
# mean variance of windows 
variance_of_noise = np.mean(local_var)
#get thresholding to determine variances less than mean 
_ , inv_th = cv.threshold(local_var,variance_of_noise , 1,cv.THRESH_BINARY )
#change type to 8 bit unsigned int 
inv_th = (inv_th).astype('uint8')
# get inverse of thresholding
th = np.ones(shape,dtype = "uint8") - inv_th
# mask on local windows variances
local_var = cv.bitwise_and(local_var,local_var,mask=inv_th)
# produce mean variance for applying in low variance places
minimum = np.ones(shape , dtype = "uint8") * variance_of_noise 
# replace mean of variance in place that variance are low
minimum = cv.bitwise_and(minimum,minimum,mask=th)
# mean variance applyied in less than mean variance
local_var = local_var + minimum
# calculata adaptive filter output
adaptive_res = noisy_skull - ( variance_of_noise / local_var ) * ( noisy_skull - local_mean )
#change type
adaptive_res = adaptive_res.astype("uint8")

# median filter
median_filtered = cv.medianBlur(noisy_skull,7)
# show original image 
plt.figure()
plt.subplot(2,3,1)
plt.imshow(noisy_skull,cmap='gray',vmin=0,vmax=255)
plt.axis(False)
plt.title("Original image")
#show histogram of original image 
plt.subplot(2,3,4)
plt.hist(black_niosy.ravel(),bins=255)
plt.ylim((0,30000))
plt.xlim((0,75))
plt.title("histogram of Original image noise")
#show median filter 
plt.subplot(2,3,2)
plt.imshow(median_filtered,cmap='gray')
plt.axis(False)
plt.title("median filtered image")

#finding noise part of median filtered image
black_niosy1 = median_filtered[:,0:150] 
black_niosy2 = median_filtered[:,(shape[1]-150):shape[1]]
black_niosy_median = np.concatenate((black_niosy1, black_niosy2),axis=1)
# plot histogram of median filtered image
plt.subplot(2,3,5)
plt.hist(black_niosy_median.ravel(),bins=255)
plt.ylim((0,30000))
plt.xlim((0,75))
plt.title("median filter noise histogram ")

#show adaptive res
plt.subplot(2,3,3)
plt.imshow(adaptive_res,cmap='gray')
plt.axis(False)
plt.title("adaptive filter ")

#finding noise part of adaptive filtered image
black_niosy1 = adaptive_res[:,0:150] 
black_niosy2 = adaptive_res[:,(shape[1]-150):shape[1]]
black_niosy_adaptive = np.concatenate((black_niosy1, black_niosy2),axis=1)
# plot histogram of adaptive_res 
plt.subplot(2,3,6)
plt.hist(black_niosy_adaptive.ravel(),bins=255)
plt.ylim((0,30000))
plt.xlim((0,75))
plt.title("hist of noise in adaptive ")

plt.show()