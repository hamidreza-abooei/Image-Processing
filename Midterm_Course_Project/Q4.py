import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

xray_checkered = cv.imread("xray_checkered.png", 0)
# fft to convert the image to freq domain
fft_img = np.fft.fft2(xray_checkered)
# shift the center
fft_img_shift = np.fft.fftshift(fft_img)
# find magnitude_spectrum
mag = 20 * np.log(np.abs(fft_img_shift))

plt.figure()
plt.subplot(121)
#show the noisy image
plt.imshow(xray_checkered, cmap='gray', vmin=0, vmax=255)
plt.title('noisy image')
plt.axis(False)
plt.subplot(122)
#show the spectrum of noisy image
plt.imshow(mag, cmap='gray', vmin=0, vmax=255)
plt.title('spectrum of noisy image')
plt.axis(False)
#the shape of noisy image
height, width = xray_checkered.shape

for i in range(height):
    for j in range(width):
        #find the noisy point
        if (mag[i][j] >= 256 and mag[i][j] <= 259):
        #replace average of 4 neighbourhood in noisy point
            fft_img_shift[i][j] = (fft_img_shift[i - 1][j] + fft_img_shift[i + 1][j] + fft_img_shift[i][j - 1] +
                                   fft_img_shift[i][j + 1]) / 4
        else:
            pass

#inverse fourier transform
f_ishift = np.fft.ifftshift(fft_img_shift)
# apply the inverse shift
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

#show the denoised image
plt.figure()
plt.imshow(img_back,cmap='gray')
plt.title("denoised image")
plt.axis(False)
plt.show()


