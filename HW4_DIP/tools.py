import cv2
import numpy as np

def logmagnitude(img):
    '''input: 2D image, output: 2D logmagnitude of fourier transform'''
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    return np.log(np.abs(dft_shift))

def normal(img):
    '''input: 2D image, output: float 2D image in range [-1, 1]'''
    img = img.astype(float)
    mi, ma = img.min(), img.max()
    return 2*(img - mi)/(ma-mi)-1

def n_range(img):
    '''input: 2D image, output: uint8 2D image in range [0, 255]'''
    mi, ma = img.min(), img.max()
    return np.round(255*(img.astype(float) - mi)/(ma-mi)).astype('uint8')