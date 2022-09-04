#Exit by pressing Esc
# import libraries
import cv2 as cv
import numpy as np
#name show window
cv.namedWindow("preview")
#capture video from webcam
vc = cv.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()

    #grayscale
    frame1 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #copy for comparison 
    # frame1 = frame
    #get webcam resoulotion
    row , col = frame1.shape
    crop_size = row
    #detecting the number of motions (it used to show new motion in the output easily)
    motion = 0
    #counter for counting how much motion detected
    counter=0
else:
    rval = False
    
while rval:
    #Show 
    cv.imshow("preview", frame[:crop_size, :crop_size])
    #copy for comparison
    frame1=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #read from webcam
    rval, frame = vc.read()
    #grayscale
    frame2 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    
    #crop image to reach square
    image1_cp = frame1[:crop_size, :crop_size] # crop to square
    image2_cp = frame2[:crop_size, :crop_size] # crop to square

    #fourier transform of images
    f1 = cv.dft(image1_cp.astype(np.float32),flags=cv.DFT_COMPLEX_OUTPUT)
    f2 = cv.dft(image2_cp.astype(np.float32),flags=cv.DFT_COMPLEX_OUTPUT)
    
    #shift to centering
    f1_shf = np.fft.fftshift(f1) 
    f2_shf = np.fft.fftshift(f2)

    #convert to complex form
    f1_shf_cplx = f1_shf[:,:,0] + 1j*f1_shf[:,:,1]
    f2_shf_cplx = f2_shf[:,:,0] + 1j*f2_shf[:,:,1]

    #absolute of complex number
    f1_shf_abs = np.abs(f1_shf_cplx)
    f2_shf_abs = np.abs(f2_shf_cplx)
    # multiple abs of two image 
    total_abs = f1_shf_abs * f2_shf_abs
    # phase corrilation algorithm
    P_real = (np.real(f1_shf_cplx)*np.real(f2_shf_cplx) + np.imag(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
    P_imag = (np.imag(f1_shf_cplx)*np.real(f2_shf_cplx) + np.real(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
    P_complex = P_real + 1j*P_imag
    # get the absolute amount of corrilation
    P_inverse = np.abs(np.fft.ifft2(P_complex)) # inverse FFT
    #this is the id of transferring 
    max_id = [0, 0]
    #maximum value of corrilation 
    max_val = 0
    #find maximum value 
    for idy in range(crop_size):
        for idx in range(crop_size):
            if P_inverse[idy,idx] > max_val:
                max_val = P_inverse[idy,idx]
                max_id = [idy, idx]
    #find transfer matrix
    shift_x = crop_size - max_id[0]
    shift_y = crop_size - max_id[1]
    # print(shift_x, shift_y)
    #detect shift 
    if shift_x != row :
        counter+=1
        print(str(counter)+"motion detected")
    elif shift_y != row :
        counter+=1
        print(str(counter)+"motion detected")

    key = cv.waitKey(1)
    if key == 27: # exit on ESC
        break
cv.destroyWindow("preview")