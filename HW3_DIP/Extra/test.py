import cv2
import numpy as np

image1 = cv2.imread("1.png")[:,:,0]# take red components
image2 = cv2.imread("2.png")[:,:,0] # take red components

crop_size = 450
# image1_cp = image1[:crop_size, :crop_size] # crop to square
# image2_cp = image2[:crop_size, :crop_size] # crop to square


f1 = cv2.dft(image1.astype(np.float32),flags=cv2.DFT_COMPLEX_OUTPUT)
f2 = cv2.dft(image2.astype(np.float32),flags=cv2.DFT_COMPLEX_OUTPUT)
f1_shf = np.fft.fftshift(f1) 
f2_shf = np.fft.fftshift(f2)

f1_shf_cplx = f1_shf[:,:,0] + 1j*f1_shf[:,:,1]
f2_shf_cplx = f2_shf[:,:,0] + 1j*f2_shf[:,:,1]

f1_shf_abs = np.abs(f1_shf_cplx)
f2_shf_abs = np.abs(f2_shf_cplx)
total_abs = f1_shf_abs * f2_shf_abs
P_real = (np.real(f1_shf_cplx)*np.real(f2_shf_cplx) +
          np.imag(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
P_imag = (np.imag(f1_shf_cplx)*np.real(f2_shf_cplx) +
          np.real(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
P_complex = P_real + 1j*P_imag

P_inverse = np.abs(np.fft.ifft2(P_complex)) # inverse FFT
max_id = [0, 0]
max_val = 0
for idy in range(crop_size):
    for idx in range(crop_size):
        if P_inverse[idy,idx] > max_val:
            max_val = P_inverse[idy,idx]
            max_id = [idy, idx]
shift_x = crop_size - max_id[0]
shift_y = crop_size - max_id[1]
print(shift_x, shift_y)