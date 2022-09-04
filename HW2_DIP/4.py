import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#read image
img = cv.imread("images/bone-scan.png",0)
#filter function 
def filterfunc(img, array):
    if type(array) == np.ndarray or type(array) == list:
        if np.shape(array) == (3,3):
            img2 = cv.copyMakeBorder( img, 1 , 1 , 1 , 1 , cv.BORDER_REFLECT101 )
            shape = img2.shape
            out = np.ndarray((shape[0]-2,shape[1]-2) , dtype = 'uint8')
            for i in range(1,shape[0]-1):
                for j in range(1,shape[1]-1):
                    
                    #I don't use down code because it decrease the speed of running code
                    # s = 0
                    # for k in range(3):
                    #     for l in range(3):
                    #         s += array[k][l] * img2[i+k-1][j+l-1]

                    #instead I use this long line. 
                    s = array[0][0] * img2[i-1][j-1] + array[1][0] * img2[i][j-1] + array[2][0] * img2[i+1][j-1] + array[0][1] * img2[i-1][j] + array[1][1] * img2[i][j] + array[2][1] * img2[i+1][j] + array[0][2] * img2[i-1][j+1] + array[1][2] * img2[i][j+1] + array[2][2] * img2[i+1][j+1] 

                    out[i-1][j-1] = np.clip(s,0,255)

            return out                            
        else:
            #return false if its not 3*3
            return False
    #median filter
    elif (array == 'median'):
        #make reflect border in order to apply filter
        img2 = cv.copyMakeBorder( img, 1 , 1 , 1 , 1 , cv.BORDER_REFLECT101 )
        #determine image shape
        shape = img2.shape
        #define zero image with respect to original image size
        out = np.ndarray((shape[0]-2,shape[1]-2) , dtype = 'uint8')
        #finding median intensity of 3*3 image
        for i in range(1,shape[0]-1):
            for j in range(1,shape[1]-1):
                a = img2[i-1:i+2,j-1:j+2]
                #reshape 3*3 to 1*9 then sort it
                out[i-1][j-1] = np.sort(np.reshape(a,(1,9))[0])[4]
        return out
    else:
        #if its not 'median'
        return False





# finding median filter
medianimg = filterfunc(img ,'median')
#subplots 1*2
fig , axs = plt.subplots(nrows = 1 , ncols = 2 )
fig.suptitle("Filtering")
#show original image
axs[0 ].imshow(img, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 ].set_title("Original image")
axs[0 ].set_axis_off() 
#show Median filtered image
axs[1].imshow(medianimg, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[1].set_title("Median Filtered")
axs[1].set_axis_off() 
#average filter
avgimg = filterfunc(img , [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])

#showing two images together
fig2 , axs = plt.subplots(nrows = 1 , ncols = 2 )
fig2.suptitle("Difference between Median and Average Filtering")
#show avg filter
axs[0 ].imshow(avgimg, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[0 ].set_title("Average Filtered")
axs[0 ].set_axis_off() 
#show median filter
axs[1].imshow(medianimg, cmap = 'gray' , vmin = 0 , vmax = 255)
axs[1].set_title("Median Filtered")
axs[1].set_axis_off() 
#calculating laplacian with my own function 
laplacian = filterfunc(medianimg , [[0 , 1, 0 ] , [ 1 , -4 , 1  ] , [ 0 , 1 , 0 ]])
laplacian= cv.convertScaleAbs(laplacian)

#calculating laplacian with predefined function
laplaciancv = cv.Laplacian(medianimg , cv.CV_8U )
laplaciancv = cv.convertScaleAbs(laplaciancv)

#tell us that our code is working well
print(np.all(laplaciancv==laplacian))

#Dynamic image  
fig, ax = plt.subplots()
#adjust subplot from bottom
plt.subplots_adjust(bottom=0.25)
#title
plt.suptitle('Adding laplacian to original image with varying C')
#define color for slider
axcolor = 'lightgoldenrodyellow'
#show image in first subplot
l=plt.subplot(1,2,1)
l=l.imshow(medianimg , cmap='gray' , vmin= 0 , vmax=255)
axc = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
#subplot for hist
m=plt.subplot(1,2,2)
#plotting histogram 
[n,X,V]=m.hist(medianimg.ravel() , bins=256 , log=True , color = 'k')
#define slider
sc = Slider(axc,'c' , -20.0 , 20.0 , valinit=0 , valstep=0.5)

#update function for responsing to changes
def update(val):
    #getting current value
    c= sc.val
    #adding laplacian to image with specific amount
    new = np.clip((medianimg + c * laplacian),0,255).astype('uint8')
    #set data
    l.set_data(new)
    #updating histogram (rewrite histogram)
    m.cla()
    [n,X,V]=m.hist(new.ravel(), bins=256 , log=True,color='k')
    fig.canvas.draw_idle()
#call update function when slider changes
sc.on_changed(update)


plt.show()

cv.waitKey(0)
