# part a:
import numpy as np 
import cv2 as cv
# pre defining an nD array to examine our code:
x=np.ndarray((2,2,2), dtype='float')
x[0][0][0]=60 
x[0][0][1]=80.9
x[0][1][0]=300.5
x[0][1][1]=50
x[1][0][0]=400
x[1][1][1]=200.8
x[1][0][1]=160.2
x[1][1][0]=350
# in the assumption, we know maximum and minimum amout of inputs as a and b :
a=50
b=400
c = np.shape(x)
# we use another array to to use as output
y=np.ndarray(c,dtype='uint8')
# in the last part we find new numbers between 0 to 255
for i in range(c[0]):
    for j in range(c[1]):
        for k in range(c[2]):
            y[i][j][k] = int((x[i][j][k] - a ) * 255 / (b-a) )
# print results
print("our input is:\n",x,"\n\n\nand our output is : \n",y )

# part b:
def transformer(x):
    ma = np.amax(x)
    mi = np.amin(x)
    shap = np.shape(x)
    # we use another array to to use as output
    y=np.ndarray(shap,dtype='uint8')
    # in the last part we find new numbers between 0 to 255
    for i in range(shap[0]):
        for j in range(shap[1]):
            for k in range(shap[2]):
                y[i][j][k] = int((x[i][j][k] - a ) * 255 / (ma-mi) )
    return y
# create 50*40*3 random array:
x = 12.5 * np.random.rand(50,40,3) - 3.2
# applying function:
y = transformer(x)
print("this is the result of transformer function:\n", y )
# show image
cv.imshow("random",y)
cv.waitKey(0)
