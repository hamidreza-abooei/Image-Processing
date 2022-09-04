# this is for sin function
import math
# this is a variable for 
integral = 0
# step is 1/200
step = 1/200
# range can only get integer numbers. 
for i in range(int(-3/step),int(4/step)):
    integral += math.sin((i*step)**2)
#integral have a (delta x) * f(x) so we have to multiple it to (delta x)
integral = integral*step
print(integral)
