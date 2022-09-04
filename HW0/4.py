import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from sympy import *
# define x as symbol to create functions
x = symbols('x')

# defining a function that gets list and convert it to a polynomial expretion 
def list2expr(inlist):
    expr = 0 
    for i in range(len(inlist)):
        expr += inlist[i] * x ** (len(inlist) - i - 1)
    return expr

# defining a function that gets a polynomial list and give derivative list 
def derivative(inlist):
    der = []
    a=len(inlist) 
    for i in range(a-1):
        der.append((a-i-1)*inlist[i])
    return der

#input expretion
experetion = [-1,5,3]
# fining derivation list
d_experetion = derivative(experetion)
#finding polynomial forms of expretion and derivation
expr = list2expr(experetion)
d_expr = list2expr(d_experetion)
# using lamdify to get f(x) to subs x with numbers
f = lambdify( x , expr , "numpy" )
d_f = lambdify( x , d_expr , "numpy" )
# x axis
a = np.arange(-5,5,step=0.01)

#plotting  
plt.figure()
plt.title("polynomial function and its derivative of "+str(expr))
plt.xlabel("x")
plt.ylabel("F(x)")
plt.plot(a,f(a) , "b" , linewidth = 2 ,label = "Original functino" )
plt.plot(a,d_f(a) , "r--" , label = "Derivative function")
plt.legend()
plt.show()