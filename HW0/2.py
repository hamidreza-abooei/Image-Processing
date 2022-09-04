import math
#defining function
def prime_checker(x):
    # difining a flag to determine prime or not
    flag = True
    #check prime
    for i in range(2,int(math.sqrt(x)+1)):
        if ( x % i == 0 ):
            #if we find any divisors, mark it as not prime and exit loop
            flag = False
            break
    return flag

#check function
for i in range(2,50):
    print( i , "is a prime number: " , prime_checker(i))
