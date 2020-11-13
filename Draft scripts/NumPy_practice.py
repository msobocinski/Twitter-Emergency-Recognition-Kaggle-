import numpy as np

# Task 1
np.__version__
np.show_config()

# Task 2 - get help
np.info(np.add)

# Task 3 - program to test whether there is zero in the array
np.info(np.array)

l1 = list(range(0,100))
l2 = list(range(1,100))

ar1 = np.array(l1)
ar2 = np.array(l2)

def is_zero (x):
    print(not(np.all(x)))
    print(0 in x)

is_zero(ar1)
is_zero(ar2)

# Task 5 - test a given array element-wise for finiteness (not infinity or not a Number)
# Change one value in ar1 to nan
ar1 = ar1.astype('float')
ar1[5] = np.nan

ar2 = ar2.astype('float')

# Create array with infinity
ar3 = np.array(object = list(np.random.randint(low = 1, high = 1000, size = 100)))
ar3 = ar3.astype('float')
ar3[13] = np.infty

def is_finite(x):
    if sum(np.isinf(x)) > 0:
        print('There is an infinite number in the array')
    if sum(np.isnan(x)) > 0:
        print('There is a nan value in the array')

is_finite(ar1)
is_finite(ar2)
is_finite(ar3)

# Task 11 - create an array with the values 1, 7, 13, 105 and determine the size of the memory occupied by the array.
ar4 = np.array(object = (1,7,13,105))
print("The array above occupies %d bytes" %(ar4.itemsize * ar4.size))

# Task 12 - Write a NumPy program to create an array of 10 zeros,10 ones, 10 fives.
ar5 = np.array(object = (np.repeat(0, 10)))
ar5 = np.append(ar5, np.repeat(1,10))
ar5 = np.append(ar5, np.repeat(5,10))

# Task 14 - create an array of even integers from 30 to 70
ar6 = np.arange(start = 30, stop = 71, step = 2)

# Task 16 - create a 3x3 identity matrix
np.identity(n = 3)

# Task 17 - generate a random number between 0 and 1
np.random.random(1)
np.random.randn(15)

# Task 19 - create a vector with values ranging from 15 to 55 and print all values except the first and last
ar7 = np.arange(start = 15, stop = 55)
print(ar7[1:len(ar7)])

# Task 22 - create a vector with values from 0 to 20 and change the sign of the numbers in the range from 9 to 15
ar8 = np.array(range(0,21))
ar8[9:16] = ar8[9:16] * -1

# Task 25 - create a 3x4 matrix filled with values from 10 to 21
np.reshape(np.array(object = range(10,22), ndmin = 2, order = 'C'), newshape  = (3,4))

# Task 28 - create a 10x10 matrix, in which the elements on the borders will be equal to 1, and inside 0
ar_zero = np.array(object = np.repeat(0,10*10), ndmin = 2)
ar_zero = np.reshape(newshape = (10,10), a = ar_zero)

ar_one = np.array(object = np.repeat(1,10))

ar_zero[0,0:10] = ar_one
ar_zero[9,0:10] = ar_one
ar_zero[0:10,0] = ar_one
ar_zero[0:10,9] = ar_one

##### Sorting ######

# 1. Sort a given array of shape 2 along the first axis, last axis and on flattened array

def sorter(what, how):
    if how.lower() == 'flattened':
        return (np.reshape(np.sort(a = what, axis = None), newshape = (np.shape(what))))
    if how.lower() == 'first':
        return (np.sort(a = what, axis=0))
    if how.lower() == 'last':
        return (np.sort(a = what, axis=-1))

sorter(ar_zero, 'last')
